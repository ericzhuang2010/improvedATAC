r"""ATAC-native probabilistic deconvolution Pyro module.

Implements the generative model:

.. math::

    y_{s,p} \sim \text{Poisson}(\mu_{s,p})

    \mu_{s,p} = d_s \cdot b_p \cdot \sum_c a_{s,c}\, w_p\, \theta_{c,p}

where:

- :math:`d_s`  -- spot-level sequencing depth (Gamma prior centred on
  the observed library size)
- :math:`b_p`  -- peak-specific technical bias (GC content, mappability,
  Tn5 insertion; Gamma prior centred at 1)
- :math:`a_{s,c}` -- abundance of cell type *c* in spot *s*
  (Gamma prior controlled by ``N_cells_per_location``)
- :math:`w_p`  -- learned peak informativeness weight (Gamma prior whose
  mean is set by pre-computed peak scores)
- :math:`\theta_{c,p}` -- reference accessibility profile (fixed)

The model supports three observation likelihoods via the ``likelihood``
parameter: ``"poisson"`` (default, ATAC-native), ``"nb"``
(negative-binomial / GammaPoisson for comparison), and ``"zip"``
(zero-inflated Poisson for extremely sparse data).
"""

from typing import Optional

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class ATACDeconvModule(PyroModule):
    """Pyro module for ATAC spatial deconvolution.

    Parameters
    ----------
    n_obs
        Number of spatial spots.
    n_peaks
        Number of chromatin accessibility peaks.
    n_celltypes
        Number of reference cell types.
    cell_state_mat
        Reference profiles :math:`\\theta`, shape ``(n_celltypes, n_peaks)``.
    library_sizes
        Per-spot total fragment counts, shape ``(n_obs,)``.
    peak_scores
        Per-peak informativeness scores, shape ``(n_peaks,)``.
        Sets the prior mean for learned peak weights :math:`w_p`.
        If ``None``, a flat prior (mean=1) is used.
    N_cells_per_location
        Expected total cell abundance per spot.
    detection_alpha
        Concentration for the spot detection-efficiency Gamma prior.
        Larger values shrink :math:`d_s` toward the observed library size.
    peak_weight_concentration
        Gamma concentration for :math:`w_p`.  Larger values shrink
        weights toward the peak-score prior mean.
    peak_bias_concentration
        Gamma concentration for :math:`b_p`.  Larger values keep the
        bias closer to 1.
    abundance_mean_var_ratio
        Mean-to-variance ratio for the cell-abundance Gamma prior.
    likelihood
        One of ``"poisson"`` (default), ``"nb"``, or ``"zip"``.
    """

    def __init__(
        self,
        n_obs: int,
        n_peaks: int,
        n_celltypes: int,
        cell_state_mat: np.ndarray,
        library_sizes: np.ndarray,
        peak_scores: Optional[np.ndarray] = None,
        N_cells_per_location: float = 8.0,
        detection_alpha: float = 20.0,
        peak_weight_concentration: float = 5.0,
        peak_bias_concentration: float = 10.0,
        abundance_mean_var_ratio: float = 5.0,
        likelihood: str = "poisson",
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_peaks = n_peaks
        self.n_celltypes = n_celltypes
        self.likelihood = likelihood

        # Reference accessibility θ_{c,p}
        self.register_buffer(
            "cell_state",
            torch.tensor(cell_state_mat, dtype=torch.float32),
        )  # (n_celltypes, n_peaks)

        # Library sizes  → (n_obs, 1)
        lib = np.asarray(library_sizes, dtype=np.float64).ravel()
        lib = lib / np.median(lib)  # relative scale
        self.register_buffer(
            "lib_sizes",
            torch.tensor(lib, dtype=torch.float32).unsqueeze(-1),
        )

        # Peak weight prior:  w_p ~ Gamma(conc*score, conc) → mean = score
        if peak_scores is not None:
            scores = np.asarray(peak_scores, dtype=np.float64).ravel()
            scores = np.clip(scores, 1e-6, None)
        else:
            scores = np.ones(n_peaks, dtype=np.float64)
        w_conc = float(peak_weight_concentration)
        self.register_buffer(
            "w_alpha",
            torch.tensor(w_conc * scores, dtype=torch.float32).unsqueeze(0),
        )  # (1, n_peaks)
        self.register_buffer(
            "w_beta",
            torch.full((1, n_peaks), w_conc, dtype=torch.float32),
        )

        # Peak bias prior:  b_p ~ Gamma(α, α) → mean = 1
        self.register_buffer(
            "b_conc",
            torch.tensor(float(peak_bias_concentration)),
        )

        # Spot detection prior:  d_s ~ Gamma(α, α / lib) → mean ≈ lib
        self.register_buffer("det_alpha", torch.tensor(float(detection_alpha)))

        # Cell abundance prior:  a ~ Gamma(mean·mvr, mvr)
        a_mean = N_cells_per_location / n_celltypes
        mvr = float(abundance_mean_var_ratio)
        self.register_buffer("a_alpha", torch.tensor(a_mean * mvr, dtype=torch.float32))
        self.register_buffer("a_beta", torch.tensor(mvr, dtype=torch.float32))

        # Helpers
        self.register_buffer("eps", torch.tensor(1e-8))
        self.register_buffer("ones_11", torch.ones(1, 1))

    # ------------------------------------------------------------------ #
    #  Pyro model interface                                                #
    # ------------------------------------------------------------------ #

    def create_plates(self, x_data, idx):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """Observation-plate variable catalogue (for optional amortised guide)."""
        return {
            "name": "obs_plate",
            "sites": {
                "d_s": 1,
                "a_sc": self.n_celltypes,
            },
        }

    def forward(self, x_data, idx):
        obs_plate = self.create_plates(x_data, idx)

        # =========== Peak-level (global) latent variables =========== #

        # Peak bias  b_p  ~  Gamma(α, α)   →  E[b_p] = 1
        b_p = pyro.sample(
            "b_p",
            dist.Gamma(self.b_conc, self.b_conc)
            .expand([1, self.n_peaks])
            .to_event(2),
        )  # (1, n_peaks)

        # Peak weight  w_p  ~  Gamma(α·score, α)   →  E[w_p] = score
        w_p = pyro.sample(
            "w_p",
            dist.Gamma(self.w_alpha, self.w_beta).to_event(2),
        )  # (1, n_peaks)

        # =========== Spot-level latent variables =========== #

        with obs_plate as ind:
            # Spot detection efficiency  d_s ~ Gamma(α, α/lib)
            lib_sz = self.lib_sizes[ind]  # (batch, 1)
            d_s = pyro.sample(
                "d_s",
                dist.Gamma(
                    self.det_alpha * self.ones_11,
                    self.det_alpha / (lib_sz + self.eps),
                ),
            )  # (batch, 1)

            # Cell-type abundances  a_{s,c} ~ Gamma(α, β)
            a_sc = pyro.sample(
                "a_sc",
                dist.Gamma(self.a_alpha, self.a_beta).expand([self.n_celltypes]),
            )  # (batch, n_celltypes)

        # =========== Expected counts & observation model =========== #

        # μ_{s,p} = d_s · Σ_c  a_{s,c}·w_p·b_p·θ_{c,p}
        weighted_ref = self.cell_state * w_p * b_p  # (n_celltypes, n_peaks)

        with obs_plate:
            mu = d_s * (a_sc @ weighted_ref) + self.eps  # (batch, n_peaks)

            if self.likelihood == "poisson":
                pyro.sample("data_target", dist.Poisson(mu), obs=x_data)
            elif self.likelihood == "nb":
                alpha_inv = pyro.sample(
                    "alpha_inv",
                    dist.Exponential(torch.ones(1, device=mu.device))
                    .expand([1, self.n_peaks])
                    .to_event(2),
                )
                alpha = 1.0 / (alpha_inv.pow(2) + self.eps)
                pyro.sample(
                    "data_target",
                    dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                    obs=x_data,
                )
            elif self.likelihood == "zip":
                gate_logit = pyro.sample(
                    "gate_logit",
                    dist.Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
                    .expand([1, self.n_peaks])
                    .to_event(2),
                )
                pyro.sample(
                    "data_target",
                    dist.ZeroInflatedPoisson(rate=mu, gate_logits=gate_logit),
                    obs=x_data,
                )
            else:
                raise ValueError(f"Unknown likelihood: {self.likelihood!r}")

        # Monitoring
        with obs_plate:
            pyro.deterministic("mu_sp", mu)
