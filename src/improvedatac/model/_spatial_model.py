"""High-level ATAC deconvolution model.

Wraps :class:`ATACDeconvModule` with data handling, Pyro SVI training,
posterior export, and cell-type proportion estimation.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import scipy.sparse as sp
import torch
from anndata import AnnData
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_feasible
from tqdm import trange

from ._spatial_module import ATACDeconvModule


class ATACDeconvModel:
    """Train and query an ATAC-native spatial deconvolution model.

    Parameters
    ----------
    adata_spatial
        Spatial ATAC AnnData (spots × peaks).  Feature names must
        match the columns of *cell_state_df*.
    cell_state_df
        Reference accessibility profiles (peaks × cell types).
        Typically produced by
        :func:`~improvedatac.model.compute_reference_signatures`.
    peak_scores
        Per-peak informativeness scores (length ``n_peaks``).
        If ``None``, a flat prior is placed on peak weights.
    layer
        Which layer of *adata_spatial* to use.  ``None`` → ``.X``.
    N_cells_per_location
        Expected total cell count per spatial spot.
    detection_alpha
        Concentration for spot-level detection-efficiency prior.
    peak_weight_concentration
        Concentration for the peak-weight Gamma prior.
    peak_bias_concentration
        Concentration for the peak-bias Gamma prior.
    abundance_mean_var_ratio
        Mean / variance ratio for cell-type abundance prior.
    likelihood
        ``"poisson"`` (default), ``"nb"``, or ``"zip"``.
    use_gpu
        Whether to move tensors to a CUDA device (if available).
    """

    def __init__(
        self,
        adata_spatial: AnnData,
        cell_state_df: pd.DataFrame,
        peak_scores: Optional[np.ndarray] = None,
        layer: Optional[str] = None,
        N_cells_per_location: float = 8.0,
        detection_alpha: float = 20.0,
        peak_weight_concentration: float = 5.0,
        peak_bias_concentration: float = 10.0,
        abundance_mean_var_ratio: float = 5.0,
        likelihood: str = "poisson",
        use_gpu: bool = True,
    ):
        # ---- validate feature alignment ----
        common = adata_spatial.var_names.intersection(cell_state_df.index)
        if len(common) == 0:
            raise ValueError("No overlapping features between spatial data and reference signatures.")
        if len(common) < len(adata_spatial.var_names):
            print(
                f"[improvedATAC] Using {len(common)} / {adata_spatial.n_vars} "
                f"peaks present in both spatial and reference data."
            )
            adata_spatial = adata_spatial[:, common].copy()

        # ---- extract count matrix ----
        X = adata_spatial.layers[layer] if layer else adata_spatial.X
        if sp.issparse(X):
            X_dense = np.asarray(X.toarray(), dtype=np.float32)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
        self._X = torch.tensor(X_dense)

        # ---- cell state matrix (n_celltypes × n_peaks) ----
        cell_state_aligned = cell_state_df.loc[common]
        cell_state_mat = cell_state_aligned.values.T.astype(np.float32)

        # ---- library sizes ----
        library_sizes = X_dense.sum(axis=1)

        # ---- peak scores ----
        if peak_scores is not None:
            peak_scores_aligned = np.asarray(peak_scores, dtype=np.float64)
            if len(peak_scores_aligned) != len(common):
                score_idx = [adata_spatial.var_names.get_loc(p) for p in common]
                peak_scores_aligned = peak_scores_aligned[score_idx]
        else:
            peak_scores_aligned = None

        # ---- device ----
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # ---- Pyro module ----
        self.module = ATACDeconvModule(
            n_obs=adata_spatial.n_obs,
            n_peaks=len(common),
            n_celltypes=cell_state_aligned.shape[1],
            cell_state_mat=cell_state_mat,
            library_sizes=library_sizes,
            peak_scores=peak_scores_aligned,
            N_cells_per_location=N_cells_per_location,
            detection_alpha=detection_alpha,
            peak_weight_concentration=peak_weight_concentration,
            peak_bias_concentration=peak_bias_concentration,
            abundance_mean_var_ratio=abundance_mean_var_ratio,
            likelihood=likelihood,
        ).to(self.device)

        self._X = self._X.to(self.device)

        # ---- guide ----
        pyro.clear_param_store()
        self.guide = AutoNormal(
            self.module,
            init_loc_fn=init_to_feasible,
        )

        # ---- metadata ----
        self.adata = adata_spatial
        self.cell_type_names = list(cell_state_aligned.columns)
        self.peak_names = list(common)
        self.history: list[float] = []
        self.is_trained = False

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(
        self,
        max_epochs: int = 30000,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
        num_particles: int = 1,
        log_every: int = 1000,
        convergence_window: int = 500,
        convergence_threshold: float = 1e-5,
    ):
        """Run stochastic variational inference.

        Parameters
        ----------
        max_epochs
            Maximum number of full passes over the dataset.
        lr
            Initial learning rate for :class:`pyro.optim.ClippedAdam`.
        batch_size
            Mini-batch size.  ``None`` → ``min(2048, n_obs)``.
        num_particles
            Number of ELBO particles (samples) per step.
        log_every
            Print progress every *log_every* epochs.
        convergence_window
            Number of recent epochs to assess convergence over.
        convergence_threshold
            Relative ELBO change threshold for early stopping.
        """
        if batch_size is None:
            batch_size = min(2048, self._X.shape[0])

        optimizer = pyro.optim.ClippedAdam({"lr": lr, "lrd": 1 - 1e-5})
        elbo = Trace_ELBO(num_particles=num_particles)
        svi = SVI(self.module, self.guide, optimizer, loss=elbo)

        n_obs = self._X.shape[0]

        pbar = trange(max_epochs, desc="Training", leave=True)
        for epoch in pbar:
            indices = torch.randperm(n_obs, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_obs, batch_size):
                idx = indices[start : start + batch_size]
                x_batch = self._X[idx]
                loss = svi.step(x_batch, idx)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.history.append(avg_loss)

            if (epoch + 1) % log_every == 0:
                pbar.set_postfix({"ELBO": f"{avg_loss:.1f}"})

            if len(self.history) > convergence_window:
                old = self.history[-convergence_window]
                new = self.history[-1]
                if abs(new - old) / (abs(old) + 1e-12) < convergence_threshold:
                    pbar.set_postfix({"ELBO": f"{avg_loss:.1f}", "status": "converged"})
                    pbar.close()
                    print(f"Converged at epoch {epoch + 1}")
                    break

        self.is_trained = True

    # ------------------------------------------------------------------ #
    #  Posterior export                                                     #
    # ------------------------------------------------------------------ #

    def export_posterior(
        self,
        num_samples: int = 1000,
        batch_size: int = 512,
    ) -> dict:
        """Sample from the approximate posterior and summarise.

        Parameters
        ----------
        num_samples
            Number of posterior draws.
        batch_size
            When sampling, process this many draws at a time to limit
            peak GPU memory.

        Returns
        -------
        Dictionary with keys:

        - ``means_cell_abundance_w_sf`` / ``q05_cell_abundance_w_sf`` /
          ``q95_cell_abundance_w_sf`` / ``stds_cell_abundance_w_sf`` --
          DataFrames (spots × cell types).
        - ``means_peak_weights`` / ``q05_peak_weights`` -- 1-d arrays.
        - ``means_peak_bias`` -- 1-d array.
        """
        self.module.eval()

        idx_all = torch.arange(self._X.shape[0], device=self.device)
        x_all = self._X

        a_sc_samples = []
        w_p_samples = []
        b_p_samples = []

        remaining = num_samples
        while remaining > 0:
            n = min(batch_size, remaining)
            predictive = pyro.infer.Predictive(
                self.module,
                guide=self.guide,
                num_samples=n,
                return_sites=["a_sc", "w_p", "b_p", "d_s"],
            )
            with torch.no_grad():
                samples = predictive(x_all, idx_all)

            a_sc_samples.append(samples["a_sc"].cpu().numpy())
            w_p_samples.append(samples["w_p"].cpu().numpy())
            b_p_samples.append(samples["b_p"].cpu().numpy())
            remaining -= n

        a_sc = np.concatenate(a_sc_samples, axis=0)  # (S, n_obs, n_celltypes)
        w_p = np.concatenate(w_p_samples, axis=0)  # (S, 1, n_peaks)
        b_p = np.concatenate(b_p_samples, axis=0)

        results = {}

        for stat_name, fn in [("means", np.mean), ("stds", np.std)]:
            results[f"{stat_name}_cell_abundance_w_sf"] = pd.DataFrame(
                fn(a_sc, axis=0),
                index=self.adata.obs_names,
                columns=self.cell_type_names,
            )
        for q, q_name in [(0.05, "q05"), (0.95, "q95")]:
            results[f"{q_name}_cell_abundance_w_sf"] = pd.DataFrame(
                np.quantile(a_sc, q, axis=0),
                index=self.adata.obs_names,
                columns=self.cell_type_names,
            )

        results["means_peak_weights"] = pd.Series(
            np.mean(w_p, axis=0).squeeze(), index=self.peak_names
        )
        results["q05_peak_weights"] = pd.Series(
            np.quantile(w_p, 0.05, axis=0).squeeze(), index=self.peak_names
        )
        results["means_peak_bias"] = pd.Series(
            np.mean(b_p, axis=0).squeeze(), index=self.peak_names
        )

        self.module.train()
        return results

    def get_proportions(
        self,
        num_samples: int = 1000,
        quantile: Optional[str] = None,
    ) -> pd.DataFrame:
        """Normalise posterior abundances to cell-type proportions.

        Parameters
        ----------
        num_samples
            Posterior draws used to compute the summary.
        quantile
            ``None`` uses posterior means; ``"q05"`` or ``"q95"`` use
            the corresponding quantile.

        Returns
        -------
        DataFrame (spots × cell types) summing to 1 per spot.
        """
        results = self.export_posterior(num_samples=num_samples)
        key = "means_cell_abundance_w_sf" if quantile is None else f"{quantile}_cell_abundance_w_sf"
        abundances = results[key]
        row_sums = abundances.sum(axis=1).replace(0, 1)
        return abundances.div(row_sums, axis=0)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def plot_history(self, skip: int = 0):
        """Plot ELBO training loss.

        Parameters
        ----------
        skip
            Number of initial epochs to omit (they are often very large
            and compress the remaining curve).
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(skip, len(self.history)), self.history[skip:])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ELBO loss")
        ax.set_title("Training convergence")
        fig.tight_layout()
        return fig

    def plot_peak_weights(self, top_n: int = 30):
        """Bar plot of the highest learned peak weights."""
        results = self.export_posterior(num_samples=200)
        w = results["means_peak_weights"].sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(w)), w.values[::-1])
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels(w.index[::-1], fontsize=7)
        ax.set_xlabel("Learned peak weight")
        ax.set_title(f"Top {top_n} informative peaks")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    #  Save / load helpers                                                 #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Persist model and guide parameters."""
        torch.save(
            {
                "module_state": self.module.state_dict(),
                "param_store": pyro.get_param_store().get_state(),
                "cell_type_names": self.cell_type_names,
                "peak_names": self.peak_names,
                "history": self.history,
            },
            path,
        )

    def load(self, path: str):
        """Restore previously saved parameters."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.module.load_state_dict(ckpt["module_state"])
        pyro.get_param_store().set_state(ckpt["param_store"])
        self.cell_type_names = ckpt["cell_type_names"]
        self.peak_names = ckpt["peak_names"]
        self.history = ckpt["history"]
        self.is_trained = True
