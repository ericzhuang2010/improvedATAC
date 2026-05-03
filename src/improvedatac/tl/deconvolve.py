"""High-level deconvolution entry point.

Controls the full pipeline: reference signature estimation →
peak scoring → model training → posterior export.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from ..model._reference import compute_reference_signatures
from ..model._spatial_model import ATACDeconvModel
from ..pp.peak_scoring import compute_peak_scores


def deconvolve(
    adata_spatial: AnnData,
    adata_ref: AnnData,
    cluster_key: str,
    *,
    layer_spatial: Optional[str] = None,
    layer_ref: Optional[str] = None,
    N_cells_per_location: float = 8.0,
    detection_alpha: float = 20.0,
    peak_weight_concentration: float = 5.0,
    peak_bias_concentration: float = 10.0,
    abundance_mean_var_ratio: float = 5.0,
    likelihood: str = "poisson",
    use_peak_scores: bool = True,
    n_top_peaks: Optional[int] = None,
    shrinkage: float = 0.0,
    max_epochs: int = 30000,
    lr: float = 0.005,
    batch_size: Optional[int] = None,
    use_gpu: bool = True,
    num_posterior_samples: int = 1000,
    results_path: str = "./improvedatac_results",
    return_model: bool = False,
) -> dict:
    r"""Run the full improved-ATAC deconvolution pipeline.

    **Step 1** -- Compute reference accessibility signatures
    :math:`\theta_{c,p}` from the scATAC reference.

    **Step 2** -- (Optional) Score peaks by cell-type specificity,
    reproducibility, and spatial variability.

    **Step 3** -- Train an ATAC-native Poisson deconvolution model with
    learned peak weights.

    **Step 4** -- Export posterior cell-type abundances and peak weights.

    Parameters
    ----------
    adata_spatial
        Spatial ATAC AnnData (spots × peaks).
    adata_ref
        Reference scATAC AnnData with cell-type annotations.
    cluster_key
        Column in ``adata_ref.obs`` containing cell-type labels.
    layer_spatial / layer_ref
        Optional layers for counts.
    N_cells_per_location
        Expected total cells per spot.
    detection_alpha
        Spot-level detection efficiency prior concentration.
    peak_weight_concentration
        Concentration for the peak-weight Gamma prior.
    peak_bias_concentration
        Concentration for the peak-bias Gamma prior.
    abundance_mean_var_ratio
        Mean / variance ratio for abundance prior.
    likelihood
        ``"poisson"`` (default), ``"nb"``, or ``"zip"``.
    use_peak_scores
        Whether to compute and incorporate peak informativeness scores
        into the model prior.
    n_top_peaks
        If set, filter to the top *n_top_peaks* before modelling.
    shrinkage
        Shrinkage factor for reference signatures (0–1).
    max_epochs / lr / batch_size
        SVI training hyper-parameters.
    use_gpu
        Use CUDA if available.
    num_posterior_samples
        Number of posterior draws for summarisation.
    results_path
        Directory for CSV outputs.
    return_model
        If ``True``, include the trained :class:`ATACDeconvModel`
        in the returned dictionary.

    Returns
    -------
    Dictionary with keys ``"proportions"``, ``"results"``,
    ``"peak_scores"``, ``"cell_state_df"``, and optionally ``"model"``.
    """

    # ---- align feature space ----
    common_peaks = adata_spatial.var_names.intersection(adata_ref.var_names)
    if len(common_peaks) == 0:
        raise ValueError("No overlapping peaks between spatial and reference data.")
    print(f"[improvedATAC] {len(common_peaks)} peaks shared between spatial and reference.")
    adata_ref_sub = adata_ref[:, common_peaks].copy()
    adata_spatial_sub = adata_spatial[:, common_peaks].copy()

    # ---- Step 1: reference signatures ----
    print("[improvedATAC] Computing reference signatures …")
    cell_state_df = compute_reference_signatures(
        adata_ref_sub,
        cluster_key=cluster_key,
        layer=layer_ref,
        shrinkage=shrinkage,
    )

    # ---- Step 2: peak scoring ----
    peak_scores: Optional[np.ndarray] = None
    if use_peak_scores:
        print("[improvedATAC] Scoring peaks …")
        peak_scores = compute_peak_scores(
            adata_ref_sub,
            cluster_key=cluster_key,
            adata_spatial=adata_spatial_sub,
            layer=layer_ref,
            n_top_peaks=n_top_peaks,
        )

    if n_top_peaks is not None and "informative_peak" in adata_ref_sub.var.columns:
        sel = adata_ref_sub.var["informative_peak"]
        adata_spatial_sub = adata_spatial_sub[:, sel].copy()
        cell_state_df = cell_state_df.loc[sel]
        if peak_scores is not None:
            peak_scores = peak_scores[sel.values]
        print(f"[improvedATAC] Filtered to {adata_spatial_sub.n_vars} informative peaks.")

    # ---- Step 3: build & train model ----
    print("[improvedATAC] Building model …")
    model = ATACDeconvModel(
        adata_spatial=adata_spatial_sub,
        cell_state_df=cell_state_df,
        peak_scores=peak_scores,
        layer=layer_spatial,
        N_cells_per_location=N_cells_per_location,
        detection_alpha=detection_alpha,
        peak_weight_concentration=peak_weight_concentration,
        peak_bias_concentration=peak_bias_concentration,
        abundance_mean_var_ratio=abundance_mean_var_ratio,
        likelihood=likelihood,
        use_gpu=use_gpu,
    )

    print("[improvedATAC] Training …")
    model.train(
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
    )

    # ---- Step 4: export posterior ----
    print("[improvedATAC] Exporting posterior …")
    results = model.export_posterior(num_samples=num_posterior_samples)

    proportions = model.get_proportions(num_samples=num_posterior_samples)

    # ---- save CSV outputs ----
    os.makedirs(results_path, exist_ok=True)
    results["q05_cell_abundance_w_sf"].to_csv(
        os.path.join(results_path, "q05_cell_abundance_w_sf.csv")
    )
    results["means_cell_abundance_w_sf"].to_csv(
        os.path.join(results_path, "means_cell_abundance_w_sf.csv")
    )
    proportions.to_csv(os.path.join(results_path, "proportions.csv"))
    results["means_peak_weights"].to_csv(
        os.path.join(results_path, "peak_weights.csv")
    )

    # ---- store in adata ----
    adata_spatial.obsm["q05_cell_abundance_w_sf"] = _align_to_original(
        results["q05_cell_abundance_w_sf"], adata_spatial, adata_spatial_sub
    )
    adata_spatial.obsm["means_cell_abundance_w_sf"] = _align_to_original(
        results["means_cell_abundance_w_sf"], adata_spatial, adata_spatial_sub
    )
    adata_spatial.obsm["proportions"] = _align_to_original(
        proportions, adata_spatial, adata_spatial_sub
    )

    print(f"[improvedATAC] Done. Results saved to {results_path}/")

    out = {
        "proportions": proportions,
        "results": results,
        "peak_scores": peak_scores,
        "cell_state_df": cell_state_df,
    }
    if return_model:
        out["model"] = model
    return out


def _align_to_original(
    df: pd.DataFrame,
    adata_full: AnnData,
    adata_sub: AnnData,
) -> pd.DataFrame:
    """Align result DataFrame back to the original adata obs index."""
    return df.reindex(adata_full.obs_names).fillna(0)
