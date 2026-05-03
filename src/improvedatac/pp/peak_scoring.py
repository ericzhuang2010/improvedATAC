"""Peak scoring for ATAC deconvolution.

Computes informativeness scores for each chromatin peak, combining:
- Cell-type specificity (variance across clusters)
- Reference reproducibility (signal consistency within clusters)
- Spatial variability (variation across spatial locations)

Scores are used both for peak pre-filtering and as informative priors
on learned peak weights in the probabilistic model.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def compute_peak_scores(
    adata_ref: AnnData,
    cluster_key: str,
    adata_spatial: Optional[AnnData] = None,
    layer: Optional[str] = None,
    n_top_peaks: Optional[int] = None,
) -> np.ndarray:
    r"""Compute composite peak informativeness scores.

    .. math::
        \text{score}_p = \text{specificity}_p \times \text{reproducibility}_p
        \times \text{spatial\_variability}_p

    Parameters
    ----------
    adata_ref
        Reference scATAC AnnData with cell type annotations.
    cluster_key
        Column in ``adata_ref.obs`` containing cell type labels.
    adata_spatial
        Optional spatial ATAC AnnData. If provided, spatial variability
        is included in the composite score.
    layer
        Layer to use for counts. ``None`` uses ``.X``.
    n_top_peaks
        If set, marks the top scoring peaks in ``adata_ref.var["informative_peak"]``.

    Returns
    -------
    Scores array of shape ``(n_peaks,)``.  Also stores
    ``adata_ref.var["peak_score"]`` and per-component columns in-place.
    """
    specificity = _cell_type_specificity(adata_ref, cluster_key, layer)
    reproducibility = _reference_reproducibility(adata_ref, cluster_key, layer)

    scores = specificity * reproducibility

    if adata_spatial is not None:
        spatial_var = _spatial_variability(adata_spatial, layer)
        common = adata_ref.var_names.intersection(adata_spatial.var_names)
        ref_idx = np.array([adata_ref.var_names.get_loc(p) for p in common])
        sp_idx = np.array([adata_spatial.var_names.get_loc(p) for p in common])
        spatial_component = np.zeros(adata_ref.n_vars)
        spatial_component[ref_idx] = spatial_var[sp_idx]
        spatial_component[spatial_component == 0] = np.median(spatial_var)
        scores = scores * spatial_component

    scores = scores / (scores.max() + 1e-12)

    adata_ref.var["peak_score"] = scores
    adata_ref.var["peak_specificity"] = specificity / (specificity.max() + 1e-12)
    adata_ref.var["peak_reproducibility"] = reproducibility / (reproducibility.max() + 1e-12)

    if n_top_peaks is not None:
        n_top_peaks = min(n_top_peaks, len(scores))
        idx = np.argpartition(scores, -n_top_peaks)[-n_top_peaks:]
        adata_ref.var["informative_peak"] = adata_ref.var.index.isin(adata_ref.var.index[idx])

    return scores


def _get_matrix(adata: AnnData, layer: Optional[str] = None):
    X = adata.layers[layer] if layer else adata.X
    return X


def _cell_type_specificity(
    adata_ref: AnnData,
    cluster_key: str,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Variance of normalized accessibility across cell-type clusters.

    Higher variance means the peak discriminates between cell types.
    Closely follows the ArchR-style HVP computation from deconvATAC.
    """
    X = _get_matrix(adata_ref, layer)
    clusters = adata_ref.obs[cluster_key]
    unique_clusters = clusters.unique()

    group_means = []
    for clust in unique_clusters:
        mask = clusters == clust
        X_clust = X[mask]
        if sp.issparse(X_clust):
            clust_sum = np.asarray(X_clust.sum(axis=0)).ravel()
        else:
            clust_sum = np.asarray(X_clust.sum(axis=0)).ravel()
        total = clust_sum.sum()
        if total > 0:
            group_means.append(clust_sum / total)
        else:
            group_means.append(np.zeros(adata_ref.n_vars))

    group_matrix = np.stack(group_means, axis=0)
    group_matrix = np.log2(group_matrix * 1e4 + 1)

    specificity = np.var(group_matrix, axis=0)
    return specificity


def _reference_reproducibility(
    adata_ref: AnnData,
    cluster_key: str,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Inverse coefficient of variation within clusters, averaged across clusters.

    Peaks with consistent signal across cells within a cluster are more
    reproducible and therefore more reliable for deconvolution.
    """
    X = _get_matrix(adata_ref, layer)
    clusters = adata_ref.obs[cluster_key]
    unique_clusters = clusters.unique()

    inv_cvs = []
    for clust in unique_clusters:
        mask = clusters == clust
        X_clust = X[mask]
        if sp.issparse(X_clust):
            X_dense = np.asarray(X_clust.todense())
        else:
            X_dense = np.asarray(X_clust)

        means = X_dense.mean(axis=0) + 1e-8
        stds = X_dense.std(axis=0) + 1e-8
        inv_cvs.append(means / stds)

    inv_cv_matrix = np.stack(inv_cvs, axis=0)
    reproducibility = inv_cv_matrix.mean(axis=0)
    return reproducibility


def _spatial_variability(
    adata_spatial: AnnData,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Variance of accessibility across spatial spots.

    Peaks that vary spatially carry positional information useful for
    resolving cell-type mixtures.
    """
    X = _get_matrix(adata_spatial, layer)
    if sp.issparse(X):
        X_dense = np.asarray(X.todense())
    else:
        X_dense = np.asarray(X)

    lib_sizes = X_dense.sum(axis=1, keepdims=True) + 1e-8
    X_norm = X_dense / lib_sizes

    spatial_var = np.var(X_norm, axis=0)
    return spatial_var
