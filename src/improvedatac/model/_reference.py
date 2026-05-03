"""Reference accessibility signature estimation.

Computes per-cell-type mean accessibility profiles (theta_{c,p}) from
scATAC reference data.  These serve as the fixed basis matrix in the
spatial deconvolution model.
"""

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def compute_reference_signatures(
    adata_ref: AnnData,
    cluster_key: str,
    layer: Optional[str] = None,
    shrinkage: float = 0.0,
) -> pd.DataFrame:
    r"""Compute per-cell-type accessibility profiles from scATAC reference.

    For each cell type *c* and peak *p*:

    .. math::
        \theta_{c,p} = \frac{1}{|C_c|} \sum_{i \in C_c} x_{i,p}

    where :math:`C_c` is the set of cells belonging to cell type *c*.

    Parameters
    ----------
    adata_ref
        Reference scATAC AnnData with raw counts.
    cluster_key
        Column in ``adata_ref.obs`` with cell type labels.
    layer
        Layer to read counts from. ``None`` uses ``.X``.
    shrinkage
        Shrinkage toward the global mean (0 = no shrinkage, 1 = full shrinkage).
        Useful when some clusters have very few cells.

    Returns
    -------
    DataFrame of shape ``(n_peaks, n_celltypes)`` with cluster-mean
    accessibility values.  Index matches ``adata_ref.var_names``.
    """
    X = adata_ref.layers[layer] if layer else adata_ref.X
    clusters = adata_ref.obs[cluster_key]
    unique_clusters = sorted(clusters.unique())

    signatures = {}
    for clust in unique_clusters:
        mask = (clusters == clust).values
        X_clust = X[mask]
        if sp.issparse(X_clust):
            mean_vec = np.asarray(X_clust.mean(axis=0)).ravel()
        else:
            mean_vec = np.asarray(X_clust).mean(axis=0).ravel()
        signatures[clust] = mean_vec

    df = pd.DataFrame(signatures, index=adata_ref.var_names)

    if shrinkage > 0:
        global_mean = df.mean(axis=1)
        df = (1 - shrinkage) * df + shrinkage * global_mean.values[:, None]

    return df


def compute_normalized_signatures(
    adata_ref: AnnData,
    cluster_key: str,
    layer: Optional[str] = None,
    shrinkage: float = 0.0,
    scale: float = 1e4,
) -> pd.DataFrame:
    r"""Library-size-normalized reference signatures.

    For each cell type *c* and peak *p*:

    .. math::
        \theta^{\text{norm}}_{c,p} =
        \frac{1}{|C_c|}\sum_{i \in C_c} \frac{x_{i,p}}{\ell_i} \cdot s

    where :math:`\ell_i` is the library size of cell *i* and *s* is a
    scaling constant (default 10 000).

    Parameters
    ----------
    adata_ref
        Reference scATAC AnnData.
    cluster_key
        Cell type column in ``adata_ref.obs``.
    layer
        Layer for counts.
    shrinkage
        Shrinkage toward global mean.
    scale
        Multiplicative scale factor applied after library-size normalisation.

    Returns
    -------
    DataFrame ``(n_peaks, n_celltypes)``.
    """
    X = adata_ref.layers[layer] if layer else adata_ref.X
    clusters = adata_ref.obs[cluster_key]
    unique_clusters = sorted(clusters.unique())

    if sp.issparse(X):
        lib_sizes = np.asarray(X.sum(axis=1)).ravel()
    else:
        lib_sizes = np.asarray(X.sum(axis=1)).ravel()

    signatures = {}
    for clust in unique_clusters:
        mask = (clusters == clust).values
        X_clust = X[mask]
        ls = lib_sizes[mask][:, None] + 1e-8

        if sp.issparse(X_clust):
            X_norm = X_clust.multiply(scale / ls)
            mean_vec = np.asarray(X_norm.mean(axis=0)).ravel()
        else:
            X_norm = np.asarray(X_clust) * (scale / ls)
            mean_vec = X_norm.mean(axis=0).ravel()

        signatures[clust] = mean_vec

    df = pd.DataFrame(signatures, index=adata_ref.var_names)

    if shrinkage > 0:
        global_mean = df.mean(axis=1)
        df = (1 - shrinkage) * df + shrinkage * global_mean.values[:, None]

    return df
