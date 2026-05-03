"""Evaluation metrics for spatial deconvolution."""

from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def rmse(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> float:
    """Root mean squared error between true and predicted proportions.

    Parameters
    ----------
    true
        Ground-truth proportions.
    predicted
        Estimated proportions (same shape).

    Returns
    -------
    Scalar RMSE.
    """
    return float(np.sqrt(np.mean((np.asarray(true) - np.asarray(predicted)) ** 2)))


def jsd(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> float:
    """Mean Jensen-Shannon divergence across samples.

    Parameters
    ----------
    true
        Ground-truth proportions (samples × cell types).
    predicted
        Estimated proportions.

    Returns
    -------
    Mean JSD (base-2 logarithm).
    """
    d = jensenshannon(np.asarray(true), np.asarray(predicted), axis=1, base=2)
    return float(np.mean(d[np.isfinite(d)]))


def pearson_per_celltype(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Per-cell-type Pearson correlation across spots.

    Parameters
    ----------
    true
        Ground-truth proportions (spots × cell types).
    predicted
        Estimated proportions.

    Returns
    -------
    Array of Pearson *r* values, one per cell type.
    """
    true_arr = np.asarray(true)
    pred_arr = np.asarray(predicted)
    n_ct = true_arr.shape[1]
    corrs = np.array(
        [np.corrcoef(true_arr[:, i], pred_arr[:, i])[0, 1] for i in range(n_ct)]
    )
    return corrs


def spearman_per_celltype(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Per-cell-type Spearman rank correlation across spots.

    Parameters
    ----------
    true
        Ground-truth proportions (spots × cell types).
    predicted
        Estimated proportions.

    Returns
    -------
    Array of Spearman *rho* values, one per cell type.
    """
    from scipy.stats import spearmanr

    true_arr = np.asarray(true)
    pred_arr = np.asarray(predicted)
    n_ct = true_arr.shape[1]
    corrs = np.array(
        [spearmanr(true_arr[:, i], pred_arr[:, i]).statistic for i in range(n_ct)]
    )
    return corrs
