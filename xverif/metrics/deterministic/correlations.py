#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:59:26 2023.

@author: ghiggi
"""
import numpy as np

# FutureWarning: The `numpy.argsort` function is not implemented by Dask array. 
# You may want to use the da.map_blocks function or something similar to silence this warning. 
# Your code may stop working in a future release.


def np_rankdata(x: np.ndarray, axis: int=-1):
    """Rank data using numpy.

    Avoids using argsort 2 times.
    """
    in_shape = x.shape
    tmp = np.argsort(x, axis=-1).reshape(-1, in_shape[axis])
    rank = np.empty_like(tmp, dtype="float64")
    np.put_along_axis(rank, tmp, np.arange(1, in_shape[axis] + 1), axis=axis)
    del tmp
    rank = rank.reshape(in_shape)
    return rank


# try:
#     from bottleneck import nanrankdata as rankdata
# except ImportError:
#     from scipy.stats import rankdata

### Ranks benchmarks
# import bottleneck
# import scipy.stats
# arr = np.arange(0, 100_000_000).reshape(10_000,10_000)

# %timeit arr.argsort(axis=1).argsort(axis=1) + 1
# %timeit np_rankdata(arr, axis=1)
# %timeit bottleneck.rankdata(arr, axis=1)
# %timeit scipy.stats.rankdata(arr, axis=1)

# x = np.array([np.nan, np.nan, 5, 4])
# x.argsort().argsort() + 1
# np_rankdata(x)
# bottleneck.rankdata(x)
# bottleneck.nanrankdata(x)
# scipy.stats.rankdata(x)

# pvalues
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L319
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L374
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L460
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L503


def __pearson_corr_coeff(x, y,
                         mean_x, mean_y,
                         std_x, std_y):
    """
    Compute the Pearson Correlation Coefficient between pairwise columns.

    To be used on x and y 2D arrays with samples across the columns,
    and variables across the rows.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (aux, sample)
    y : np.ndarray
        2D array with shape (aux, sample)
    mean_x : np.ndarray
        Mean 2D array with shape (aux,)
    mean_y : np.ndarray
        Mean 2D array with shape (aux,)
    std_x : np.ndarray
        Standard deviation 2D array with shape (aux,)
    std_y : np.ndarray
        Standard deviation 2D array with shape (aux,)

    Returns
    -------
    corr : np.ndarray
        Correlation coefficient with shape (aux,)

    """
    mean_x = np.expand_dims(mean_x, axis=1)
    mean_y = np.expand_dims(mean_y, axis=1)

    # Compute the covariance between x and y
    cov = np.nanmean((x - mean_x) * (y - mean_y), axis=1)

    # Compute the correlation coefficient
    corr = cov / (std_x * std_y)

    # Clip to avoid numerical artefacts
    corr = np.clip(corr, -1, 1)

    return corr


def _pearson_corr_coeff(x, y):
    """
    Compute the Pearson Correlation Coefficient between pairwise columns.

    To be used on x and y 2D arrays with samples across the columns,
    and variables across the rows.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (aux, sample)
    y : np.ndarray
        2D array with shape (aux, sample)

    Returns
    -------
    corr : np.ndarray
        Correlation coefficient with shape (aux,)

    """
    corr = __pearson_corr_coeff(x=x, y=y,
                                mean_x=np.nanmean(x, axis=1),
                                mean_y=np.nanmean(y, axis=1),
                                std_x=np.nanstd(x, axis=1),
                                std_y=np.nanstd(y, axis=1))
    return corr


def _spearman_corr_coeff(x, y):
    """
    Compute the Spearman Correlation Coefficient between pairwise columns.

    To be used on x and y 2D arrays with samples across the columns,
    and variables across the rows.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (aux, sample)
    y : np.ndarray
        2D array with shape (aux, sample)

    Returns
    -------
    corr : np.ndarray
        Correlation coefficient with shape (aux,)

    """
    # TODO: implement np.nan_rankdata (now is wrong)
    corr = _pearson_corr_coeff(x=np_rankdata(x, axis=1),
                               y=np_rankdata(y, axis=1))
    return corr

