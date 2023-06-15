#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:59:26 2023.

@author: ghiggi
"""
import numpy as np

try:
    from bottleneck import nanrankdata as rankdata
except ImportError:
    from scipy.stats import rankdata


# import scipy.stats
# arr = np.arange(0, 100_000_000).reshape(10_000,10_000)
# %timeit scipy.stats.rankdata(arr, axis=1)
# %timeit bottleneck.rankdata(arr, axis=1)


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
    # Compute the means of x and y
    # mean_x = np.mean(x, axis=1)
    # mean_y = np.mean(y, axis=1)

    # # Compute the standard deviations of x and y
    # std_x = np.std(x, axis=1)
    # std_y = np.std(y, axis=1)

    mean_x = np.expand_dims(mean_x, axis=1)
    mean_y = np.expand_dims(mean_y, axis=1)

    # Compute the covariance between x and y
    cov = np.mean((x - mean_x) * (y - mean_y), axis=1)

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
    # Compute the means of x and y
    mean_x = np.mean(x, axis=1)
    mean_y = np.mean(y, axis=1)

    # Compute the standard deviations of x and y
    std_x = np.std(x, axis=1)
    std_y = np.std(y, axis=1)

    # Compute Pearson correlation cofficient
    corr = __pearson_corr_coeff(x=x, y=y,
                                mean_x=mean_x,
                                mean_y=mean_y,
                                std_x=std_x,
                                std_y=std_y)
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
    # Compute rank data
    x_rank = rankdata(x, axis=1)
    y_rank = rankdata(y, axis=1)

    # Compute Spearman correlation cofficient
    corr = _pearson_corr_coeff(x=x_rank, y=y_rank)
    return corr

