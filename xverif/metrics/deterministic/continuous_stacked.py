#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:38:26 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xverif import EPS
from xverif.metrics.deterministic.correlations import (
    __pearson_corr_coeff,
    _spearman_corr_coeff,
)
from xverif.utils.timing import print_elapsed_time


def get_aux_dims(pred, aggregating_dim):
    """Infer aux_dims from prediction dataset."""
    dims = list(pred.dims)
    aux_dims = set(dims).difference(aggregating_dim)
    return aux_dims


def get_stacking_dict(pred, aggregating_dim):
    """Get stacking dimension dictionary."""
    aux_dims = get_aux_dims(pred, aggregating_dim)
    stacking_dict =  {
        "aux": aux_dims,
        "sample": aggregating_dim,
    }
    return stacking_dict


def _get_metrics(pred, obs, **kwargs):
    """
    Deterministic metrics for continuous predictions forecasts.

    Parameters
    ----------
    pred : np.ndarray
        2D prediction array of shape (aux, sample)
        The columns corresponds to the sample predictions to be used
        to compute the metrics (for each row).
    obs : TYPE
        2D observation array with same shape as pred: (aux, sample).
        The columns corresponds to the sample observations to be used
        to compute the metrics (for each row).


    Returns
    -------
    skills : np.ndarray
        A 2D array of shape (aux, n_skills).

    """
    ##------------------------------------------------------------------------.
    # - Error
    error = pred - obs
    error_squared = error**2
    error_abs = np.abs(error)
    error_perc = error / (obs + EPS)

    ##------------------------------------------------------------------------.
    # - Mean
    pred_mean = pred.mean(axis=1)
    obs_mean = obs.mean(axis=1)
    error_mean = error.mean(axis=1)
    ##------------------------------------------------------------------------.
    # - Standard deviation
    pred_std = pred.std(axis=1)
    obs_std = obs.std(axis=1)
    error_std = error.std(axis=1)

    ##------------------------------------------------------------------------.
    # - Coefficient of variability
    pred_CoV = pred_std / (pred_mean + EPS)
    obs_CoV = obs_std / (obs_mean + EPS)
    error_CoV = error_std / (error_mean + EPS)

    ##------------------------------------------------------------------------.
    # - Magnitude metrics
    BIAS = error_mean
    MAE = error_abs.mean(axis=1)
    MSE = error_squared.mean(axis=1)
    RMSE = np.sqrt(MSE)

    percBIAS = error_perc.mean(axis=1) * 100
    percMAE = np.abs(error_perc).mean(axis=1) * 100

    relBIAS = BIAS / (obs_mean + EPS)
    relMAE = MAE / (obs_mean + EPS)
    relMSE = MSE / (obs_mean + EPS)
    relRMSE = RMSE / (obs_mean + EPS)

    ##------------------------------------------------------------------------.
    # - Average metrics
    rMean = pred_mean / (obs_mean + EPS)
    diffMean = pred_mean - obs_mean

    ##------------------------------------------------------------------------.
    # - Variability metrics
    rSD = pred_std / (obs_std + EPS)
    diffSD = pred_std - obs_std
    rCoV = pred_CoV / obs_CoV
    diffCoV = pred_CoV - obs_CoV

    # - Correlation metrics
    pearson_R = __pearson_corr_coeff(x=pred,
                                     y=obs,
                                     mean_x=pred_mean,
                                     mean_y=obs_mean,
                                     std_x=pred_std,
                                     std_y=obs_std)
    pearson_R2 = pearson_R**2
    spearman_R = _spearman_corr_coeff(x=pred,
                                      y=obs)
    spearman_R2 = spearman_R**2

    # pearson_R_pvalue = scipy.stats.pearsonr(pred, obs)
    # spearman_R_pvalue = scipy.stats.spearmanr(pred, obs)

    ##------------------------------------------------------------------------.
    # - Overall skill metrics
    obs_diff_from_ltm_mean = np.expand_dims(obs_mean, axis=1) - obs
    factor = (obs_diff_from_ltm_mean ** 2).sum(axis=1)
    sum_error_squared = error_squared.sum(axis=1)
    NSE = 1 - (sum_error_squared / (factor + EPS))
    KGE = 1 - (np.sqrt((pearson_R - 1) ** 2 + (rSD - 1) ** 2 + (rMean - 1) ** 2))

    ##------------------------------------------------------------------------.
    skills = np.stack(
        [
            pred_CoV,
            obs_CoV,
            error_CoV,

            # Magnitude
            BIAS,
            MAE,
            MSE,
            RMSE,
            percBIAS,
            percMAE,
            relBIAS,
            relMAE,
            relMSE,
            relRMSE,

            # Average
            rMean,
            diffMean,

            # Variability
            rSD,
            diffSD,
            rCoV,
            diffCoV,

            # Correlation
            pearson_R,
            pearson_R2,
            spearman_R,
            spearman_R2,
            # pearson_R_pvalue,
            # spearman_R_pvalue,

            # Overall skill
            NSE,
            KGE,
        ], axis = -1
    )
    return skills


def get_metrics_info():
    """Get metrics information."""
    func = _get_metrics
    skill_names = [
        "pred_CoV",
        "obs_CoV",
        "error_CoV",
        # Magnitude
        "BIAS",
        "MAE",
        "MSE",
        "RMSE",
        "percBIAS",
        "percMAE",
        "relBIAS",
        "relMAE",
        "relMSE",
        "relRMSE",
        # Average
        "rMean",
        "diffMean",
        # Variability
        "rSD",
        "diffSD",
        "rCoV",
        "diffCoV",
        # Correlation
        "pearson_R",
        "pearson_R2",
        "spearman_R",
        "spearman_R2",
        # "pearson_R_pvalue",
        # "spearman_R_pvalue",

        # Overall skill
        "NSE",
        "KGE",
    ]
    return func, skill_names


@print_elapsed_time(task="deterministic continuous")
def _xr_apply_routine(
    pred, obs, dims=("time"), **kwargs,
):
    """Routine to compute metrics in a vectorized way.

    The input xarray objects are first stacked to have 2D dimensions: (aux, sample).
    Then custom preprocessing is applied to deal with NaN, non-finite values,
    samples with equals values (i.e. 0) or samples outside specific value ranges.
    The resulting array is then passed to the function computing the metrics.

    Each chunk of the xr.DataArray is processed in parallel using dask, and the
    results recombined using xr.apply_ufunc.
    """
    # Broadcast obs to pred
    # - Creates a view, not a copy !
    obs_broadcasted = obs.broadcast_like(pred)
    # obs_broadcasted['var0'].data.flags # view (Both contiguous are False)
    # obs_broadcasted['var0'].data.base  # view (Return array and not None)

    # Stack pred and obs to have 2D dimensions (aux, sample)
    stacking_dict = get_stacking_dict(pred, aggregating_dim=dims)
    stacked_pred = pred.stack(stacking_dict)
    stacked_obs = obs_broadcasted.stack(stacking_dict)

    # Retrieve function and skill names
    func, skill_names = get_metrics_info()

    # Define gufunc kwargs
    input_core_dims = [["sample"], ["sample"]]
    dask_gufunc_kwargs = {
        "output_sizes":
            {
                "skill": len(skill_names),
             }
    }

    # Apply ufunc
    ds_skill = xr.apply_ufunc(
        func,
        stacked_pred,
        stacked_obs,
        kwargs=kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=[["skill"]],
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        output_dtypes=["float64"],
    )

    # Compute the skills
    with ProgressBar():
        ds_skill = ds_skill.compute()

    # Add skill coordinates
    ds_skill = ds_skill.assign_coords({"skill": skill_names})

    # Unstack
    ds_skill = ds_skill.unstack("aux")

    # Return the skill Dataset
    return ds_skill




