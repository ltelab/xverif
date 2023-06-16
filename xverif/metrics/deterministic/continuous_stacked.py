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


def _get_metrics(pred: np.ndarray, obs: np.ndarray, **kwargs) -> np.ndarray:
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
    pred_Mean = pred.nanmean(axis=1)
    obs_Mean = obs.nanmean(axis=1)
    error_Mean = error.nanmean(axis=1)
    ##------------------------------------------------------------------------.
    # - Standard deviation
    pred_SD = pred.nanstd(axis=1)
    obs_SD = obs.nanstd(axis=1)
    error_SD = error.nanstd(axis=1)
    ##------------------------------------------------------------------------.
    # Spread
    # - Scatter
    # --> Half the distance between the 16% and 84% percentiles with error dB(pred/obs)

    # TODO: check
    # error_db = 10 * np.log10(pred / (obs + EPS))
    # q16, q84 = np.nanquantiles(error_db, q=[0.16, 0.25, 0.75, 0.84], axis=1)
    # SCATTER = (q84 - q16) / 2.0

    # # - IQR
    # q25, q75 = np.nanquantiles(error, q=[0.25, 0.75], axis=1)
    # IQR = q75 - q25

    ##------------------------------------------------------------------------.
    # - Coefficient of variability
    pred_CoV = pred_SD / (pred_Mean + EPS)
    obs_CoV = obs_SD / (obs_Mean + EPS)
    error_CoV = error_SD / (error_Mean + EPS)

    ##------------------------------------------------------------------------.
    # - Magnitude metrics
    BIAS = error_Mean
    MAE = error_abs.nanmean(axis=1)
    MSE = error_squared.nanmean(axis=1)
    RMSE = np.sqrt(MSE)

    percBIAS = error_perc.nanmean(axis=1) * 100
    percMAE = np.abs(error_perc).nanmean(axis=1) * 100

    relBIAS = BIAS / (obs_Mean + EPS)
    relMAE = MAE / (obs_Mean + EPS)
    relMSE = MSE / (obs_Mean + EPS)
    relRMSE = RMSE / (obs_Mean + EPS)

    ##------------------------------------------------------------------------.
    # - Average metrics
    rMean = pred_Mean / (obs_Mean + EPS)
    diffMean = pred_Mean - obs_Mean

    ##------------------------------------------------------------------------.
    # - Variability metrics
    rSD = pred_SD / (obs_SD + EPS)
    diffSD = pred_SD - obs_SD
    rCoV = pred_CoV / obs_CoV
    diffCoV = pred_CoV - obs_CoV

    # - Correlation metrics
    pearson_R = __pearson_corr_coeff(x=pred,
                                     y=obs,
                                     mean_x=pred_Mean,
                                     mean_y=obs_Mean,
                                     std_x=pred_SD,
                                     std_y=obs_SD)
    pearson_R2 = pearson_R**2
    spearman_R = _spearman_corr_coeff(x=pred,
                                      y=obs)
    spearman_R2 = spearman_R**2

    # pearson_R_pvalue = scipy.stats.pearsonr(pred, obs)
    # spearman_R_pvalue = scipy.stats.spearmanr(pred, obs)

    ##------------------------------------------------------------------------.
    # - Overall skill metrics
    # - Nash Sutcliffe efficiency / Reduction of variance
    # - --> 1 - MSE/Var(obs)
    NSE = 1 - (error_squared.nansum(axis=1) / (((obs - np.expand_dims(obs_Mean, axis=1)) ** 2).nansum(axis=1) + EPS))

    # - Klinga Gupta Efficiency Score
    KGE = 1 - (np.sqrt((pearson_R - 1) ** 2 + (rSD - 1) ** 2 + (rMean - 1) ** 2))

    # - Modified Index of Agreement (Wilmott et al., 1981)
    # https://search.r-project.org/CRAN/refmans/hydroGOF/html/md.html
    # https://www.nature.com/articles/srep19401 --> IoA_Gallo
    # --> TODO: j1, j2, j3
    # j = 1
    # denominator = ((np.abs((np.expand_dims(obs_Mean, axis=1) - obs)) +
    #                 np.abs((np.expand_dims(obs_Mean, axis=1) - pred))) ** j).nansum(axis=1)
    # mIoA = 1 - (error ** j).nansum(axis=1) / (denominator + EPS)

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

            # IoA,
        ], axis = -1
    )
    return skills


def get_metrics_info():
    """Get metrics information."""
    func = _get_metrics
    skill_names = [
        "obs_Mean",
        "pred_Mean",
        "obs_SD",
        "pred_SD",
        "error_SD",
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
    obs = obs.broadcast_like(pred)
    # obs_broadcasted['var0'].data.flags # view (Both contiguous are False)
    # obs_broadcasted['var0'].data.base  # view (Return array and not None)

    # Stack pred and obs to have 2D dimensions (aux, sample)
    # - This operation doubles the memory
    stacking_dict = get_stacking_dict(pred, aggregating_dim=dims)
    pred = pred.stack(stacking_dict)
    obs = obs.stack(stacking_dict)

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
        pred,
        obs,
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




