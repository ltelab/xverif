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
from xverif.utils.warnings import suppress_warnings


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

def _mIOA(error, pred, obs_anomaly, obs_Mean, j=1):
    """Return the modified Index of Agreement. 
    
    See Wilmott et al., 1981.
    """
    # TOREAD
    # https://search.r-project.org/CRAN/refmans/hydroGOF/html/md.html
    # https://www.nature.com/articles/srep19401 --> IoA_Gallo
    pred_anomaly = pred - np.expand_dims(obs_Mean, axis=1)  
    denominator = np.nansum(((np.abs(obs_anomaly) + np.abs(pred_anomaly)) ** j), axis=1)
    mIoA = 1 - np.nansum(error ** j, axis=1) / (denominator + EPS)
    return mIoA 


def get_available_metrics():
    """Return available metrics."""
    arr = np.zeros((1,3))
    dict_metrics = _get_metrics(arr, arr)
    return list(dict_metrics)


def check_metrics(metrics): 
    """Checks metrics validity. 
    
    If None, returns all available metrics.
    """
    available_metrics = get_available_metrics()
    if isinstance(metrics, str): 
        metrics = [metrics]
    if isinstance(metrics, type(None)):
        metrics = available_metrics
    if isinstance(metrics, (tuple, list)):
        metrics = np.array(metrics, dtype="str") 
        
    idx_invalid = np.where(np.isin(metrics, available_metrics, invert=True))[0]
    if len(idx_invalid) > 0:
        invalid_metrics = metrics[idx_invalid]
        raise ValueError(f"These metrics are invalid: {invalid_metrics}")
    return metrics 

              
@suppress_warnings  
def _get_metrics(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Deterministic metrics for continuous predictions forecasts.
    
    Robust metrics are computed using the median and mean absolute deviation instead 
    of the mean and the standard deviation. This metrics are prefixed with 'rob_'

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
    skills : dict
        A dictionary of format <metric>: <value>.
    """
    # Number of valid samples 
    N = np.logical_and(np.isfinite(pred), np.isfinite(obs)).sum(axis=1)

    ##------------------------------------------------------------------------.
    # - Error
    error = pred - obs
    error_squared = error**2
    error_abs = np.abs(error)
    error_perc = error / (obs + EPS)
    error_abs_perc = error_abs / np.maximum(np.abs(obs), EPS)
    
    ##------------------------------------------------------------------------.
    # Sum of errors
    SE = np.nansum(error, axis=1)
    SAE = np.nansum(error_abs, axis=1)
    SSE = np.nansum(error_squared, axis=1)
    
    ##------------------------------------------------------------------------.
    # - Mean
    pred_Mean = np.nanmean(pred, axis=1)
    obs_Mean =  np.nanmean(obs, axis=1)
    error_Mean = np.nanmean(error, axis=1)
    
    obs_anomaly = obs - np.expand_dims(obs_Mean, axis=1)
    
    ##------------------------------------------------------------------------.
    # - Standard deviation
    pred_SD = np.nanstd(pred, axis=1)
    obs_SD = np.nanstd(obs, axis=1)
    error_SD = np.nanstd(error, axis=1)
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
    MAE = np.nanmean(error_abs, axis=1)
    MSE = np.nanmean(error_squared, axis=1)
    RMSE = np.sqrt(MSE)
    
    MaxAbsError = np.nanmax(error_abs, axis=1)

    percBIAS = np.nanmean(error_perc, axis=1) * 100
    percMAE = np.nanmean(np.abs(error_perc), axis=1) * 100 # MAPE
    MAPE = np.nanmean(error_abs_perc, axis=1) * 100

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
    
    # Explained Variance Score 
    # - Variance accounted for (VAF)
    EVS = 1 - error_SD**2 / obs_SD**2 
    
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
    denominator = np.nansum(obs_anomaly ** 2, axis=1)
    NSE = 1 - (SSE / (denominator + EPS))

    # - Klinga Gupta Efficiency Score
    KGE = 1 - (np.sqrt((pearson_R - 1) ** 2 + (rSD - 1) ** 2 + (rMean - 1) ** 2))
    
    # - Modified Index of Agreement (Wilmott et al., 1981)
    mIoA1 = _mIOA(error, pred, obs_anomaly, obs_Mean, j=1)
    mIoA2 = _mIOA(error, pred, obs_anomaly, obs_Mean, j=2)
    mIoA3 = _mIOA(error, pred, obs_anomaly, obs_Mean, j=3)
 
    ##------------------------------------------------------------------------.
    # - Median
    pred_Median = np.nanmedian(pred, axis=1)
    obs_Median = np.nanmedian(obs, axis=1)
    error_Median = np.nanmedian(error, axis=1)
       
    ##------------------------------------------------------------------------.
    # - Mean Absolute Deviation
    pred_MAD = np.nanmedian(np.abs(pred - np.expand_dims(pred_Median, axis=1)), axis=1)
    obs_MAD = np.nanmedian(np.abs(obs - np.expand_dims(obs_Median, axis=1)), axis=1)
    error_MAD = np.nanmedian(np.abs(error - np.expand_dims(error_Median, axis=1)), axis=1)
      
    ##------------------------------------------------------------------------.
    # - Robust Coefficient of variability
    rob_pred_CoV = pred_MAD / (pred_Median + EPS)
    rob_obs_CoV = obs_MAD / (obs_Median + EPS)
    rob_error_CoV = error_MAD / (error_Median + EPS)
      
    ##------------------------------------------------------------------------.
    # - Robust Magnitude metrics
    rob_BIAS = error_Median
    rob_MAE = np.nanmedian(error_abs, axis=1)
    rob_MSE = np.nanmedian(error_squared, axis=1)
    rob_RMSE = np.sqrt(rob_MSE)
      
    rob_percBIAS = np.nanmedian(error_perc, axis=1) * 100
    rob_percMAE = np.nanmedian(np.abs(error_perc), axis=1) * 100 
    
    rob_relBIAS = rob_BIAS / (obs_Median + EPS)
    rob_relMAE = rob_MAE / (obs_Median + EPS)
    rob_relMSE = rob_MSE / (obs_Median + EPS)
    rob_relRMSE = rob_RMSE / (obs_Median + EPS)
      
    ##------------------------------------------------------------------------.
    # - Average metrics
    rMedian = pred_Median / (obs_Median + EPS)
    diffMedian = pred_Median - obs_Median
      
    ##------------------------------------------------------------------------.
    # - Variability metrics
    rMAD = pred_MAD / (obs_MAD + EPS)
    diffMAD = pred_MAD - obs_MAD
    rob_rCoV = rob_pred_CoV / rob_obs_CoV
    rob_diffCoV = rob_pred_CoV - rob_obs_CoV
     

    ##------------------------------------------------------------------------.
    dictionary = {
        "N": N, 
        # Sum of errors
        "SE": SE,
        "SAE": SAE,
        "SSE": SSE,
        
        # Mean  
        "obs_Mean": obs_Mean,
        "pred_Mean": pred_Mean,
        "rMean": rMean,
        "diffMean": diffMean,
        
        # Median  
        "obs_Median": obs_Median,
        "pred_Median": pred_Median,
        "rMedian": rMedian,
        "diffMedian": diffMedian,
        
        # Variability 
        "obs_SD": obs_SD,
        "pred_SD": pred_SD,
        "error_SD": error_SD,
        "diffSD": diffSD,
        "rSD": rSD,
         
        "obs_CoV": obs_CoV,
        "pred_CoV": pred_CoV,
        "error_CoV": error_CoV,
        "rCoV": rCoV,
        "diffCoV": diffCoV,
        "EVS": EVS,
        
        "obs_MAD": obs_MAD,
        "pred_MAD": pred_MAD,
        "error_MAD": error_MAD,
        "diffMAD": diffMAD,
        "rMAD": rMAD,
        
        "rob_obs_CoV": rob_obs_CoV,
        "rob_pred_CoV": rob_pred_CoV,
        "rob_error_CoV": rob_error_CoV,
        "rob_diffCoV": rob_diffCoV,
        "rob_rCoV": rob_rCoV,
        
        # BIAS 
        "BIAS": BIAS,
        "percBIAS": percBIAS,
        "relBIAS": relBIAS,
        
        "rob_BIAS": rob_BIAS,
        "rob_percBIAS": rob_percBIAS,
        "rob_relBIAS": rob_relBIAS,
            
        # Magnitude error 
        "MaxAbsError": MaxAbsError,
        
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        
        "rob_MAE": rob_MAE,
        "rob_MSE": rob_MSE,
        "rob_RMSE": rob_RMSE,
        
        "percMAE": percMAE,
        "MAPE": MAPE,
        
        "rob_percMAE": rob_percMAE,
          
        "relMAE": relMAE,
        "relMSE": relMSE,
        "relRMSE": relRMSE,
        
        "rob_relMAE": rob_relMAE,
        "rob_relMSE": rob_relMSE,
        "rob_relRMSE": rob_relRMSE,
        
        # Correlations
        "pearson_R": pearson_R,
        "pearson_R2": pearson_R2,
        "spearman_R": spearman_R,
        "spearman_R2": spearman_R2,
        
        # Overall 
        "NSE": NSE,
        "KGE": KGE,
        "mIoA1": mIoA1,
        "mIoA2": mIoA2,
        "mIoA3": mIoA3,
    }
    
    return dictionary


def get_metrics(pred: np.ndarray, obs: np.ndarray, metrics: np.ndarray, 
                **kwargs) -> np.ndarray:
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
    metrics : np.ndarray
        List of metrics to compute.

    Returns
    -------
    skills : np.ndarray
        A 2D array of shape (aux, n_skills).
    """
    dict_metrics = _get_metrics(pred=pred, obs=obs)
    list_skills = [dict_metrics[metric] for metric in metrics]
    skills = np.stack(list_skills, axis = -1)
    return skills 


@print_elapsed_time(task="deterministic continuous")
def _xr_apply_routine(
    pred, obs, dims=("time"), metrics=None, compute=True, **kwargs,
):
    """Routine to compute metrics in a vectorized way.

    The input xarray objects are first stacked to have 2D dimensions: (aux, sample).
    Then custom preprocessing is applied to deal with NaN, non-finite values,
    samples with equals values (i.e. 0) or samples outside specific value ranges.
    The resulting array is then passed to the function computing the metrics.

    Each chunk of the xr.DataArray is processed in parallel using dask, and the
    results recombined using xr.apply_ufunc.
    """
    metrics = check_metrics(metrics)
    
    # Broadcast obs to pred
    # - Creates a view, not a copy !
    obs = obs.broadcast_like(pred)    
    # obs_broadcasted['var0'].data.flags # view (Both contiguous are False)
    # obs_broadcasted['var0'].data.base  # view (Return array and not None)
    
    # Ensure obs same chunks of pred on auxiliary dims 
    # TODO: now makes same as pred --> TO IMPROVE 
    obs = obs.chunk(pred.chunks) 

    # Stack pred and obs to have 2D dimensions (aux, sample)
    # - This operation doubles the memory
    stacking_dict = get_stacking_dict(pred, aggregating_dim=dims)
    pred = pred.stack(stacking_dict)
    obs = obs.stack(stacking_dict)
    
    # Preprocess 
    # - Based on kwargs
    
    
    
    
    # Define gufunc kwargs
    input_core_dims = [["sample"], ["sample"]]
    dask_gufunc_kwargs = {
        "output_sizes":
            {
                "skill": len(metrics),
             }
    }

    # Apply ufunc
    ds_skill = xr.apply_ufunc(
        get_metrics,
        pred,
        obs,
        kwargs={"metrics": metrics},
        input_core_dims=input_core_dims,
        output_core_dims=[["skill"]],
        vectorize=False,
        dask="allowed",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        output_dtypes=["float64"],
    )

    # Compute the skills
    if compute:
        with ProgressBar():
            ds_skill = ds_skill.compute()

    # Add skill coordinates
    ds_skill = ds_skill.assign_coords({"skill": metrics})

    # Unstack
    ds_skill = ds_skill.unstack("aux")

    # Return the skill Dataset
    return ds_skill




