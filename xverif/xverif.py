#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:51:43 2021

@author: ghiggi
"""
import os
from sklearn.metrics import accuracy_score
import xarray as xr 
import numpy as np 
import pandas as pd
import time 
import scipy.stats
import time
from dask.diagnostics import ProgressBar
from xskillscore import crps_ensemble
from skimage.metrics import structural_similarity as ssim
from pysteps.verification.spatialscores import fss

# https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r_eff_p_value.html
##----------------------------------------------------------------------------.
## Weighting for equiangular 
# weights_lat = np.cos(np.deg2rad(lat))
# weights_lat /= weights_lat.sum()  
# error * weights_lat 

##----------------------------------------------------------------------------.
def xr_common_vars(x,y):
    """ Retrieve common variables between two xr.Dataset."""  
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting xr.Dataset.")
    if not isinstance(y, xr.Dataset):
        raise TypeError("Expecting xr.Dataset.")
    # Retrieve common vars
    x_vars = list(x.data_vars.keys())
    y_vars = list(y.data_vars.keys())
    common_vars = list(set(x_vars).intersection(set(y_vars)))
    if len(common_vars) == 0: 
        return None 
    else: 
        return common_vars

##----------------------------------------------------------------------------.
def _match_nans(a, b, weights=None):
    """
    Considers missing values pairwise. 
    If a value is missing in a, the corresponding value in b is turned to nan, and
    vice versa.
    
    Code source: https://github.com/xarray-contrib/xskillscore
     
    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with nans placed at pairwise locations.
    """
    if np.isnan(a).any() or np.isnan(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isnan(a), np.isnan(b))
        a[idx], b[idx] = np.nan, np.nan
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights[idx] = np.nan
    return a, b, weights

def _drop_nans(a, b, weights=None):
    """
    Considers missing values pairwise. 
    If a value is missing in a or b, the corresponding indices are dropped.
         
    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with nans placed at pairwise locations.
    """
    if np.isnan(a).any() or np.isnan(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isnan(a), np.isnan(b))
        a = np.delete(a, idx)
        b = np.delete(b, idx)
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights = np.delete(weights, idx)
    return a, b, weights


def _drop_infs(a, b, weights=None):
    """
    Considers infinite values pairwise. 
    If a value is infinite in a or b, the corresponding indices are dropped.
         
    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with infs placed at pairwise locations.
    """
    if np.isinf(a).any() or np.isinf(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isinf(a), np.isinf(b))
        a = np.delete(a, idx)
        b = np.delete(b, idx)
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights = np.delete(weights, idx)
    return a, b, weights
 

def _drop_pairwise_elements(a, b, weights=None, element=0):
    """
    Considers values pairwise. 
    If the element is the same in a and b, the corresponding indices are dropped.
         
    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with elements placed at pairwise locations.
    """
    idx = np.logical_and(a == element, b == element)
    if idx.any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isinf(a), np.isinf(b))
        a = np.delete(a, idx)
        b = np.delete(b, idx)
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights = np.delete(weights, idx)
    return a, b, weights

##----------------------------------------------------------------------------.
# Covariance/Correlation functions for xarray 
def _inner(x, y):
    result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
    return result[..., 0, 0]

def _xr_inner_product(x, y, dim, dask="parallelized"):
    if dim is not None:
        if isinstance(dim, str):
            dim = [dim]
        if isinstance(dim, tuple):
            dim = list()
        if len(dim) == 2: 
            # TODO reshape data to aggregate_dims x 'time'   
            raise NotImplementedError
        input_core_dims = [dim, dim] # [[x_dim, y_dim]
    else: 
        raise ValueError("Requiring a dimension...")
    return xr.apply_ufunc(_inner, x, y, 
                          input_core_dims=input_core_dims,
                          dask="parallelized",
                          output_dtypes=[float])

def _xr_covariance(x, y, aggregating_dims=None, dask="parallelized"):
    x_mean = x.mean(aggregating_dims)
    y_mean = y.mean(aggregating_dims)
    N = x.count(aggregating_dims)
    return _xr_inner_product(x - x_mean, y - y_mean, dim=aggregating_dims, dask=dask) / N
    
def _xr_pearson_correlation(x, y, aggregating_dims=None, thr=0.0000001, dask="parallelized"):
    x_std = x.std(aggregating_dims) + thr
    y_std = y.std(aggregating_dims) + thr
    return _xr_covariance(x, y, aggregating_dims=aggregating_dims, dask=dask)/(x_std*y_std)

# import bottleneck
# def _xr_rank(x, dim, dask="parallelized"): 
#     return xr.apply_ufunc(bottleneck.rankdata, x,
#                           input_core_dims=[[dim]],  
#                           dask="parallelized")

# def _xr_spearman_correlation(x, y, aggregating_dims=None, thr=0.0000001):
#     x_rank= x.rank(dim=aggregating_dims) 
#     y_rank = y.rank(dim=aggregating_dims)
#     return _xr_pearson_correlation(x_rank,y_rank, aggregating_dims=aggregating_dims, thr=thr)
##----------------------------------------------------------------------------.
def _det_cont_metrics(pred, obs, thr=0.000001, skip_na=True, skip_infs=True, skip_zeros=True):
    """Deterministic metrics for continuous predictions forecasts.

    This function expects pred and obs to be 1D vector of same size
    """   
    # TODO robust with median and IQR / MAD 
    ##------------------------------------------------------------------------.
    # Preprocess data (remove NaN if asked)
    if skip_na: 
        pred, obs, _ = _drop_nans(pred, obs)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 2:
            return np.ones(27)*np.nan

    # Preprocess data (remove NaN if asked)
    if skip_infs: 
        pred, obs, _ = _drop_infs(pred, obs)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 2:
            return np.ones(27)*np.nan
    
    if skip_zeros:
        pred, obs, _ = _drop_pairwise_elements(pred, obs, element=0)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 1:
            return np.ones(27)*np.nan
    ##------------------------------------------------------------------------.
    # - Error 
    error = pred - obs
    error_squared = error**2
    error_perc = error/(obs + thr)
    ##------------------------------------------------------------------------.
    # - Mean 
    pred_mean = pred.mean() 
    obs_mean = obs.mean() 
    error_mean = error.mean() 
    ##------------------------------------------------------------------------. 
    # - Standard deviation
    pred_std = pred.std() 
    obs_std = obs.std() 
    error_std = error.std() 
    ##------------------------------------------------------------------------.
    # - Coefficient of variability
    pred_CoV = pred_std / (pred_mean + thr) 
    obs_CoV = obs_std / (obs_mean + thr)
    error_CoV = error_std / (error_mean + thr)
    ##------------------------------------------------------------------------.
    # - Magnitude metrics
    BIAS = error_mean
    MAE = np.abs(error).mean() 
    MSE = error_squared.mean() 
    RMSE = np.sqrt(MSE)
    
    percBIAS = error_perc.mean()*100
    percMAE = np.abs(error_perc).mean()*100  
    
    relBIAS = BIAS / (obs_mean + thr)
    relMAE = MAE / (obs_mean + thr)
    relMSE = MSE / (obs_mean + thr)
    relRMSE = RMSE / (obs_mean + thr)
    ##------------------------------------------------------------------------.
    # - Average metrics 
    rMean = pred_mean / (obs_mean + thr)
    diffMean = pred_mean - obs_mean
    ##------------------------------------------------------------------------.
    # - Variability metrics 
    rSD = pred_std / (obs_std + thr)
    diffSD = pred_std - obs_std  
    rCoV = pred_CoV / obs_CoV
    diffCoV = pred_CoV - obs_CoV
    # - Correlation metrics 
    pearson_R, pearson_R_pvalue = scipy.stats.pearsonr(pred, obs)                                  
    pearson_R2 = pearson_R**2
    
    spearman_R, spearman_R_pvalue = scipy.stats.spearmanr(pred, obs)
    spearman_R2 = spearman_R**2
    ##------------------------------------------------------------------------.
    # - Overall skill metrics 
    LTM_forecast_error = ((obs_mean - obs)**2).sum() # Long-term mean as prediction
    NSE = 1 - ( error_squared.sum()/ (LTM_forecast_error + thr) )
    KGE = 1 - ( np.sqrt((pearson_R - 1)**2 + (rSD - 1)**2 + (rMean - 1)**2) )
    ##------------------------------------------------------------------------.
    skills = np.array([pred_CoV,
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
                       pearson_R_pvalue,
                       pearson_R2,
                       spearman_R,
                       spearman_R_pvalue,
                       spearman_R2,
                       # Overall skill 
                       NSE,
                       KGE])
    return skills 
##----------------------------------------------------------------------------.
def _deterministic_continuous_metrics(pred, obs, 
                                      dim = "time", 
                                      skip_na = True,
                                      skip_infs=True, 
                                      skip_zeros=True,
                                      thr=0.000001):                 
    ds_skill = xr.apply_ufunc(_det_cont_metrics,
                              pred,
                              obs,
                              kwargs = {'thr': thr, 'skip_na': skip_na,
                                        "skip_infs": skip_infs, 
                                        "skip_zeros": skip_zeros}, 
                              input_core_dims=[[dim], [dim]],  
                              output_core_dims=[["skill"]],  # returned data has one dimension
                              vectorize=True,
                              dask="parallelized",
                              dask_gufunc_kwargs = {'output_sizes': {'skill': 27,}},                         
                              output_dtypes=['float64'])  # dtype  
    # Compute the skills
    with ProgressBar():                        
        ds_skill = ds_skill.compute()
    # Add skill coordinates
    skill_str = ["pred_CoV", "obs_CoV", "error_CoV",
            # Magnitude
            "BIAS", "MAE", "MSE", "RMSE",
            "percBIAS", "percMAE",
            "relBIAS", "relMAE", "relMSE", "relRMSE",
            # Average
            "rMean", "diffMean",
            # Variability
            "rSD", "diffSD", "rCoV", "diffCoV",
            # Correlation
            "pearson_R", "pearson_R_pvalue", "pearson_R2",
            "spearman_R", "spearman_R_pvalue", "spearman_R2",
            # Overall skill 
            "NSE", "KGE"]         
    ds_skill = ds_skill.assign_coords({"skill": skill_str})    
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill


#-----------------------------------------------------------------------------.
# #########################################
#### Deterministic categorical metrics ####
# #########################################

def _det_cat_metrics(pred, obs, thr=0.000001, skip_na=True, skip_infs=True, skip_zeros=True):
    """Deterministic metrics for continuous predictions forecasts.

    This function expects pred and obs to be 1D vector of same size
    """   
    # TODO robust with median and IQR / MAD 
    ##------------------------------------------------------------------------.
    # Preprocess data (remove NaN if asked)
    if skip_na: 
        pred, obs, _ = _drop_nans(pred, obs)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 2:
            return np.ones(12)*np.nan

    # Preprocess data (remove NaN if asked)
    if skip_infs: 
        pred, obs, _ = _drop_infs(pred, obs)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 2:
            return np.ones(12)*np.nan
    
    if skip_zeros:
        pred, obs, _ = _drop_pairwise_elements(pred, obs, element=0)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 1:
            return np.ones(12)*np.nan
    
    # apply threshold
    predb = pred > thr
    obsb = obs > thr

    # calculate hits, misses, false positives, correct rejects
    H = np.nansum(np.logical_and(predb == 1, obsb == 1), dtype="float64")
    F = np.nansum(np.logical_and(predb == 1, obsb == 0), dtype="float64")
    M = np.nansum(np.logical_and(predb == 0, obsb == 1), dtype="float64")
    R = np.nansum(np.logical_and(predb == 0, obsb == 0), dtype="float64")

    # Probability of detection
    POD = H / (H + M)
    # False alarm ratio
    FAR = F / (H + F)
    # False alaram rate (prob of false detection)
    FA = F / (F + R)
    s = (H + M) / (H + M + F + R)

    # Accuracy (fraction correct)
    ACC = (H + R) / (H + M + F + R)
    # Critical success index
    CSI = H / (H + M + F)
    # Frequency bias
    FB = (H + F) / (H + M)

    # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
    HSS = 2 * (H * R - F * M) / ((H + M) * (M + R) + (H + F) * (F + R))
    # Hanssen-Kuipers Discriminant
    HK = POD - FA

    # Gilbert Skill Score
    GSS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
    # Symmetric extremal dependence index
    SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA)) / (
        np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA)
    )
    # Matthews correlation coefficient
    MCC = (H * R - F * M) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))
    # F1 score
    F1 = 2 * H / (2 * H + F + M)
    
    skills = np.array([POD,
                       FAR,
                       FA,
                       ACC,
                       CSI,
                       FB,
                       HSS,
                       HK,
                       GSS,
                       SEDI,
                       MCC,
                       F1
                    ])
    
    return skills 
##----------------------------------------------------------------------------.
def _deterministic_categorical_metrics(pred, obs, 
                                       dim = "time", 
                                       skip_na = True,
                                       skip_infs=True, 
                                       skip_zeros=True,
                                       thr=0.000001):                 
    ds_skill = xr.apply_ufunc(_det_cat_metrics,
                              pred,
                              obs,
                              kwargs = {'thr': thr, 'skip_na': skip_na,
                                        "skip_infs": skip_infs, 
                                        "skip_zeros": skip_zeros}, 
                              input_core_dims=[[dim], [dim]],  
                              output_core_dims=[["skill"]],  # returned data has one dimension
                              vectorize=True,
                              dask="parallelized",
                              dask_gufunc_kwargs = {'output_sizes': {'skill': 12,}},                         
                              output_dtypes=['float64'])  # dtype  
    # Compute the skills
    with ProgressBar():                        
        ds_skill = ds_skill.compute()
    # Add skill coordinates
    skill_str = ["POD",
                 "FAR",
                 "FA",
                 "ACC",
                 "CSI",
                 "FB",
                 "HSS",
                 "HK",
                 "GSS",
                 "SEDI",
                 "MCC",
                 "F1"]         
    ds_skill = ds_skill.assign_coords({"skill": skill_str})    
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill


#-----------------------------------------------------------------------------.
# #############################
#### Probabilistic metrics ####
# #############################

def _prob_metrics(pred, obs, dims, crps_ref, dim_member="member"):
    ##------------------------------------------------------------------------.
    # - Percentage of observations within ensemble range (WER)
    min_pred = pred.min(axis=0)
    max_pred = pred.max(axis=0)
    
    within_range = ((obs >= min_pred) & (obs <= max_pred))
    within_range_mean = within_range.mean(dim=dims)
    
    ##------------------------------------------------------------------------.
    # - CRPS (Continous Ranked Probability Score)
    dims = list(pred.dims).copy()
    dims.remove(dim_member)
    dims.remove('lead_time')
    crps = crps_ensemble(obs, pred, dim=dims)

    ##------------------------------------------------------------------------.
    # - CRPSS (Continous Ranked Probability Skill Score)    
    crpss = 1 - (crps/crps_ref)

    skills = np.array([within_range_mean, crps, crpss])
    
    return skills

##----------------------------------------------------------------------------.
def _probabilistic_metrics(pred, obs, dims, crps_ref, dim_member="member"):                 
    ds_skill = xr.apply_ufunc(_prob_metrics,
                              pred,
                              obs,
                              kwargs = {'dim_member': dim_member, 'crps_ref': crps_ref,}, 
                              input_core_dims=[["member", "time"], ["time"]],  
                              output_core_dims=[["skill"]],  # returned data has one dimension
                              vectorize=True,
                              dask="parallelized",
                              dask_gufunc_kwargs = {'output_sizes': {'skill': 3,}},                         
                              output_dtypes=['float64'])  # dtype  
    # Compute the skills
    with ProgressBar():                        
        ds_skill = ds_skill.compute()
    # Add skill coordinates
    skill_str = ["WER", "CRPS", "CRPSS"]         
    ds_skill = ds_skill.assign_coords({"skill": skill_str})    
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill

#-----------------------------------------------------------------------------.
# #############################
####### Spatial metrics #######
# #############################

def _spatial_metrics(pred, obs, win_size=5, thr=0.1):
    pred = pred.copy()
    obs = obs.copy()

    pred[np.isnan(pred)] = 0
    obs[np.isnan(obs)] = 0
    # - Fraction Skill Score
    fraction_ss = fss(pred, obs, thr, win_size)
    
    # - Structural Similarity index
    mssim = ssim(pred, obs, win_size=win_size)

    skills = np.array([
        mssim,
        fraction_ss
    ])

    return skills

def _spatial_metrics_xarray(pred, obs, 
                            dim = "time", 
                            thr=0.000001,
                            win_size=5):
    input_core_dims = [[dim], [dim]] if type(dim) != list else [dim, dim]                 
    ds_skill = xr.apply_ufunc(_spatial_metrics,
                              pred,
                              obs,
                              kwargs = {"win_size": win_size, "thr": thr}, 
                              input_core_dims=input_core_dims,
                              output_core_dims=[["skill"]],  # returned data has one dimension
                              vectorize=True,
                              dask="parallelized",
                              dask_gufunc_kwargs = {'output_sizes': {'skill': 2,}},                         
                              output_dtypes=['float64'])  # dtype  
    # Compute the skills
    with ProgressBar():                        
        ds_skill = ds_skill.compute()
    # Add skill coordinates
    skill_str = ["SSIM", "FSS"]         
    ds_skill = ds_skill.assign_coords({"skill": skill_str})    
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill
#-----------------------------------------------------------------------------.
# #############################
#### Verification Wrappers ####
# #############################
def deterministic(pred, obs, 
                  forecast_type="continuous",
                  aggregating_dim=None,
                  skip_na=True,
                  skip_infs=True, 
                  skip_zeros=True,
                  thr=0.000001,
                  win_size=5):
    """Compute deterministic skill metrics."""
    # Check
    if not isinstance(forecast_type, str): 
        raise TypeError("'forecast_type' must be a string specifying the forecast type.")
    if forecast_type not in ["continuous", "categorical", "spatial"]:
        raise ValueError("'forecast_type' must be either 'continuous', 'categorical' or 'spatial'.") 
    #------------------------------------------------------------------------.
    # Align dataset (dimensions)
    pred, obs = xr.align(pred, obs, join="inner")
    # Align dataset (variables): 
    common_vars = xr_common_vars(pred, obs)
    if common_vars is None: 
        raise ValueError("No common variables between obs and pred xr.Dataset.")
    pred = pred[common_vars]
    obs = obs[common_vars]
    #------------------------------------------------------------------------.
    # Run deterministic verification
    if forecast_type == 'continuous':
        t_i = time.time()
        ds_skill = _deterministic_continuous_metrics(pred = pred,
                                                     obs = obs, 
                                                     dim = aggregating_dim,
                                                     skip_na = skip_na,
                                                     skip_infs = skip_infs, 
                                                     skip_zeros = skip_zeros,
                                                     thr = thr)
        print("- Elapsed time for forecast continuous deterministic verification: {:.2f} minutes.".format((time.time() - t_i)/60))
    elif forecast_type == "categorical": 
        t_i = time.time()
        ds_skill = _deterministic_categorical_metrics(pred = pred,
                                                      obs = obs, 
                                                      dim = aggregating_dim,
                                                      skip_na = skip_na,
                                                      skip_infs = skip_infs, 
                                                      skip_zeros = skip_zeros,
                                                      thr = thr)
        print("- Elapsed time for forecast categorical deterministic verification: {:.2f} minutes.".format((time.time() - t_i)/60))
    else:
        t_i = time.time()
        ds_skill = _spatial_metrics_xarray(pred = pred,
                                           obs = obs, 
                                           dim = aggregating_dim,
                                           thr = thr,
                                           win_size = win_size)
        print("- Elapsed time for forecast spatial deterministic verification: {:.2f} minutes.".format((time.time() - t_i)/60))
    
    return ds_skill
#-----------------------------------------------------------------------------.
# #############################
#### Spatial Summaries     ####
# #############################
def global_summary(ds, area_coords="area"):
    """Compute global statistics weighted by grid cell area."""
    # Check area_coords
    area_weights = ds[area_coords]/ds[area_coords].values.sum()
    aggregating_dims = list(area_weights.dims)
    ds_weighted = ds.weighted(area_weights)
    return ds_weighted.mean(aggregating_dims)

def latitudinal_summary(ds, lat_dim='lat', lon_dim='lon', lat_res=5):
    """Compute latitudinal (bin) statistics, averaging over longitude."""
    # Check lat_dim and lon_dim 
    # Check lat_res < 90 
    # TODO: lon between -180 and 180 , lat between -90 and 90 
    aggregating_dims = list(ds[lon_dim].dims)
    bins = np.arange(-90,90+lat_res, step=lat_res)
    labels = bins[:-1] + lat_res/2
    return ds.groupby_bins(lat_dim, bins, labels=labels).mean(aggregating_dims) 
    
def longitudinal_summary(ds, lat_dim='lat', lon_dim='lon', lon_res=5):
    """Compute longitudinal (bin) statistics, averaging over latitude."""
    # Check lat_dim and lon_dim 
    # Check lon_res < 180 
    # TODO: lon between -180 and 180 , lat between -90 and 90 
    aggregating_dims = list(ds[lon_dim].dims)
    bins = np.arange(-180,180+lon_res, step=lon_res)
    labels= bins[:-1] + lon_res/2
    return ds.groupby_bins(lon_dim, bins, labels=labels).mean(aggregating_dims) 

#-----------------------------------------------------------------------------.



