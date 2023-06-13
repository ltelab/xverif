#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:48:21 2023.

@author: ghiggi
"""
import xarray as xr

from xverif.deterministic.categorical_metrics import _deterministic_categorical_metrics
from xverif.deterministic.continuous_metrics import _deterministic_continuous_metrics
from xverif.deterministic.spatial_metrics import _spatial_metrics_xarray


##----------------------------------------------------------------------------.
def xr_common_vars(x, y):
    """Retrieve common variables between two xr.Dataset."""
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

def ensure_dataset_same_variables(pred, obs):
    """Return xr.Datasets with common variables."""
    common_vars = xr_common_vars(pred, obs)
    if common_vars is None:
        raise ValueError("No common variables between obs and pred xr.Dataset.")
    pred = pred[common_vars]
    obs = obs[common_vars]
    return pred, obs


def check_forecast_type(forecast_type):
    """Check forecast_type validity."""
    if not isinstance(forecast_type, str):
        raise TypeError(
            "'forecast_type' must be a string specifying the forecast type."
        )
    if forecast_type not in ["continuous", "categorical", "spatial"]:
     raise ValueError(
         "'forecast_type' must be either 'continuous', 'categorical' or 'spatial'."
     )
     return forecast_type


def align_datasets(pred, obs):
    """Align xarray Dataset."""
    # Align dataset dimensions
    pred, obs = xr.align(pred, obs, join="inner")

    # Align dataset variables
    # TODO: check that both are xr.Dataset !
    pred, obs  = ensure_dataset_same_variables(pred, obs)
    return pred, obs



def deterministic(
    pred,
    obs,
    forecast_type="continuous",
    aggregating_dim=None,
    skip_na=True,
    # TODO ADD
    skip_infs=True,
    skip_zeros=True,
    win_size=5,
    thr=0.000001,
):
    """Compute deterministic skill metrics."""
    # ------------------------------------------------------------------------.
    # Check input arguments
    # TODO
    forecast_type = check_forecast_type(forecast_type)

    # ------------------------------------------------------------------------.
    # Align datasets
    pred, obs = align_datasets(pred, obs)

    # ------------------------------------------------------------------------.
    # Run deterministic verification
    if forecast_type == "continuous":
        ds_skill = _deterministic_continuous_metrics(
            pred=pred,
            obs=obs,
            dim=aggregating_dim,
            skip_na=skip_na,
            thr=thr
        )
    elif forecast_type == "categorical":
        ds_skill = _deterministic_categorical_metrics(
            pred=pred,
            obs=obs,
            dim=aggregating_dim,
            skip_na=skip_na,
            skip_infs=skip_infs,
            skip_zeros=skip_zeros,
            thr=thr,
        )
    else:
        ds_skill = _spatial_metrics_xarray(
            pred=pred,
            obs=obs,
            dim=aggregating_dim,
            thr=thr,
            win_size=win_size
        )
    return ds_skill


# -----------------------------------------------------------------------------.
