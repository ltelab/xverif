#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:48:21 2023.

@author: ghiggi
"""
from importlib import import_module

import numpy as np
import xarray as xr

VALID_FORECAST_TYPE = ["continuous", "categorical_binary", "categorical_multiclass"]
VALID_METRIC_TYPE = ["deterministic", "probabilistic", "spatial"]

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


def check_aggregating_dim(aggregating_dim):
    """Check aggregating_dims format."""
    if isinstance(aggregating_dim, str):
        aggregating_dim = [aggregating_dim]
    return aggregating_dim


def check_validity_aggregating_dim(aggregating_dim, xr_obj):
    """Check validity of aggregating dimensions."""
    dims = list(xr_obj.dims)
    aggregating_dim = np.array(aggregating_dim)
    unvalid_dims = aggregating_dim[np.isin(aggregating_dim, dims, invert=True)].tolist()
    if len(unvalid_dims) > 0:
        if len(unvalid_dims==1):
            raise ValueError(f"The aggregating dimension {unvalid_dims} is not an xarray dimension.")
        else:
             raise ValueError(f"The aggregating dimensions {unvalid_dims} are not xarray dimensions.")


def check_forecast_type(forecast_type):
    """Check forecast_type validity."""
    if not isinstance(forecast_type, str):
        raise TypeError("'forecast_type' must be a string.")
    if forecast_type not in VALID_FORECAST_TYPE:
        raise ValueError(f"Valid 'forecast_type' are {VALID_FORECAST_TYPE}")
    return forecast_type


def check_metric_type(metric_type):
    """Check metric_type validity."""
    if not isinstance(metric_type, str):
        raise TypeError("'metric_type' must be a string.")
    if metric_type not in VALID_METRIC_TYPE:
        raise ValueError(f"Valid 'metric_type' are {VALID_METRIC_TYPE}")
    return metric_type


def _get_xr_routine(metric_type, forecast_type):
    """Retrieve xarray routine to compute the metrics."""
    # Check inputs
    forecast_type = check_forecast_type(forecast_type)
    metric_type = check_metric_type(metric_type)
    # Define module path
    module_path = f"xverif.metrics.{metric_type}.{forecast_type}"
    # Import the module
    module = import_module(module_path)
    # Import the function
    function = module._xr_apply_routine
    return function


def ensure_common_xarray_format(pred, obs):
    """Ensure common format. Either xr.DataArray or xr.Dataset."""
    if not isinstance(pred, (xr.DataArray, xr.Dataset)):
        raise TypeError("'pred' is not an xarray object.")
    if not isinstance(obs, (xr.DataArray, xr.Dataset)):
        raise TypeError("'obs' is not an xarray object.")
    if isinstance(pred, xr.Dataset) and not isinstance(obs, xr.Dataset):
        raise TypeError("Both input datasets must be xr.Dataset or xr.DataArray.")
    if isinstance(pred, xr.DataArray) and not isinstance(obs, xr.DataArray):
        raise TypeError("Both input datasets must be xr.Dataset or xr.DataArray.")
    return pred, obs


def ensure_dataarray(xr_obj):
    """Convert xr.Dataset to DataArray.

    It expects that 'variable' is not a dimension of the xr.Dataset
    """
    if isinstance(xr_obj, xr.Dataset):
        if "variable" in list(xr_obj.dims):
            raise ValueError("Please do not provide a xr.Dataset with a dimension named 'variable'.")
        xr_obj = xr_obj.to_array(dim="variable")
    return xr_obj


def align_xarray_objects(pred, obs):
    """Align xarray objects.

    - Ensure coordinate alignment for matching dimensions.
    - Ensure common subset of Dataset variables.
    """
    # Align xarray dimensions
    pred, obs = xr.align(pred, obs, join="inner")

    # Align xarray dataset variables
    if isinstance(pred, xr.Dataset):
        pred, obs = ensure_dataset_same_variables(pred, obs)
    return pred, obs


def deterministic(
    pred,
    obs,
    forecast_type="continuous",
    aggregating_dim=None,
    # TODO: to refactor name
    skip_na=True,
    skip_infs=True,
    skip_zeros=True,
):
    """Compute deterministic skill metrics."""
    # Check input arguments
    pred, obs = ensure_common_xarray_format(pred, obs)
    aggregating_dim = check_aggregating_dim(aggregating_dim)
    forecast_type = check_forecast_type(forecast_type)

    check_validity_aggregating_dim(aggregating_dim, pred)
    check_validity_aggregating_dim(aggregating_dim, obs)

    # Check that obs dims is equal or subset of pred dims
    # TODO:

    # Reshape Datasets to DataArray
    # pred = ensure_dataarray(pred)
    # obs = ensure_dataarray(obs)

    # Align Datasets
    pred, obs = align_xarray_objects(pred, obs)

    # Broadcast obs to pred
    # pred, obs = xr.broadcast(pred, obs)

    # Retrieve xarray routine
    _xr_routine = _get_xr_routine(metric_type="deterministic", forecast_type=forecast_type)

    # Compute skills
    ds_skill = _xr_routine(
        pred=pred,
        obs=obs,
        dims=aggregating_dim,
        skip_na=skip_na,
        skip_infs=skip_infs,
        skip_zeros=skip_zeros,
    )
    return ds_skill


def probabilistic():
    """Compute probabilistic skill metrics."""


def spatial():
    """Compute spatial skill metrics."""


# -----------------------------------------------------------------------------.
