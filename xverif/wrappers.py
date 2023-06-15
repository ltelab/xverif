#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:48:21 2023.

@author: ghiggi
"""
import typing as tp
from importlib import import_module

import xarray as xr

ValidForecastType = tp.Literal[
    "continuous", "categorical_binary", "categorical_multiclass"
]
ValidMetricType = tp.Literal["deterministic", "probabilistic", "spatial"]


def _check_args(pred, obs) -> None:
    if not (isinstance(xr.DataArray, pred) and isinstance(xr.DataArray, obs)):
        raise ValueError(
            "'pred' and 'obs' arguments must be xarray DataArray instances. Got "
            f"{type(pred)} and {type(obs)} types respectively."
        )
    if len(set(pred.dims) & set(obs.dims)) == 0:
        raise ValueError(
            "'pred' and 'obs' Datasets do not have any variable in common."
        )


def _check_shared_dims(dims, pred, obs):
    dims = set(dims)
    if dims < set(fdim := pred.dims) and dims < set(odim := obs.dims):
        pass
    else:
        raise ValueError(
            f"The aggregating dimensions {dims} must be present in both forecast "
            f"and observation objects, but they have {fdim} and {odim} respectively."
        )


def _check_forecast_type(forecast_type: str) -> None:
    if forecast_type not in (valid := tp.get_args(ValidForecastType)):
        raise ValueError(
            f"{forecast_type} is not a valid forecast type. Must be one of {valid}."
        )


def _check_metric_type(metric_type):
    """Check metric_type validity."""
    if not isinstance(metric_type, str):
        raise TypeError("'metric_type' must be a string.")
    if metric_type not in tp.get_args(ValidMetricType):
        raise ValueError(f"Valid 'metric_type' are {tp.get_args(ValidMetricType)}")
    return metric_type


def _get_xr_routine(metric_type, forecast_type, implementation="stacked"):
    """Retrieve xarray routine to compute the metrics."""
    # Check inputs
    forecast_type = _check_forecast_type(forecast_type)
    metric_type = _check_metric_type(metric_type)
    # Define module path
    # TODO: module_path = f"xverif.metrics.{metric_type}.{forecast_type}
    module_path = f"xverif.metrics.{metric_type}.{forecast_type}_{implementation}"
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
            raise ValueError(
                "Please do not provide a xr.Dataset with a dimension named 'variable'."
            )
        xr_obj = xr_obj.to_array(dim="variable")
    return xr_obj


def align_xarray_objects(pred, obs):
    """Align xarray objects.

    - Ensure coordinate alignment for matching dimensions.
    - Ensure common subset of Dataset variables.
    """
    # Align xarray dimensions
    pred, obs = xr.align(pred, obs, join="inner")

    return pred, obs


def deterministic(
    pred,
    obs,
    forecast_type="continuous",
    aggregating_dim=None,
    implementation="stacked",
    # TODO: to refactor name
    skip_na=True,
    skip_infs=True,
    skip_zeros=True,
):
    """Compute deterministic metrics."""
    aggregating_dim = list(aggregating_dim)

    _check_args(pred, obs)
    _check_shared_dims(aggregating_dim)
    _check_forecast_type(forecast_type)

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
    _xr_routine = _get_xr_routine(
        metric_type="deterministic",
        forecast_type=forecast_type,
        implementation=implementation,
    )

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
