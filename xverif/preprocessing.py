#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:39:34 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr


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


def time_at_inittime_leadtime(
    obj: xr.DataArray | xr.Dataset,
    init_time_name: str = "forecast_reference_time",
    lead_time_name: str = "lead_time",
):
    """Compute a 2d array of the actual time given the initialization and lead times."""
    reftime = obj[init_time_name].values
    leadtime = obj[lead_time_name].values

    if (dtype := leadtime.dtype) != np.dtype("timedelta64[ns]"):
        raise ValueError(
            f"{lead_time_name} coordinate must be of dtype 'np.timedelta' but is {dtype}"
        )

    return xr.DataArray(
        reftime[:, None] + leadtime[None],
        coords={init_time_name: reftime, lead_time_name: leadtime},
        dims=[init_time_name, lead_time_name],
    )


def temporal_broadcast_observations(
    obs: xr.DataArray | xr.Dataset,
    fcts: xr.DataArray | xr.Dataset,
    obs_time_name: str = "time",
    fcts_init_time_name: str = "forecast_reference_time",
    fcts_lead_time_name: str = "lead_time",
):
    """Align temporal coordinates of observations to those of forecasts.

    Conform an observation object that has a single temporal dimension,
    representing the actual time of the observation, to a forecast object
    that has two temporal dimensions representing the forecast initialization
    time and the lead time (i.e. the temporal hozizon of the forecast).

    Parameters
    ----------
    obs: xr.DataArray or xr.Dataset
        The observation object, where the temporal dimension must have a coordinate
        array of dtype `np.datetime64[ns]`.
    fcts: xr.DataArray or xr.Dataset
        The forecast object, where the forecast initialization and lead time dimensions must have
        coordinate arrays of dtype `np.datetime64[ns]` and `np.timedelta64[ns]` respectively.
    obs_time_tame: str
        The name of the temporal dimension on the observation object.
    fcts_init_time_name: str
        The name of the initialization time dimension on the forecast object.
    fcts_lead_time_name: str
        The name of the lead time dimension on the forecast object.

    Returns
    -------
    obs: xr.DataArray or xr.Dataset
    """
    indexer = time_at_inittime_leadtime(fcts, fcts_init_time_name, fcts_lead_time_name)
    try:
        obs = obs.sel({obs_time_name: indexer})
    except KeyError:
        obs = obs.reindex(time=np.unique(indexer.values.reshape(-1)))
        obs = obs.sel(time=indexer)
    fcts = fcts.assign_coords({obs_time_name: indexer})
    return obs, fcts
