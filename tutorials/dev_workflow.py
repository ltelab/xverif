#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:02:13 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
import xverif
from xverif.datasets import (
    create_ensemble_dataset,
    create_ensemble_forecast_dataset,
    # create_multimodel_dataset,
    # create_multimodel_ensemble_forecast_dataset,
    create_spatial2d_dataset,
    # create_spatial3d_dataset,
    create_timeseries_dataset,
    create_timeseries_forecast_dataset,
)
from xverif.wrappers import align_xarray_objects

# Simulate obs and pred
obs = create_spatial2d_dataset(obs_number=1000)

pred = create_ensemble_dataset(obs_number=1000)
pred = create_ensemble_forecast_dataset(obs_number=1000)


# Broadcasting obs to pred
pred1, obs1 = xr.broadcast(pred, obs)
np.testing.assert_allclose(
    obs1.isel({"realization": 0})["var1"].data,
    obs1.isel({"realization": 1})["var1"].data,
)

# Align datasets
pred2, obs2 = align_xarray_objects(pred, obs)

####--------------------------------------------------------------------------.
#### Compute deterministic skills (time series dataset)
# --> (time: 100, stations: 100)
obs = create_timeseries_dataset(100)
pred = create_timeseries_forecast_dataset(100)

skip_options = None
skip_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    # {"equal_values": False},
    {"values": 0, "conditioned_on": "both"},
    {"below_threshold": 1.5, "conditioned_on": "both"},
    # {"above_threshold": 3, "conditioned_on": "obs"},
]

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim="time",
    implementation="loop",
    skip_options=skip_options
)


ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim="time",
    implementation="vectorized", 
    skip_options=skip_options
)

# Convert Dataset skills to Dataset Variable
ds_skills.to_array(dim="skill").to_dataset(dim="variable")

####--------------------------------------------------------------------------.
#### Compute deterministic skills (spatial ensemble forecast)
# --> (time: 100, x: 10, y: 10, realization: 24, leadtime: 48)

obs = create_spatial2d_dataset(1000)
pred = create_ensemble_forecast_dataset(1000)

obs = create_spatial2d_dataset(100)
pred = create_ensemble_forecast_dataset(100)

forecast_type = "continuous"
aggregating_dim = ["x", "y"]
dims = aggregating_dim
skip_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    {"values": 0, "conditioned_on": "both"},
    {"below_threshold": 1.5, "conditioned_on": "both"},
]
implementation = "vectorized"
metrics = None
metrics = ["MAE", "rob_MAE"]

# Chunk finely on auxiliary dimension if going out of memory !
pred = pred.chunk(
    {
        "x": -1,
        "y": -1,
        # "realization": 1,
        "leadtime": 1,
    }
)
obs = obs.chunk({"x": -1, "y": -1})

# TODO: check auxiliary dimensions are chunked      (do not rechunk ... but chunk to parallelize computations)
# TODO: check that aggregating dims are not chunked (rechunk to -1)

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim=["x", "y"],
    implementation=implementation,
    metrics=metrics,
    # Preprocessing
    skip_options=skip_options,
)

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim=["x", "y"],
    implementation=implementation,
    metrics=metrics,
    compute=False,
    # Preprocessing
    skip_options=skip_options,
)

ds_skills.data.visualize()



####--------------------------------------------------------------------------.
from xverif.metrics.deterministic.continuous_vectorized import get_available_metrics

d = get_available_metrics()

# To avoid creating the large chunks, set the option
#  with dask.config.set(**{'array.slicing.split_large_chunks': True}) :


# Emulate xr_apply_ufunc dimension reordering in continuous ndarray
# pred = pred.transpose(..., "x","y")["var1"].data
# obs = np.expand_dims(obs.transpose(..., "x","y")["var1"].data, axis=[1,2])
# axis = [3, 4]
# pred.shape
# obs.shape


implementations = ["loop", "vectorized"]

for implementation in implementations:
    print(f"Implementation: {implementation}")
    ds_skills = xverif.deterministic(
        pred=pred,
        obs=obs,
        forecast_type="continuous",
        aggregating_dim=["x", "y"],
        implementation=implementation,
        skip_options=[],
    )


###----------------------------------------------------------------------------
#### DEBUG of vectorized code
# from xverif.metrics.deterministic.continuous_vectorized import get_stacking_dict
# obs_broadcasted = obs.broadcast_like(pred)
# stacking_dict = get_stacking_dict(pred, aggregating_dim=dims)
# pred = pred.stack(stacking_dict)
# obs = obs_broadcasted.stack(stacking_dict)


# pred = pred.data
# obs = obs.data
