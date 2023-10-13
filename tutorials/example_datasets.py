#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:53:54 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from xverif.datasets import (
    create_ensemble_dataset,
    create_ensemble_forecast_dataset,
    create_multimodel_dataset,
    create_multimodel_ensemble_forecast_dataset,
    create_spatial2d_dataset,
    create_spatial3d_dataset,
    create_timeseries_dataset,
    create_timeseries_forecast_dataset,
)

# Simulate datasets
create_timeseries_dataset(1000, data_type="continuous")  # default
create_timeseries_dataset(1000, data_type="categorical", n_class=3)
create_timeseries_dataset(1000, data_type="probability", n_class=3)
create_timeseries_forecast_dataset(1000)

create_spatial2d_dataset(obs_number=1000)
create_spatial3d_dataset(obs_number=1000)
create_ensemble_forecast_dataset(obs_number=1000)
create_multimodel_dataset(obs_number=1000)
create_multimodel_ensemble_forecast_dataset(obs_number=1000)


create_spatial2d_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10),
    aux_dims=["x", "y"],
    n_class=2,
    n_vars=3,
)

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
from xverif.wrappers import align_xarray_objects

pred2, obs2 = align_xarray_objects(pred, obs)


# Compute deterministic skills (1 sample_dims)
import xverif
from xverif.datasets import (
    create_timeseries_dataset,
    create_timeseries_forecast_dataset,
)

obs = create_timeseries_dataset(100)
pred = create_timeseries_forecast_dataset(100)

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim="time",
    skip_na=True,
    skip_infs=True,
    skip_zeros=True,
)


ds_skills.to_array(dim="variables").to_dataset(dim="skill")


# Compute deterministic skills (2 sample_dims)
import xverif
from xverif.datasets import create_ensemble_forecast_dataset, create_spatial2d_dataset

# obs = create_spatial2d_dataset(15)
# pred = create_spatial2d_dataset(15)
# pred = create_ensemble_forecast_dataset(15)

obs = create_spatial2d_dataset(100)
pred = create_ensemble_forecast_dataset(100)

obs = create_spatial2d_dataset(1000)
pred = create_ensemble_forecast_dataset(1000)


forecast_type = "continuous"
aggregating_dim = ["x", "y"]
dims = aggregating_dim
skip_na = True
skip_infs = True
skip_zeros = True
kwargs = {}
implementation = "stacked"
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


ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim=["x", "y"],
    implementation=implementation,
    metrics=metrics,
    # Preprocessing
    skip_na=True,
    skip_infs=True,
    skip_zeros=True,
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
    skip_na=True,
    skip_infs=True,
    skip_zeros=True,
)


ds_skills.data.visualize()


# TODO: check auxiliary dimensions are chunked      (do not rechunk ... but chunk to parellelize computations)
# TODO: check that aggregating dims are not chunked (rechunk to -1)


# Remove warnings
from xverif.metrics.deterministic.continuous_stacked import get_available_metrics

d = get_available_metrics()


# To avoid creating the large chunks, set the option
#  with dask.config.set(**{'array.slicing.split_large_chunks': True}) :


# Emulate xr_apply_ufunc dimension reordering in continuous ndarray
# pred = pred.transpose(..., "x","y")["var1"].data
# obs = np.expand_dims(obs.transpose(..., "x","y")["var1"].data, axis=[1,2])
# axis = [3, 4]
# pred.shape
# obs.shape


implementations = ["loop", "stacked", "ndarray"]

for implementation in implementations:
    print(f"Implementation: {implementation}")
    ds_skills = xverif.deterministic(
        pred=pred,
        obs=obs,
        forecast_type="continuous",
        aggregating_dim=["x", "y"],
        implementation=implementation,
        skip_na=True,
        skip_infs=True,
        skip_zeros=True,
    )


###----------------------------------------------------------------------------
### DEBUG of stacked code
# from xverif.metrics.deterministic.continuous_stacked import get_stacking_dict
# obs_broadcasted = obs.broadcast_like(pred)
# stacking_dict = get_stacking_dict(pred, aggregating_dim=dims)
# pred = pred.stack(stacking_dict)
# obs = obs_broadcasted.stack(stacking_dict)


# pred = pred.data
# obs = obs.data
