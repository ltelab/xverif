#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:02:13 2023.

@author: ghiggi
"""
import xarray as xr
import xverif
from xverif.datasets import (
    create_ensemble_forecast_dataset,
    create_spatial2d_dataset,
    create_timeseries_dataset,
    create_timeseries_forecast_dataset,
)

####--------------------------------------------------------------------------.
#### Compute deterministic skills (time series dataset)
# --> (time: 100, stations: 100)
data_type = "continuous"
n_categories = None

data_type = "binary"
n_categories = 2

data_type = "multiclass"
n_categories = 4


obs = create_timeseries_dataset(100, data_type=data_type, n_categories=n_categories)
pred = create_timeseries_forecast_dataset(
    100, data_type=data_type, n_categories=n_categories
)

# pred["var0"].data


sample_dims = "time"
skip_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
]

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    data_type=data_type,
    sample_dims=sample_dims,
    implementation="loop",
    skip_options=skip_options,
    n_categories=n_categories,
)


ds_skills1 = xverif.deterministic(
    pred=pred,
    obs=obs,
    data_type=data_type,
    sample_dims=sample_dims,
    implementation="vectorized",
    skip_options=skip_options,
    n_categories=n_categories,
)


for var in ds_skills.data_vars:
    xr.testing.assert_allclose(ds_skills[var], ds_skills1[var])
    # xr.testing.assert_equal(ds_skills[var], ds_skills1[var])
    # ds_skills[var] - ds_skills1[var]


# Convert Dataset skills to Dataset Variable
ds_skills.to_array(dim="skill").to_dataset(dim="variable")

####--------------------------------------------------------------------------.
#### Compute deterministic skills (spatial ensemble forecast)
# --> (time: 100, x: 10, y: 10, realization: 24, leadtime: 48)

obs = create_spatial2d_dataset(1000)
pred = create_ensemble_forecast_dataset(1000)

obs = create_spatial2d_dataset(100)
pred = create_ensemble_forecast_dataset(100)

data_type = "continuous"
data_type = "binary"
sample_dims = ["x", "y"]
skip_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    {"values": 0, "conditioned_on": "both"},
    {"below_threshold": 1.5, "conditioned_on": "both"},
]

implementation = "loop"
implementation = "vectorized"


metrics = ["MAE", "rob_MAE"]
metrics = None

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
    data_type=data_type,
    sample_dims=sample_dims,
    implementation=implementation,
    metrics=metrics,
    # Preprocessing
    skip_options=skip_options,
)

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    data_type=data_type,
    sample_dims=sample_dims,
    implementation=implementation,
    metrics=metrics,
    compute=False,
    # Preprocessing
    skip_options=skip_options,
)

ds_skills.data.visualize()


####--------------------------------------------------------------------------.
implementations = ["loop", "vectorized"]
for implementation in implementations:
    print(f"Implementation: {implementation}")
    ds_skills = xverif.deterministic(
        pred=pred,
        obs=obs,
        data_type="continuous",
        sample_dims=["x", "y"],
        implementation=implementation,
        skip_options=[],
    )
