#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:20:29 2023.

@author: ghiggi
"""
from xverif.datasets import create_ensemble_forecast_dataset, create_spatial2d_dataset
from xverif.metrics.deterministic.continuous_vectorized import (
    _get_metrics,
    get_available_metrics,
    get_metrics,
    get_stacking_dict,
)
from xverif.wrappers import ensure_dataarray

sample_dims = ["x", "y"]
metrics = get_available_metrics()

obs = create_spatial2d_dataset(1000)
pred = create_ensemble_forecast_dataset(1000)


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

print(pred.chunks)


# Reshaping operations
pred = ensure_dataarray(pred)
obs = ensure_dataarray(obs)


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
stacking_dict = get_stacking_dict(pred, sample_dims=sample_dims)
pred = pred.stack(stacking_dict)
obs = obs.stack(stacking_dict)

# Visualize dask operations here
obs.data.visualize()

# Compute metrics on xr.DataArray
a = get_metrics(pred.data, obs.data, metrics=metrics)


# Compute metrics on dask array
a = get_metrics(pred.data, obs.data, metrics=metrics)

a_dict = _get_metrics(pred.data, obs.data)

# Visualize tasks
a.visualize()
