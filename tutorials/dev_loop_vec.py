#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:28:13 2023.

@author: ghiggi
"""
##-----------------------------------------------------------------------------.
### Patterns to search and adapt from vectorized to loop
# ,axis=axis
# axis=axis
# nan
# , )
# np.expand_dims

##-----------------------------------------------------------------------------.
#### Loop metrics
import numpy as np
from xverif.metrics.deterministic.binary_loop import _get_metrics as _get_binary_metrics
from xverif.metrics.deterministic.continuous_loop import (
    _get_metrics as _get_continuous_metrics,
)

pred = np.arange(0, 10)
obs = np.arange(0, 10)

metrics = _get_continuous_metrics(pred, obs, drop_options=None)


pred = np.array([0, 1, 0, 1, 1])
obs = np.array([0, 1, 0, 1, 1])

metrics = _get_binary_metrics(pred, obs, drop_options=None)

len(metrics)
 

##-----------------------------------------------------------------------------.
#### Vectorized metrics
import numpy as np
from xverif.metrics.deterministic.binary_vectorized import (
    _get_metrics as _get_binary_metrics,
)
from xverif.metrics.deterministic.continuous_vectorized import (
    _get_metrics as _get_continuous_metrics,
)

# Vectorized
pred = np.zeros((5, 10))
obs = np.zeros((5, 10))

_get_binary_metrics(pred, obs)

pred = np.arange(0, 50).reshape(5,10)
obs = np.arange(0, 50).reshape(5,10)


_get_continuous_metrics(pred, obs)


import dask.array 
axis = 1
obs_q25, obs_Median, obs_q75 = np.nanquantile(obs, q=[0.25, 0.50, 0.75], axis=axis)
obs_q25


axis = 1
obs = dask.array.from_array(obs, chunks=("auto", -1))
obs.chunks
obs_q25, obs_Median, obs_q75 = np.nanquantile(obs, q=[0.25, 0.50, 0.75], axis=axis)
obs_q25, obs_Median, obs_q75 = dask_nanquantiles(obs, q=[0.25, 0.50, 0.75], axis=axis)

def dask_nanquantiles(arr, q, axis):
    
    def _compute_quantiles(arr, axis, q):
        return np.nanquantile(arr, q=q, axis=axis).T
    
    # Check that the presence of a single chunk along the axis
    if len(arr.chunks[axis]) > 1:
        raise ValueError(f"Array has multiple chunks along axis {axis}. Ensure a single chunk.")
    
    # Define output chunk 
    output_chunks = list(arr.chunks)
    output_chunks[axis] = (len(q),)
      
    # Compute quantiles using map_blocks
    quantiles = dask.array.map_blocks(_compute_quantiles, arr, 
                                      axis=axis, q=q,
                                      dtype=arr.dtype, 
                                      chunks=output_chunks)    
    return quantiles

a = dask_nanquantiles(obs, q=[0.25, 0.50, 0.75], axis=axis)
a.shape
a.compute().shape 

obs_q25, obs_Median, obs_q75 = a
# 5 x 3 




##-----------------------------------------------------------------------------.
