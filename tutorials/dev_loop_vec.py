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

pred = np.arange(0, 50).reshape(5, 10)
obs = np.arange(0, 50).reshape(5, 10)


_get_continuous_metrics(pred, obs)

#### - Dask based
import dask.array

axis = 1
obs = dask.array.from_array(obs, chunks=("auto", -1))
pred = dask.array.from_array(obs, chunks=("auto", -1))


#### Multiclass
pred = np.array([2, 0, 2, 0, 1])
obs = np.array([0, 0, 2, 0, 2])
n_categories = 4

obs = np.array([0, 1, 2, 3])
pred = np.array([0, 2, 1, 3])

obs = np.array([0, 1, 0, 0, 1, 0])
pred = np.array([0, 1, 0, 0, 0, 1])


##-----------------------------------------------------------------------------.
