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

_get_continuous_metrics(pred, obs)

##-----------------------------------------------------------------------------.
