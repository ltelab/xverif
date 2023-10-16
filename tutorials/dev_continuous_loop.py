#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:28:13 2023

@author: ghiggi
"""
### Patterns to search and adapt from vectorized to loop 
# ,axis=axis
# axis=axis 
# nan 
# , )
# np.expand_dims

import numpy as np
from xverif.metrics.deterministic.continuous_loop import _get_metrics

pred = np.arange(0,10)
obs = np.arange(0, 10)

metrics = _get_metrics(pred, obs, drop_options=None)
len(metrics)
 
