#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:26:23 2023

@author: ghiggi
"""

####--------------------------------------------------------------------------.
##############
#### TODO ####
##############
#### - Defaults choice
# - aggregating_dim --> None?
# - Change name of aggregating_dim
# - Change name of sample and aux dimensions

####--------------------------------------------------------------------------.
#### Daskify functions

####--------------------------------------------------------------------------.
#### - Suppress warnings
"invalid value encountered in divide"  # division by 0
"Degrees of freedom <= 0 for slice"  # (when all np.nan in nanstd)
"All-NaN slice encountered"
"invalid value encountered in double_scalars"

# sklearn way to deal with division by 0
# https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/metrics/_classification.py#L1301

