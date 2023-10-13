#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:44:07 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr


def _get_function_keys(): 
    keys = ['nan', 'inf', 'equal_values', 'values', 'above_threshold', 'below_threshold']
    return keys


def _check_conditioned_on(conditioned_on):
    if conditioned_on not in ["obs", "pred", "any", "both"]:
        raise ValueError("conditioned_on must be one of 'obs', 'pred', 'any', 'both'.")
    return conditioned_on
    

def _check_masking_option_keys(masking_option):
    keys = list(masking_option.keys())
    if len(keys) > 2:
        raise ValueError("A masking option must contain at most two keys.")
    if len(keys) == 2: 
        if "conditioned_on" not in keys: 
            raise ValueError("A masking option with two keys must contain 'conditioned_on' as one of the keys.")
    valid_keys = ['nan', 'inf', 'equal_values', 'values', 'above_threshold', 'below_threshold', 'conditioned_on']
    unvalid_keys =  [key for key in keys if key not in valid_keys]
    if len(unvalid_keys) > 0:
        raise ValueError(f"Invalid masking option keys: {unvalid_keys}. Valid keys are {valid_keys}.")


def _check_masking_option_values(masking_option):
    valid_values_dict = {
        'nan': [True, False],
        'inf': [True, False],
        'equal_values': [True, False],
        'values': (int, float, list, np.ndarray),
        'above_threshold': (int, float),
        'below_threshold': (int, float),
        'conditioned_on': ["obs", "pred", "any", "both"],
    } 
    for key, value in masking_option.items():
        if key in ['nan', 'inf', 'equal_values', 'conditioned_on']:
            if value not in valid_values_dict[key]:
                valid_values = valid_values_dict[key]
                raise ValueError(f"Invalid masking option '{key}' key value. Valid values are {valid_values}.")
        elif key in ['above_threshold', 'below_threshold', 'values']:
            if not isinstance(value, valid_values_dict[key]):
                valid_values = valid_values_dict[key]
                raise TypeError(f"Invalid type for masking option '{key}' key value'. Valid types are {valid_values}.")


def _set_default_conditioned_on_value(masking_options):
    default_values = {
        "nan": "any",
        "inf": "any",
        "equal_values": "both", # dummy 
        "values": "both",
        "above_threshold": "both",
        "below_threshold": "both",
    }
    for i, masking_option in enumerate(masking_options): 
        if "conditioned_on" not in masking_option:
            key = list(masking_option)[0]
            masking_options[i]["conditioned_on"] = default_values[key]
    return masking_options

def _check_masking_option(masking_option):
    _check_masking_option_keys(masking_option)
    _check_masking_option_values(masking_option)
    

def is_valid_masking_option(masking_option): 
    try: 
        _check_masking_option(masking_option)
        validity = True
    except Exception: 
        validity = False
    return validity
        
    
def check_masking_options(masking_options): 
    if isinstance(masking_options, type(None)):
        return []
    if isinstance(masking_options, dict):
        if is_valid_masking_option(masking_options):
            masking_options = [masking_options]
        else: 
            raise TypeError("Specify masking_options as a list of dictionaries.")
    if not isinstance(masking_options, list):
        raise TypeError("masking_options must be a list of dictionaries.")
    for masking_option in masking_options:
       _check_masking_option(masking_option)
    masking_options = _set_default_conditioned_on_value(masking_options)
    return masking_options


def _get_masking_option_func_key(masking_option):
    keys = list(masking_option)
    valid_func_keys = _get_function_keys()
    keys = [key for key in keys if key in valid_func_keys]
    if len(keys) != 1: 
        raise ValueError(f"A masking option must contain only one of the following keys: {valid_func_keys}'.")
    return keys[0]


class MaskingDataArrays:
    def __init__(self, pred, obs, masking_options=None, masking_value = np.nan):
        masking_options = check_masking_options(masking_options)
        # TODO: check masking_value same type of pred and obs 
        self.pred = pred
        self.obs = obs
        self.masking_value = masking_value
        self.masking_options = masking_options
        
    def apply(self):
        if len(self.masking_options) == 0: 
            return self.pred, self.obs 

        masked_pred = self.pred.copy()
        masked_obs = self.obs.copy()

        func_dict = {
            'nan': self.mask_nan,
            'inf': self.mask_inf,
            'equal_values': self.mask_equal_values,
            'values': self.mask_values,
            'above_threshold': self.mask_above_threshold,
            'below_threshold': self.mask_below_threshold,
        }

        for masking_option in self.masking_options:
            option_kwargs = masking_option.copy()
            func_key = _get_masking_option_func_key(masking_option)
            func = func_dict[func_key]
        
            if func_key in ["nan", "inf", "equal_values"]: 
                if not masking_option[func_key]: 
                    continue # do not execute masking if False
                option_kwargs.pop(func_key, None)    

            if "masking_value" not in option_kwargs: 
                option_kwargs["masking_value"] = self.masking_value
            
            # Apply masking 
            masked_pred, masked_obs = func(masked_pred, 
                                           masked_obs,
                                           **option_kwargs)

        return masked_pred, masked_obs
    
    def mask_nan(self, pred, obs, conditioned_on, masking_value):
        condition_map = {
            "obs": lambda: np.isnan(obs),
            "pred": lambda: np.isnan(pred),
            "any": lambda: np.logical_or(np.isnan(obs), np.isnan(pred)),
            "both": lambda: np.logical_and(np.isnan(obs), np.isnan(pred))
        }
        isnan = condition_map[conditioned_on]()
        pred = pred.where(~isnan, other=masking_value)
        obs = obs.where(~isnan, other=masking_value)
        return pred, obs
    
    def mask_inf(self, pred, obs, conditioned_on, masking_value):
        condition_map = {
            "obs": lambda: np.isinf(obs),
            "pred": lambda: np.isinf(pred),
            "any": lambda: np.logical_or(np.isinf(obs), np.isinf(pred)),
            "both": lambda: np.logical_and(np.isinf(obs), np.isinf(pred))
        }
        isinf = condition_map[conditioned_on]()
        pred = pred.where(~isinf, other=masking_value)
        obs = obs.where(~isinf, other=masking_value)
        return pred, obs
    
    def mask_equal_values(self, pred, obs, masking_value, conditioned_on="dummy"):     
        isequal = pred == obs
        pred = pred.where(~isequal, other=masking_value)
        obs = obs.where(~isequal, other=masking_value)
        return pred, obs
 
    def mask_values(self, pred, obs, values, conditioned_on, masking_value):
        if isinstance(values, (int, float)): 
            values = [values]
        condition_map = {
            "obs": lambda: np.isin(obs, values),
            "pred": lambda: np.isin(pred, values),
            "any": lambda: np.logical_or(np.isin(obs, values), np.isin(pred, values)),
            "both": lambda: np.logical_and(np.isin(obs, values), np.isin(pred, values))
        }
        isvalues = condition_map[conditioned_on]()
        pred = pred.where(~isvalues, other=masking_value)
        obs = obs.where(~isvalues, other=masking_value)
        return pred, obs
    
    def mask_above_threshold(self, pred, obs, above_threshold, conditioned_on, masking_value):
        condition_map = {
            "obs": lambda: obs > above_threshold,
            "pred": lambda: pred > above_threshold,
            "any": lambda: np.logical_or(obs > above_threshold, pred > above_threshold),
            "both": lambda: np.logical_and(obs > above_threshold, pred > above_threshold)
        }
        isabove = condition_map[conditioned_on]()
        pred = pred.where(~isabove, other=masking_value)
        obs = obs.where(~isabove, other=masking_value)
        return pred, obs 
        
    def mask_below_threshold(self, pred, obs, below_threshold, conditioned_on, masking_value):
        condition_map = {
            "obs": lambda: obs < below_threshold,
            "pred": lambda: pred < below_threshold,
            "any": lambda: np.logical_or(obs < below_threshold, pred < below_threshold),
            "both": lambda: np.logical_and(obs < below_threshold, pred < below_threshold)
        }
        isbelow = condition_map[conditioned_on]()
        pred = pred.where(~isbelow, other=masking_value)
        obs = obs.where(~isbelow, other=masking_value)
        return pred, obs
    
####--------------------------------------------------------------------------.
#### Wrappers


def mask_dataarrays(pred, obs, masking_options):
    pred, obs = MaskingDataArrays(pred, obs, masking_options=masking_options).apply()
    return pred, obs


def mask_datasets(pred, obs, masking_options): 
    pass 

# Dataset case 
# - Different options per variable ! 
#     --> {var1: {}, var2: {}}
#     --> var not specified set to default masking options 
