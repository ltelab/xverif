#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:11:33 2021.

@author: ghiggi
"""
import os
import shutil

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from rechunker import rechunk

##########################
#### Compressor utils ####
##########################


def is_numcodecs(compressor):
    """Check is a numcodec compressor."""
    if type(compressor).__module__.find("numcodecs") == -1:
        return False
    return True


def check_compressor(compressor, variable_names, default_compressor=None):
    """Check compressor validity for zarr writing.

    compressor = None --> No compression.
    compressor = "auto" --> Use default_compressor if specified. Otherwise will default to ds.to_zarr() default compressor.
    compressor = <numcodecs class> --> Specify the same compressor to all Dataset variables
    compressor = {..} --> A dictionary specifying a compressor for each Dataset variable.

    default_compressor: None or numcodecs compressor. None will default to ds.to_zarr() default compressor.
    variable_names: list of xarray Dataset variables
    """
    # Check variable_names type
    if not isinstance(variable_names, (list, str)):
        raise TypeError("'variable_names' must be a string or a list")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s, str) for s in variable_names]):
        raise TypeError(
            "Specify all variable names as string within the 'variable_names' list."
        )

    # Check compressor type
    if not (
        isinstance(compressor, (str, dict, type(None))) or is_numcodecs(compressor)
    ):
        raise TypeError(
            "'compressor' must be a dictionary, numcodecs compressor, 'auto' string or None."
        )
    if isinstance(compressor, str):
        if not compressor == "auto":
            raise ValueError("If 'compressor' is specified as string, must be 'auto'.")
    if isinstance(compressor, dict):
        if not np.all(np.isin(list(compressor.keys()), variable_names)):
            raise ValueError(
                "The 'compressor' dictionary must contain the keys {}".format(
                    variable_names
                )
            )

    # Check default_compressor type
    if not (
        isinstance(default_compressor, (dict, type(None)))
        or is_numcodecs(default_compressor)
    ):
        raise TypeError("'default_compressor' must be a numcodecs compressor or None.")
    if isinstance(default_compressor, dict):
        if not np.all(np.isin(list(default_compressor.keys()), variable_names)):
            raise ValueError(
                "The 'default_compressor' dictionary must contain the keys {}".format(
                    variable_names
                )
            )
    ##------------------------------------------------------------------------.
    # If a string --> "Auto" --> Apply default_compressor (if specified)
    if isinstance(compressor, str):
        if compressor == "auto":
            compressor = default_compressor

    ##------------------------------------------------------------------------.
    # If a dictionary, check keys validity and compressor validity
    if isinstance(compressor, dict):
        if not all(
            [
                is_numcodecs(cmp) or isinstance(cmp, type(None))
                for cmp in compressor.values()
            ]
        ):
            raise ValueError(
                "The compressors specified in the 'compressor' dictionary must be numcodecs (or None)."
            )
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if is_numcodecs(compressor) or isinstance(compressor, type(None)):
        compressor = {var: compressor for var in variable_names}

    ##------------------------------------------------------------------------.
    return compressor


# -----------------------------------------------------------------------------.
######################
#### Chunks utils ####
######################


def _all_valid_chunks_values(values):
    """Checks chunks value validity."""
    bool_list = []
    for x in values:
        if isinstance(x, str):
            if x == "auto":
                bool_list.append(True)
            else:
                bool_list.append(False)
        elif isinstance(x, int):
            bool_list.append(True)
        elif isinstance(x, type(None)):
            bool_list.append(True)  # Require caution
        else:
            bool_list.append(False)
    return all(bool_list)


def get_ds_chunks(ds):
    """Get dataset chunks dictionary."""
    variable_names = list(ds.data_vars.keys())
    chunks = {}
    for var in variable_names:
        if ds[var].chunks is not None:
            chunks[var] = {dim: v[0] for dim, v in zip(ds[var].dims, ds[var].chunks, strict=True)}
        else:
            chunks[var] = None
    return chunks



def _check_all_keys(dictionary, keys, error_message):
    """Check a dictionary contains all the keys."""
    if not np.all(np.isin(list(dictionary.keys()), keys)):
        raise ValueError(error_message)


def _check_chunks_dicts(chunks, default_chunks, variable_names, dim_names):
    """Check chunks dictionaries keys validity."""
    _check_all_keys(chunks, variable_names, "Please specify specific chunks for each Dataset variable.")
    _check_all_keys(chunks, dim_names, "Please specify specific chunks for each Dataset dimension.")
    _check_all_keys(default_chunks, variable_names, "Please specify specific default_chunks for each Dataset variable.")
    _check_all_keys(default_chunks, dim_names, "Please specify specific default_chunks for each Dataset dimension.")


def _define_chunks_auto(ds, default_chunks):
    """"Define chunks if 'auto'."""
    # If default_chunks is a dict, assign to chunks
    if isinstance(default_chunks, dict):
        chunks = default_chunks
    # If default_chunks is None, assign "auto" to all dimensions
    else:
        chunks = {dim: "auto" for dim in list(ds.dims)}
    return chunks


def check_chunks(ds, chunks, default_chunks=None):
    """Check chunks validity.

    chunks = None --> Keeps current chunks.
    chunks = "auto" --> Use default_chunks if specified, otherwise defaults to xarray "auto" chunks.
    chunks = {.-.}  -->  Custom chunks
    # - Option 1: A dictionary of chunks definitions for each Dataset variable
    # - Option 2: A single chunk definition to be applied to all Dataset variables
    # --> Attention: -1 and None are equivalent chunk values !!!

    default_chunks is used only if chunks = "auto"
    """
    # Check chunks
    if not isinstance(chunks, (str, dict, type(None))):
        raise TypeError("'chunks' must be a dictionary, 'auto' or None.")
    if isinstance(chunks, str):
        if not chunks == "auto":
            raise ValueError("If 'chunks' is specified as string, must be 'auto'.")
    # Check default chunks
    if not isinstance(default_chunks, (dict, type(None))):
        raise TypeError("'default_chunks' must be either a dictionary or None.")
    # Check variable_names
    if not isinstance(ds, xr.Dataset):
        raise TypeError("'ds' must be an xarray Dataset.")
    # -------------------------------------------------------------------------.
    # Retrieve Dataset infos
    variable_names = list(ds.data_vars.keys())
    dim_names = list(ds.dims)
    # -------------------------------------------------------------------------.
    # Retrieve chunks and default_chunks formats when provided as dictionary
    if isinstance(chunks, dict):
        _check_chunks_dicts(chunks=chunks,
                            default_chunks=default_chunks,
                            variable_names=variable_names,
                            dim_names=dim_names)
    ##------------------------------------------------------------------------.
    # If chunks = "auto"
    if isinstance(chunks, str):
        chunks = _define_chunks_auto(ds, default_chunks)

    ##------------------------------------------------------------------------.
    # If chunks = None
    if isinstance(chunks, type(None)):
        chunks = get_ds_chunks(ds)

    ##------------------------------------------------------------------------.
    # If a dictionary, check chunks valid keys and values
    if isinstance(chunks, dict):
        is_chunks_per_variable=True
        # TODO: is_chunks_dict_per_variable
        # TODO: is_chunks_dict_per_dimension
        # If 'chunks' specify specific chunks for each Dataset variable
        if is_chunks_per_variable:
            # - For each variable
            for var in variable_names:
                # - Check that the chunk value for each dimension is specified
                if not np.all(np.isin(list(chunks[var].keys()), list(ds[var].dims))):
                    raise ValueError(
                        "The 'chunks' dictionary of {} must contain the keys {}".format(
                            var, list(ds[var].dims)
                        )
                    )
                # - Check chunks value validity
                if not _all_valid_chunks_values(list(chunks[var].values())):
                    raise ValueError("Unvalid 'chunks' values for {}.".format(var))
        # If 'chunks' specify specific chunks for all Dataset dimensions
        elif not is_chunks_per_variable:
            # - Checks chunks value validity
            if not _all_valid_chunks_values(list(chunks.values())):
                raise ValueError("Unvalid 'chunks' values")
            # - Create dictionary for each variable
            new_chunks = {}
            for var in variable_names:
                new_chunks[var] = {dim: chunks[dim] for dim in ds[var].dims}
            chunks = new_chunks
        else:
            raise ValueError("This chunks option has not been implemented.")
    ##------------------------------------------------------------------------.
    return chunks


def sanitize_chunks_dict(chunks_dict, ds):
    """Sanitize chunks dictionary.

    Change chunk value '-1' to length of the dataset dimension
    Rechunk and zarr do not currently support -1 specification used by dask and xarray.
    """
    dict_dims = dict(ds.dims)
    for var in chunks_dict.keys():
        if chunks_dict[var] is not None:
            for k, v in chunks_dict[var].items():
                if v == -1:
                    chunks_dict[var][k] = dict_dims[k]
    return chunks_dict


# -----------------------------------------------------------------------------.
#########################
#### Rechunker wrapper ##
#########################
def rechunk_Dataset(ds, chunks, target_store, temp_store, max_mem="1GB", force=False):
    """
    Rechunk on disk a xarray Dataset read lazily from a zarr store.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset opened with open_zarr().
    chunks : dict
        Custom chunks of the new Dataset.
        If not specified for each Dataset variable, implicitly assumed.
    target_store : str
        Filepath of the zarr store where to save the new Dataset.
    temp_store : str
        Filepath of a zarr store where to save temporary data.
        This store is removed at the end of the rechunking operation.
    max_mem : str, optional
        The amount of memory (in bytes) that workers are allowed to use.
        The default is '1GB'.

    Returns
    -------
    None.

    """
    # TODO
    # - Add compressors options
    # compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    # options = dict(compressor=compressor)
    # rechunk(..., target_options=options)

    ##------------------------------------------------------------------------.
    # Check target_store do not exist already
    if os.path.exists(target_store):
        if force:
            shutil.rmtree(target_store)
        else:
            raise ValueError(
                f"A zarr store already exists at {target_store}. If you want to overwrite, specify force=True"
            )

    ##------------------------------------------------------------------------.
    # Remove temp_store if still exists
    if os.path.exists(temp_store):
        shutil.rmtree(temp_store)

    ##------------------------------------------------------------------------.
    # Check chunks
    target_chunks = check_chunks(ds=ds, chunks=chunks, default_chunks=None)
    target_chunks = sanitize_chunks_dict(target_chunks, ds)

    ##------------------------------------------------------------------------.
    # Plan rechunking
    r = rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )

    ##------------------------------------------------------------------------.
    # Execute rechunking
    with ProgressBar():
        r.execute()

    ##------------------------------------------------------------------------.
    # Remove temporary store
    shutil.rmtree(temp_store)
    ##------------------------------------------------------------------------.





# -----------------------------------------------------------------------------.
