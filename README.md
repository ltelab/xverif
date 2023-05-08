# xverif - A package to perform forecast verification builds upon xarray.

The code in this repository provides a scalable and flexible framework to perform forecast verification.
It principally builds upon xarray, dask and flox libraries.

ATTENTION: The code is subject to changes in the coming months.

The folder `tutorials` (will) provide jupyter notebooks describing various features of xverif.

The folder `docs` (will) contains slides and notebooks explaining the x-forecasting framework.

## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/ghiggi/xverif.git
   cd xverif
   ```

2. Install manually the following dependencies:
   ```sh
   conda create --name xverif-dev python=3.8
   conda install xarray dask cdo h5py h5netcdf netcdf4 zarr numcodecs rechunker
   conda install notebook jupyterlab
   conda install numpy pandas numba scipy bottleneck
   ```

2. Alternatively install the dependencies using one of the appropriate below
   environment.yml files:
   ```sh
   conda env create -f TODO.yml
   ```

## Tutorials

## Reproducing our results

## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
* [Yann Yasser Haddad](https://www.linkedin.com/in/yann-yasser-haddad)

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
