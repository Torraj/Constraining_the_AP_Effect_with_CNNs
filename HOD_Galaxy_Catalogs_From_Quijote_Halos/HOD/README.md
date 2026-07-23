# pyHOD

pyHOD is a Python3 library for constructing galaxy mock catalogs based on the standard Halo Occupation Distribution (HOD) model, as described in the paper by [_Zheng et al. (2007)_](https://iopscience.iop.org/article/10.1086/521074/pdf). The library utilizes an analytic description to determine the radius of the galaxies in the halo, following the method proposed by [_Robotham & Howlett (2018)_](https://iopscience.iop.org/article/10.3847/2515-5172/aacc70).


## Installation

Required Python packages:

- `numpy`
- `scipy`
- `sklearn`
- `numba`

To install using pip, navigate to the root directory and run the following command:

> `python -m pip install .`

If you already have an existing installation and want to update the package after pulling the latest changes, use the following command:

> `python -m pip install --upgrade .`

## Usage

This library provides tools for drawing galaxies within halos using the NFW profile.

The code expects a Halo catalog with a shape of (n, 7), where each row represents a halo and the columns correspond to the halo parameters: mass, pos_x, pos_y, pos_y, vel_x, vel_y, vel_z. The library generates a galaxy mock catalog that includes the following information for each galaxy:

- Galaxy 3D-position 
- Galaxy 3D-velocity
- Galaxy type: 1 for central galaxy, 0 for satallite galaxy.
- Halo Parent: A numeric label indicating the group of galaxies contained within the parent halo.


A basic implementation of pyHOD is available in the tutorials:

- Notebook version: `Examples/notebook-example.pynb` 

- Script version: `Examples/pyhod_test.py`. You can execute it from a bash script `Scripts/script_test.sh`

## Additional tools

This version of pyHOD includes the following utility functions:

- `displaceRSD`: Displaces particles to account for redshift space distortion.

- `M_star`: Computes the typical collapsing mass parameter.
