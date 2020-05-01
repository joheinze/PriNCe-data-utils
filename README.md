# PriNCe-data-utils
Data tables and utilities to construct the database file for PriNCe

The [PriNCe](https://github.com/joheinze/PriNCe) UHECR propagation code required a number of tabulated
data for photon fields, photo-nuclear and photo-hadronic interactions to run. This collection of utilities
gives everybody access to change all the details of these data or add new models.

These tools were developed and used mainly in [Heinze et al., Astrophys.J. 873 (2019)](https://doi.org/10.3847/1538-4357/ab05ce)

## Requirements

This is a light pure python code. No specific architecture requirements.

Dependencies (list might be incomplete):

- [*PriNCe* propagation code](https://github.com/joheinze/PriNCe)
- python-3.7 or later
- numpy
- scipy
- matplotlib
- astropy
- hdf5py
- jupyter notebook or jupyter lab (optional, but needed for the plotting example)

## Usage

The main script will generate an HDF database file that can be copied to prince-dir/prince/data

## Experimenting with data

A number of notebooks used for cross-checking the data and demonstrating output are located in the notebooks directory. Have a look.

## Citation

If you are using this code in your work, please cite:

*A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models*  
J. Heinze, A. Fedynitch, D. Boncioli and W. Winter  
[Astrophys.J. 873 (2019) no.1, 88](https://doi.org/10.3847/1538-4357/ab05ce)

## Author

- Jonas Heinze
- Anatoli Fedynitch

## Copyright and license

Licensed under BSD 3-Clause License (see LICENSE.md)
