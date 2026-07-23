import argparse
import numpy as np

## Notebook configuration setup:
## ----------------------------

def input_params_nb():
  params={}

  # Halo catalog parameters:
  simulation_parameters={
          'box'    : 1000,             # BOX SIZE OF THE SIMULATIONS IN MPC/H
          'grid'   : 256,              # SIZE OF THE FFT GRID
          'z'      : 0.5,              # REDSHIFT
          'Deltac' : 200,              # overdensity constant \Delta_c
          'M_star' : 1.670992e+13,     # typical collapsing mass
          'cat'    : 'Quijote_Halo',   # Catalog type. Valid entries: 'Eos', 'Quijote_Halo'
          'om'     : 0.3175,           # OMEGA MATTER
          'ol'     : 0.6825,           # OMEGA LAMDBA
          'h'      : 0.6711}           # HUBBLE CONSTANT 
  params.update(simulation_parameters)

  # HOD model setup
  HOD_parameters = {
          'method'    : 'NFW',          #valid entries:'NFW', 'NFWv', 'SNAP', 'SNAPknn'
          'model'     : 'Zheng',        #valid entries: 'Zheng','exponential'
          'logMmin'   : 13.0,
          'alpha'     : 0.75,
          'logM1'     : 14.25,
          'sigmalogM' : 0.2,
          'logM0'     : 13.1,
          'vel_bias'  : 1.0
          }
  params.update(HOD_parameters)

  # Output parameters
  file_parameters = {
          'ifile'     : 'catalog.dat',  # input filename
          'ifile_snapshot'  : 'snapshot',     # snapshot filename, used if 'method' = 'SNAP' or 'SNAPknn'
          'outfile'   : 'output.npz',   # output filename
          'iRSD'      : 0,              # Redshift Space Displacement on the output file: 
                                        #   0: Real space, 1,2,3: ALONG THE X,Y,Z AXIS, RESPECTIVELY
          'zerr'      : 0,              # Redshift error type: 0:None, 1:spectroscopic, 2:photometric
          'seed'      : 5761,           # Seed the random number generator
          }
  params.update(file_parameters)

  return params

## Script configuration setup:
## --------------------------

def input_params():
  params={}
  simulation_parameters={
          'box'    : 1000,
          'grid'   : 256,
          'z'      : 0.5,
          'Deltac' : 200,
          'M_star' : 1.670992e+13,
          'cat'    : 'Quijote_Halo', 
          'om'     : 0.3175,
          'ol'     : 0.6825,
          'h'      : 0.6711}
  params.update(simulation_parameters)
  HOD_parameters = {
          'method'    : 'NFW',
          'model'     : 'Zheng',
          'logMmin'   : 13.0,
          'alpha'     : 0.75,
          'logM1'     : 14.25,
          'sigmalogM' : 0.2,
          'logM0'     : 13.1
          }
  params.update(HOD_parameters)

  HOD_steps = {
          'nsteps'       : 1,
          'st_logMmin'   : 0.035,
          'st_alpha'     : 0.3,
          'st_logM1'     : 0.3,
          'st_sigmalogM' : 0.035,
          'st_logM0'     : 0.3
          }
  params.update(HOD_steps)

  args = vars(parse_arguments())
  params.update(args)

  return params

## bash output parameters setup:
## ----------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ifile', required=True, default = "catalog.dat", help = 'input file')
    parser.add_argument('--snapshot', default = "snapshot", help = 'snapshot file')
    parser.add_argument('--outfile', required=True, default = "output.npz", help = 'Output file')
    parser.add_argument('--iRSD', default = 0, type = int,help='Redshift Space Displacement. 0: Real space,\
                                1,2,3: ALONG THE X,Y,Z AXIS, RESPECTIVELY, 5: wide angle observer')
    parser.add_argument('--zerr', default = 0, type = int,help='redshift error type: 0:None, 1:spectroscopic, 2:photometric')
    parser.add_argument('--seed', default = 5761, type = int,help='seed the random number generator.')

    args = parser.parse_args()

    return args