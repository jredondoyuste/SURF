# SURF
Spectral Ultra-relativistic Fluids: Code to study the dynamics of null horizons and their Carrollian fluid duals. 

## Requirements:

The packages that might not be pre-installed and that are needed to run SURF are:
1. TQDM, for Command-Line visualization of the progress https://github.com/tqdm/tqdm.git
2. QUADPY, to compute the spherical integrals https://github.com/sigma-py/quadpy.git

## Use: 

Before starting to use the code, do the following:

1. Create an empty folder called 'SphIntegrals'. 
2. Cretae an empty folder called 'Data'. 

The first script (SphericalIntegrals.py) needs to be run once in order to compute some integrals involving spin-weighted spherical harmonics (SWSH) and store them in the folder 'SphIntegrals' as pkl files for faster memory access. 

The second script (Def-Geom-Eqs.py) contains the evolution itself. The code considers a small perturbation on a Schwarzschild black hole and solves the Einstein equations up to second order in the perturbation, and under the quasispherical approximation, in the near-horizon gauge for the horizon geometry. Moreover, it computes the evolution of the Carrollian fluid dual using the correspondance described in Laura Donnay and Charles Marteau 2019 Class. Quantum Grav. 36 165002. The outputs of the evolution are produced as .dat files stored in the folder 'Data'. 

## Inputs:

There are several inputs that can be modified:

1. Regarding the evolution: 
  - t0: Final time of the evolution
  - h: Stepsize (recommended at least h=0.1)
  - ltop: Number of physical angular modes (recommended at least l=3 for convergence. Past l=4 the evolution is slow). 

2. Regarding the source:
  - tS: Time at which the source peaks.
  - coefLM: Excitation coefficients of the (l, m) = (2, 0) / (2, \pm 1) / (2, \pm 2) QNMs. 
  - sigmaS: Width of the Gaussian pulse that is convoluted with the source.  
