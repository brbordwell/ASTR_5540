Doubly-diffusive convection problems for ASTR/ATOC 5540 *Mathematical Methods* and for ASTR 5410 *Fluids, Instabilities, Waves, and Turbulence*.

To use:

Install the full Dedalus stack (see https://bitbucket.org/dedalus-project/dedalus2/pull-request/23/wip-install-script/diff and https://groups.google.com/d/msg/dedalus-users/VoCpUDP4kCI/kRTvqgBFs6sJ).  

The latest version of the install script is located at:

https://bitbucket.org/dedalus-project/dedalus2/raw/tip/docs/install.sh

Once Dedalus is installed and activated, do the following:
```
#!bash
python3 TS.py
python3 plot_results_parallel.py TS slices 1 1 10
```
This can be run in parallel, using:
```
#!bash
mpirun -np $NCORES_RUN python3 TS.py
mpirun -np $NCORES_VIZ python3 plot_results_parallel.py TS slices 1 1 10
```
where NCORES_RUN evenly divides the number of Nz modes (here, 16; must use at most 16/2), and NCORES_VIZ evenly divides the number of output slice files (here, 10; can use up to 10).  Valid values include NCORES_RUN=8 and NCORES_VIZ=10.

The solution should achieve a class of waves.  Are they standing or travelling?  To modify the behaviour, change parameters in TS.py.