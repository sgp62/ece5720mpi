#!/bin/bash
mpirun -np 8 ./sgp62_jacobi --mca opal_warn_on_missing_libcuda 0 #note -np XX must equal ntasks from the .sub
