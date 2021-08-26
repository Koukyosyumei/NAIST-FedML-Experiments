#!/bin/sh
#$ -S /bin/bash
#$ -q grid_short.q
#$ -pe mpi 1056

module load compiler/gcc/7
module load mpi/openmpi/3.0.0

export MPICC=$(which mpicc)
printenv | grep MPICC

mpirun -np 44 -npernode 1 ps aux | grep FedAvg
mpirun -np 44 -npernode 1 pkill FedAvg
