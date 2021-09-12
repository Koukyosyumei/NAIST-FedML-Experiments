#!/bin/sh
#$ -S /bin/bash
#$ -q pascal_short.q
#$ -pe mpi 144

module load compiler/gcc/7
module load mpi/openmpi/3.0.0

export MPICC=$(which mpicc)
printenv | grep MPICC

mpirun -np 12 -npernode 2 ps aux | grep FedAvg
mpirun -np 12 -npernode 2 pkill FedAvg
