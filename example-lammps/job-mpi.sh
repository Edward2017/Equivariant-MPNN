#!/bin/sh
#PBS -V
#PBS -q sugon10
#PBS -N 6.2A
#PBS -l nodes=1:ppn=24
#export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=24
source /share/home/bjiangch/group-zyl/.bash_profile
#export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH #for specify the cuda path to overcome error: failed to open libnvrtc-builtins.so.11.1
cd $PBS_O_WORKDIR
mpirun -n 1 ./lmp_mpi -in in.h2o >out
