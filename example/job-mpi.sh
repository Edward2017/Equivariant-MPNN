#!/bin/sh
#SBATCH -J h2o-1
#SBATCH --gpus=1
#SBATCH -N 1
##SBATCH --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
echo Running on hosts
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
# Your conda environment
export OMP_NUM_THREADS=8

module add cuda/11.7

#ATTENTION! HERE MUSTT BE ONE LINE,OR ERROR!
source ~/.bashrc
source activate pt200
module load gcc/9.3
module load intel/mkl/2019
cd $PWD

path="/data/home/scv2201/run/zyl/program/Equi-MPNN/"
python3 $path >out
