export OMP_NUM_THREADS=8

module add cuda/11.7

source ~/.bashrc
source activate pt200
module load gcc/9.3
module load intel/mkl/2019

path="/data/home/scv2201/run/zyl/program/Equi-MPNN/"
python3 $path >out
