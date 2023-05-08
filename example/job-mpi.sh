#!/bin/bash
#SBATCH -J MD-40
#SBATCH -N 1
##SBATCH -n 64
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -p hfacnormal02
#SBATCH -t 60000
##SBATCH --mem-per-cpu=1G
#SBATCH -o out
#SBATCH -e job.err
##SBATCH -n 24
#SBATCH --exclusive
#SBATCH --no-requeue

echo Running on hosts
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
conda_env=pt200
export OMP_NUM_THREADS=128

cd $PWD


#The path you place your code
path="/public/home/bjiangch/zyl/Equi-MPNN/"
#This command to run your pytorch script


python $path > out
