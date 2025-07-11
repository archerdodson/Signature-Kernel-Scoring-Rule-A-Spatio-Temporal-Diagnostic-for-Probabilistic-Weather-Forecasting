#!/bin/bash

#SBATCH --job-name=SigKernel
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Archer.Dodson@warwick.ac.uk


module purge
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1

#module load SciPy-bundle  

source ~/newenv4/bin/activate  #Change environment

#pip show zarr numcodecs xarray-beam rechunker pandas torch weatherbench2

srun python ProbScoreCard.py

deactivate
