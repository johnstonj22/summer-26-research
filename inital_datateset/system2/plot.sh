#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --output=sa.out
#SBATCH --partition=bigjay,thompson,sixhour
#SBATCH --constraint=avx512
#SBATCH --exclude=r16r22n01,r16r21n01,r16r31n01

module purge
module load StdEnv
module load conda/latest
eval "$(/kuhpc/sw/conda/latest/bin/conda shell.bash hook)"
conda activate myenv
srun python plot_energies.py