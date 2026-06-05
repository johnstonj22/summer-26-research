#!/bin/bash
#SBATCH --partition=bigjay
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 9
# Walltime
#SBATCH -t 6:00:00
#SBATCH -o logs/CP2K.%j
#SBATCH --mem=400G
# Name of job
#SBATCH -J CP2K_r2
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------

CP2K_IN_FILE1="cp2k_restart.inp"
CP2K_OUT_FILE1="silica.out"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

module purge
module load cp2k/2025.1

echo "# [$(date)] Running CP2K first job..."
mpirun -n "${SLURM_NTASKS}" cp2k.psmp -i "${CP2K_IN_FILE1}" -o "${CP2K_OUT_FILE1}"
echo "# [$(date)] CP2K first job finished."

sleep 2
exit
