#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/05/15
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# QoS/Partition/SubPartition
#SBATCH --partition=sixhour,bigjay
#SBATCH -C nvidia&ib
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o LAMMPS_DeepMD.%j
#SBATCH -e LAMMPS_DeepMD.%j
# Name of job
#SBATCH -J LAMMPS_DeepMD
#SBATCH --mem=50G
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------

DeepMD_MODEL_VERSION="3.0"
DeepMD_MODEL_FILES=("graph_1_001_compressed.pb" "graph_2_001_compressed.pb" "graph_3_001_compressed.pb")
LAMMPS_IN_FILE="silica_water_1_002.in"
LAMMPS_LOG_FILE="silica_water_1_002.log"
LAMMPS_OUT_FILE="silica_water_1_002.out"
EXTRA_FILES=("001_silica_water_2_00003_00002.lmp")

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }

module purge
module load conda/latest

source /kuhpc/sw/conda/latest/etc/profile.d/conda.sh
conda activate /kuhpc/work/thompson/e497b540/.conda/deepmd-kit
export PATH="${CONDA_PREFIX}/bin:${PATH}"

NVIDIA_LIBS=$(find "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export DP_INFER_BATCH_SIZE=8192

# Example if your run in a scratch folder
export TEMPWORKDIR=./JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/JOB-${SLURM_JOBID}"

cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied successfully"
for f in "${DeepMD_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running LAMMPS..."
lmp -in "${LAMMPS_IN_FILE}" -log "${LAMMPS_LOG_FILE}" -screen none > "${LAMMPS_OUT_FILE}" 2>&1
echo "# [$(date)] LAMMPS finished."

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit
