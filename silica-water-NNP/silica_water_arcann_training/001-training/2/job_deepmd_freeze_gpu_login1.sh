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
# You must keep the name file as job_deepmd_freeze_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# Project/Account
#SBATCH --partition=sixhour
#SBATCH -C nvidia
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o DeepMD_Freeze.%j
#SBATCH -e DeepMD_Freeze.%j
# Name of job
#SBATCH -J DeepMD_Freeze
#SBATCH --mem=150G
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

DeepMD_MODEL_VERSION="3.0"
DeepMD_MODEL_FILE="graph_2_001.pb"
DeepMD_CKPT_FILE="checkpoint"
DeepMD_LOG_FILE="graph_2_001_freeze.log"
DeepMD_OUT_FILE="graph_2_001_freeze.out"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f ${DeepMD_CKPT_FILE} ] || { echo "${DeepMD_CKPT_FILE} does not exist. Aborting..."; exit 1; }

module purge
module load conda/latest

source /kuhpc/sw/conda/latest/etc/profile.d/conda.sh
conda activate /kuhpc/work/thompson/j281j388/.conda/envs/deepmd-kit-3.0
export PATH="${CONDA_PREFIX}/bin:${PATH}"

NVIDIA_LIBS=$(find "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export DP_INFER_BATCH_SIZE=8192

# Run the DeepMD freeze
echo "# [$(date)] Running DeepMD freeze..."
dp freeze -o ${DeepMD_MODEL_FILE} --log-path ${DeepMD_LOG_FILE} > ${DeepMD_OUT_FILE} 2>&1
echo "# [$(date)] DeepMD freeze finished."

sleep 2
exit
