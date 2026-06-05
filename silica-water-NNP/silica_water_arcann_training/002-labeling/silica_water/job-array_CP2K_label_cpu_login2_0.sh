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
# You must keep the name file as job-array_CP2K_label_ARCHTYPE_myHPCkeyword1.sh.
#----------------------------------------------
# QoS/Partition/SubPartition
##SBATCH --qos=myqos
#SBATCH --partition=bigjay,sixhour
#SBATCH -C ib,intel
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 18
#SBATCH --cpus-per-task 1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t 6:00:00
# Merge Output/Error
#SBATCH -o CP2K.%A_%a
#SBATCH -e CP2K.%A_%a
# Name of job
#SBATCH -J CP2K_silica_water_002
#SBATCH --mem=80G
# Array
#SBATCH --array=0-31%250
#SBATCH --nodelist=r16r07n01,r16r08n01,r16r09n01,r16r10n01,r16r11n01,r16r17n01,r16r18n01,r16r19n01,r16r20n01,r16r21n01,r16r22n01,r16r27n01,r16r28n01,r16r29n01,r16r30n01,r16r31n01,r17r07n01,r17r08n01,r17r09n01,r17r10n01,r17r11n01,r17r17n01,r17r18n01,r17r19n01,r17r20n01,r17r21n01,r17r22n01,r17r27n01,r17r28n01,r17r29n01,r17r30n01,r17r31n01,r18r07n01,r18r08n01,r18r09n01,r18r10n01,r18r11n01,r18r17n01,r18r18n01,r18r19n01,r18r20n01,r18r21n01,r18r22n01,r18r27n01,r18r28n01,r18r29n01,r18r30n01,r18r31n01,r19r07n01,r19r08n01,r19r09n01,r19r10n01,r19r11n01,r19r17n01,r19r18n01,r19r19n01,r19r20n01,r19r21n01,r19r22n01,r19r27n01,r19r28n01,r19r29n01,r19r30n01,r20r07n01,r20r08n01,r21r21n01,r21r22n01,r21r27n01,r21r28n01,r21r29n01,r21r30n01,r31r05n01,r31r10n01,r31r15n01,r31r20n01,r31r25n01,r31r35n01,r11r20n04,r11r22n01,r11r22n02,r11r28n02,r11r28n03,r11r28n04,r11r30n01,r11r30n02,r11r30n03,r11r30n04,r12r06n01,r12r08n01,r12r10n01,r12r12n03,r12r18n01,r12r20n01,r12r22n01,r12r26n01,r12r28n01,r12r30n01,r13r06n01,r13r08n01,r13r10n01,r13r12n01,r13r18n01,r13r20n01,r13r22n01,r13r28n01,r13r28n03,r13r28n04,r13r30n01,r13r30n02,r13r30n03,r13r30n04,r14r12n01,r14r18n01,r14r20n01,r14r20n02,r14r22n04,r14r28n02,r14r28n04,r14r30n04,r14r38n03,r14r38n04
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------
SLURM_ARRAY_TASK_ID_LARGE=$((SLURM_ARRAY_TASK_ID + 0))
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID_LARGE}")

CP2K_IN_FILE1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.inp"
CP2K_OUT_FILE1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.out"
CP2K_IN_FILE2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.inp"
CP2K_OUT_FILE2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.out"
CP2K_XYZ_FILE="labeling_${SLURM_ARRAY_TASK_ID_PADDED}.xyz"
CP2K_WFRST_FILE="labeling_${SLURM_ARRAY_TASK_ID_PADDED}-SCF.wfn"

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_CP2K_label_ARCHTYPE_myHPCkeyword1.sh
# Don't forget to replace the job_labeling_array_ARCHTYPE_myHPCkeyword1.sh at the end of the file (replacling ARCHTYPE and myHPCkeyword1)
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }

module purge
#module load compiler/intel/24
#module load openmpi
module load cp2k/2025.1

MAX_CP2K_RETRIES=3
CP2K_STAGE1_TIMEOUT="4h"
CP2K_STAGE2_TIMEOUT="2h"
CP2K_STAGE1_DONE=".cp2k_stage1.done"
CP2K_STAGE2_DONE=".cp2k_stage2.done"
CP2K_STAGE1_RETRIES=".cp2k_stage1.retries"
CP2K_STAGE2_RETRIES=".cp2k_stage2.retries"

SUBMIT_SCRIPT=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | awk -F= '/Command=/{print $2; exit}')
if [ -z "${SUBMIT_SCRIPT}" ] || [ ! -f "${SUBMIT_SCRIPT}" ]; then
SUBMIT_SCRIPT="${SLURM_SUBMIT_DIR:-$PWD}/$(basename "$0")"
fi

find_local_candidate_script() {
find "${PWD}" -maxdepth 1 -type f -name 'job_CP2K_label_*_cpu_login2.sh' -print -quit
}

resubmit_current_job() {
local local_submit_script=""
local local_status=0

local_submit_script=$(find_local_candidate_script)
if [ -n "${local_submit_script}" ] && [ -f "${local_submit_script}" ]; then
echo "# [$(date)] Resubmitting local candidate job ${local_submit_script}"
sbatch "${local_submit_script}"
local_status=$?
if [ "${local_status}" -ne 0 ]; then
echo "Job resubmission failed. Aborting..."
exit 1
fi
exit 0
fi

if [ ! -f "${SUBMIT_SCRIPT}" ]; then
echo "Could not find submit script ${SUBMIT_SCRIPT}. Aborting..."
exit 1
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
echo "# [$(date)] Resubmitting array task ${SLURM_ARRAY_TASK_ID} from ${SUBMIT_SCRIPT}"
(cd "${SLURM_SUBMIT_DIR}" && sbatch --array="${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_TASK_ID}" "${SUBMIT_SCRIPT}")
else
echo "# [$(date)] Resubmitting job from ${SUBMIT_SCRIPT}"
(cd "${SLURM_SUBMIT_DIR}" && sbatch "${SUBMIT_SCRIPT}")
fi

if [ $? -ne 0 ]; then
echo "Job resubmission failed. Aborting..."
exit 1
fi
exit 0
}

retry_or_stop() {
local stage_name="$1"
local retry_file="$2"
local retries=0

if [ -f "${retry_file}" ]; then
retries=$(cat "${retry_file}")
fi
retries=$((retries + 1))
echo "${retries}" > "${retry_file}"

if [ "${retries}" -le "${MAX_CP2K_RETRIES}" ]; then
echo "# [$(date)] ${stage_name} failed or timed out. Retry ${retries}/${MAX_CP2K_RETRIES}; resubmitting."
resubmit_current_job
fi

echo "# [$(date)] ${stage_name} failed after ${MAX_CP2K_RETRIES} retries. Aborting..."
exit 1
}

cleanup_stage_outputs() {
local in_file="$1"
local out_file="$2"
local stem="${in_file%.inp}"

echo "# [$(date)] Removing stale output files for ${in_file}."
rm -f "${out_file}"
rm -f "${stem}-Force_Eval.fe" "${stem}-Forces.for"
rm -f "${stem}-${CP2K_WFRST_FILE}"*.wfn
}

validate_cp2k_stage() {
local stage_name="$1"
local out_file="$2"
local retry_file="$3"

if ! grep -q "PROGRAM ENDED" "${out_file}"; then
echo "${stage_name} did not reach PROGRAM ENDED."
retry_or_stop "${stage_name}" "${retry_file}"
fi

if grep -q "SCF run NOT converged" "${out_file}"; then
echo "${stage_name} finished but SCF did not converge."
retry_or_stop "${stage_name}" "${retry_file}"
fi

if ! grep -q "SCF run converged" "${out_file}"; then
echo "${stage_name} finished but no SCF convergence line was found."
retry_or_stop "${stage_name}" "${retry_file}"
fi
}

run_cp2k_stage() {
local stage_name="$1"
local in_file="$2"
local out_file="$3"
local retry_file="$4"
local stage_timeout="$5"

cleanup_stage_outputs "${in_file}" "${out_file}"
echo "# [$(date)] Running ${stage_name} with timeout ${stage_timeout}..."
timeout --kill-after=60s "${stage_timeout}" mpirun -n "${SLURM_NTASKS}" cp2k.psmp -i "${in_file}" > "${out_file}"
local status=$?

if [ "${status}" -eq 124 ] || [ "${status}" -eq 137 ]; then
echo "${stage_name} exceeded ${stage_timeout}."
retry_or_stop "${stage_name}" "${retry_file}"
elif [ "${status}" -ne 0 ]; then
echo "${stage_name} failed with exit code ${status}."
retry_or_stop "${stage_name}" "${retry_file}"
fi

validate_cp2k_stage "${stage_name}" "${out_file}" "${retry_file}"
}

find_wfn_restart() {
local in_file="$1"
ls -t "${in_file%.inp}-${CP2K_WFRST_FILE}"*.wfn 2>/dev/null | head -n 1
}

cleanup_wavefunction_copies() {
echo "# [$(date)] Removing extra CP2K wavefunction restart copies."
rm -f "1_${CP2K_WFRST_FILE}" "2_${CP2K_WFRST_FILE}"
rm -f "${CP2K_IN_FILE1%.inp}-${CP2K_WFRST_FILE}"*.wfn
rm -f "${CP2K_IN_FILE2%.inp}-${CP2K_WFRST_FILE}"*.wfn
}

if [ -f "${CP2K_STAGE1_DONE}" ]; then
if [ ! -f "${CP2K_WFRST_FILE}" ] && [ -f "1_${CP2K_WFRST_FILE}" ]; then
cp "1_${CP2K_WFRST_FILE}" "${CP2K_WFRST_FILE}"
fi
if [ -f "${CP2K_WFRST_FILE}" ]; then
echo "# [$(date)] CP2K first job already completed; skipping to second job."
else
echo "Stage 1 marker exists, but no usable wavefunction restart was found. Rerunning stage 1."
rm -f "${CP2K_STAGE1_DONE}"
fi
fi

if [ ! -f "${CP2K_STAGE1_DONE}" ]; then
run_cp2k_stage "CP2K first job" "${CP2K_IN_FILE1}" "${CP2K_OUT_FILE1}" "${CP2K_STAGE1_RETRIES}" "${CP2K_STAGE1_TIMEOUT}"
FIRST_WFN=$(find_wfn_restart "${CP2K_IN_FILE1}")
if [ -z "${FIRST_WFN}" ]; then
echo "No first-stage CP2K wavefunction restart file found."
retry_or_stop "CP2K first job" "${CP2K_STAGE1_RETRIES}"
fi
cp "${FIRST_WFN}" "${CP2K_WFRST_FILE}"
cp "${FIRST_WFN}" "1_${CP2K_WFRST_FILE}"
touch "${CP2K_STAGE1_DONE}"
echo "# [$(date)] CP2K first job finished."
fi

if [ -f "${CP2K_STAGE2_DONE}" ]; then
echo "# [$(date)] CP2K second job already completed."
else
run_cp2k_stage "CP2K second job" "${CP2K_IN_FILE2}" "${CP2K_OUT_FILE2}" "${CP2K_STAGE2_RETRIES}" "${CP2K_STAGE2_TIMEOUT}"
SECOND_WFN=$(find_wfn_restart "${CP2K_IN_FILE2}")
if [ -z "${SECOND_WFN}" ]; then
echo "No second-stage CP2K wavefunction restart file found."
retry_or_stop "CP2K second job" "${CP2K_STAGE2_RETRIES}"
fi
cp "${SECOND_WFN}" "2_${CP2K_WFRST_FILE}"
touch "${CP2K_STAGE2_DONE}"
cleanup_wavefunction_copies
echo "# [$(date)] CP2K second job finished."
fi

rm -f "${CP2K_STAGE1_RETRIES}" "${CP2K_STAGE2_RETRIES}"

sleep 2
exit
