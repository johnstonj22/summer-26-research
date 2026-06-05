#!/bin/bash
#SBATCH --partition=bigjay,sixhour,thompson
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 9
# Walltime
#SBATCH -C ib,intel
#SBATCH -t 6:00:00
#SBATCH -o logs/CP2K.%j
#SBATCH --mem=400G
# Name of job
#SBATCH -J CP2K_3

# Settings
MAX_CHUNKS=10
CHUNK=${CHUNK:-1}
INPUT_FILE="cp2k.inp"
OUTPUT_FILE="silica.out"
CP2K_EXE="cp2k.psmp"

mkdir -p logs

echo "======================================"
echo "Starting chunk $CHUNK of $MAX_CHUNKS"
echo "Job ID: $SLURM_JOB_ID"
echo "Host(s): $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

module purge
module load cp2k/2025.1

mpirun -n ${SLURM_NTASKS} ${CP2K_EXE} -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"

CP2K_EXIT=$?

echo "CP2K exit code: $CP2K_EXIT"
echo "End time: $(date)"

if [ "$CHUNK" -lt "$MAX_CHUNKS" ]; then
    NEXT_CHUNK=$((CHUNK + 1))
    echo "Submitting next chunk: $NEXT_CHUNK"

    sbatch --export=ALL,CHUNK=$NEXT_CHUNK run_cp2k_restart.sh
else
    echo "Reached maximum of $MAX_CHUNKS chunks. Done."
fi