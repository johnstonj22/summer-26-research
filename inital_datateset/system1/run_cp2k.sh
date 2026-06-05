#!/bin/bash
#SBATCH --partition=bigjay,sixhour,thompson
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=9
#SBATCH -C ib,intel
#SBATCH -t 6:00:00
#SBATCH -o logs/CP2K.%j
#SBATCH --mem=400G
#SBATCH -J CP2K_1

# Settings
MAX_CHUNKS=10
MAX_RETRIES=3

CHUNK=${CHUNK:-1}
RETRY=${RETRY:-0}

INPUT_FILE="cp2k.inp"
OUTPUT_FILE="silica.out"
CP2K_EXE="cp2k.psmp"
RESTART_SCRIPT="run_cp2k_restart.sh"


mkdir -p logs

echo "======================================"
echo "Starting initial chunk $CHUNK of $MAX_CHUNKS"
echo "Retry $RETRY of $MAX_RETRIES"
echo "Job ID: $SLURM_JOB_ID"
echo "Host(s): $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

module purge
module load cp2k/2025.1

mpirun -n "${SLURM_NTASKS}" "${CP2K_EXE}" -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"
CP2K_EXIT=$?

echo "CP2K exit code: $CP2K_EXIT"
echo "End time: $(date)"

if [ "$CP2K_EXIT" -eq 0 ]; then
    echo "Chunk $CHUNK succeeded."

    if [ "$CHUNK" -lt "$MAX_CHUNKS" ]; then
        NEXT_CHUNK=$((CHUNK + 1))
        echo "Submitting restart chunk $NEXT_CHUNK with retry reset to 0."
        sbatch --export=ALL,CHUNK="$NEXT_CHUNK",RETRY=0 "$RESTART_SCRIPT"
    else
        echo "Reached maximum of $MAX_CHUNKS chunks. Done."
    fi
else
    NEW_RETRY=$((RETRY + 1))
    echo "Chunk $CHUNK failed."
    echo "Retry count for chunk $CHUNK is now $NEW_RETRY of $MAX_RETRIES."

    if [ "$NEW_RETRY" -le "$MAX_RETRIES" ]; then
        echo "Resubmitting initial chunk $CHUNK."
        sbatch --export=ALL,CHUNK="$CHUNK",RETRY="$NEW_RETRY" run_cp2k_initial.sh
    else
        echo "Maximum retries reached for initial chunk $CHUNK. Stopping chain."
        exit 1
    fi
fi