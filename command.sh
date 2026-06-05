#!/bin/bash
#SBATCH --job-name=copy_rename
#SBATCH --partition=bigjay,thompson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=command%j.out
#SBATCH --error=command%j.err

# Always start in the directory where you ran `sbatch`
cd "$SLURM_SUBMIT_DIR"

# Load Python (adjust to match how you usually run Python on the cluster)
module purge
module load StdEnv
module load conda/latest
eval "$(/kuhpc/sw/conda/latest/bin/conda shell.bash hook)"
conda activate myenv   # <-- change to your env name if different

du -sh .
echo 'done'