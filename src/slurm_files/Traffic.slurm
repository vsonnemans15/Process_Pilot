#!/bin/bash
#SBATCH --job-name=Traffic
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=0-48:00:00
#SBATCH --output=logsreal/Traffic_%A_%a.out
#SBATCH --error=logsreal/Traffic_%A_%a.err   

module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

# Define configs as lists (same order)
ABSTRACTIONS=("contextual" "False" "structural-knn" "action-set-knn"
        "full_k_means" "full_k_means"
        "partial_k_means" "partial_k_means" "branchi")
KS=(
    0 0 0 0 
    1000 500
    1000 500 0)  

DATASET="traffic_fines_1"

# Fetch the current parameter values
ABSTRACTION=${ABSTRACTIONS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}


exec > logsreal/Traffic_${ABSTRACTION}_k${K}_%A_%a.out 2> logsreal/Traffic_${ABSTRACTION}_k${K}_%A_%a.err
cd ../


# Run the Python script with all arguments
python training.py \
  --dataset $DATASET \
  --state_abstraction $ABSTRACTION \
  --k $K \
  --action_masking True \
  --binary_outcome False \
  --num_episodes 100000 \
  --checkpoint_interval 100000
  

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s