#!/bin/bash
#SBATCH --job-name=SimBank
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH --time=0-20:30:00
#SBATCH --output=logsreal/SimBank_%A_%a.out
#SBATCH --error=logsreal/SimBank_%A_%a.err   

module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env


# Define configs as lists (same order)
ABSTRACTIONS=("partial_k_means" "partial_k_means" "partial_k_means" "partial_k_means" "partial_k_means" 
              "partial_k_means" "partial_k_means" "partial_k_means" "partial_k_means" "partial_k_means")
KS=(
    100 200 300 400 500
    600 700
    800 900 1000) 

DATASET="SimBank"

# Fetch the current parameter values
ABSTRACTION=${ABSTRACTIONS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}


exec > logsreal/SimBank_${ABSTRACTION}_k${K}_%A_%a.out 2> logsreal/SimBank_${ABSTRACTION}_k${K}_%A_%a.err
cd ../

# Run the Python script with all arguments
python training.py \
  --dataset $DATASET \
  --state_abstraction $ABSTRACTION \
  --k $K \
  --action_masking True \
  --binary_outcome False \
  --num_episodes  100000 \
  --checkpoint_interval 100000

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s