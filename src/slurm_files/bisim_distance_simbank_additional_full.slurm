#!/bin/bash
#SBATCH --job-name=simd
#SBATCH --array=0-9   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --output=logsbisim/simbank_%A_%a.out
#SBATCH --error=logsbisim/simbank_%A_%a.err   
#SBATCH --time=0-20:30:00

module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

# Define configs
ABSTRACTIONS=("full_k_means" "full_k_means" "full_k_means" "full_k_means" "full_k_means" 
              "full_k_means" "full_k_means" "full_k_means" "full_k_means" "full_k_means")
KS=(
    100 200 300 400 500
    600 700
    800 900 1000) 

# Define datasets
DATASETS=("SimBank")

# Compute dataset index and config index
CONFIG_COUNT=${#ABSTRACTIONS[@]}   # 9
DATASET_INDEX=$(( SLURM_ARRAY_TASK_ID / CONFIG_COUNT ))
CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID % CONFIG_COUNT ))

DATASET=${DATASETS[$DATASET_INDEX]}
ABSTRACTION=${ABSTRACTIONS[$CONFIG_INDEX]}
K=${KS[$CONFIG_INDEX]}

# Redirect logs per dataset/config
exec > logsbisim/${DATASET}_${ABSTRACTION}_k${K}_%A_%a.out 2> logsbisim/${DATASET}_${ABSTRACTION}_k${K}_%A_%a.err

cd ../

# Run Python script
python bisimulation_distance.py \
  --dataset $DATASET \
  --state_abstraction $ABSTRACTION \
  --k $K \
  --binary_outcome False \
  --pair_sample_ratio 0.1

##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
