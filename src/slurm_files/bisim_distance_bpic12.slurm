#!/bin/bash
#SBATCH --job-name=b12d
#SBATCH --array=0-8   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --output=logsbisim/bpic12_%A_%a.out
#SBATCH --error=logsbisim/bpic12_%A_%a.err   
#SBATCH --time=0-20:30:00

module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

ABSTRACTIONS=("contextual" "False" "structural-knn" "action-set-knn"
        "full_k_means" "full_k_means"
        "partial_k_means" "partial_k_means" "branchi")
KS=(
    0 0 0 0 
    1000 500
    1000 500 0)      # 0 if k not used (e.g., for False)
DATASET="bpic2012_accepted"

# Fetch the current parameter values
ABSTRACTION=${ABSTRACTIONS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}

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
