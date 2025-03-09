#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00

source_env() {
  ENV_PATH="/proj/rep-learning-robotics/users/x_andri/lerobot/.env"
  echo "Checking for .env at: $ENV_PATH"
  
  if [ -f "$ENV_PATH" ]; then
    set -a
    . "$ENV_PATH"
    set +a
  else
    echo "Error: .env file not found at $ENV_PATH!"
    exit 1
  fi
}
source_env

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate lerobot
python train_privileged_MLP.py \
    --output-dir ../../outputs/train/diffusion_pusht_keypoints_state_expanded \
    --push-to-hub \
    --hub-repo-id "the-future-dev/diffusion_pusht_keypoints_state_expanded" \
    --seed 42
