#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-791
#SBATCH -J libero_train
#SBATCH --output=logs/libero_out_%j.txt
#SBATCH --error=logs/libero_err_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mail-user=lruizhe@kth.se
#SBATCH --mail-type=ALL
#SBATCH -p alvis

# -------------------------------
# Setup environment
# -------------------------------
module purge
module load GCCcore/12.2.0  # load basic GCCcore so that 'module spider' works
#module spider Python   # check what Python versions are available

# Load a valid Python module (check this by running `module avail Python`)
module load Python/3.10.8-GCCcore-12.2.0  # ested version on Alvis

# If your venv does not exist yet, create it (only once)
# python -m venv $HOME/univla_venv

# Activate venv
source /mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/vlaoft_venv/bin/activate
export WANDB_API_KEY=922db5ed979e11a6bd8cba12e8d3487927152192

# Add torchrun to PATH manually if needed
export PATH="/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/vlaoft_venv/bin:$PATH"
export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/Openvla-opt:$PYTHONPATH"
# -------------------------------
# Debug info
# -------------------------------
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# -------------------------------
# Run your training script
# -------------------------------
python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/lebiro/libero_spatial_no_noops/lora_train/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts--continuous_acts--rnn--finetune_RNN--100000_chkpt --task_suite_name libero_spatial