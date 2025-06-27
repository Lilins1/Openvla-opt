#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-791
#SBATCH -J libero_train
#SBATCH --output=logs/libero_out_%j.txt
#SBATCH --error=logs/libero_err_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=100:00:00
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

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/Bezier/libero/lora_train \
  --load_Lora_path /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/action/libero/lora_train/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r128+dropout-0.0--image_aug--action8chunk_libero_acc16--80000_chkpt \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --grad_accumulation_steps 2\
  --learning_rate 5e-4 \
  --num_steps_before_decay 30005 \
  --max_steps 80005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 128 \
  --wandb_project vla-libero-Bezierfit \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --shuffle_buffer_size 100000 \
  --use_model use_bezier_regression_continuous\
  --finetune_lora True\
  --save_vla True\
  --rnn_in_batch True\
  --run_id_note action8chunk_libero_acc16