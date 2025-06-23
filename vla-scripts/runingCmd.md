# OpenVLA-oft

## Origin
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name aloha1_put_X_into_pot_300_demos \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film

## libero Bezier continuous
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/Model/Libero/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/BezierAndDCT/Bezier/log/libero_spatial_no_noops/lora_train \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio False \
  --batch_size 1 \
  --grad_accumulation_steps 16\
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 200005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 80 \
  --wandb_project vla-libero-Bezierfit \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --shuffle_buffer_size 100000 \
  --use_model use_bezier_regression_continuous\
  --finetune_lora True\
  --save_vla False\
  --rnn_in_batch True\
  --run_id_note Bezier1Curve_libero_acc16

## libero Bezier
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/Model/Libero/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/BezierAndDCT/Bezier/log/libero_spatial_no_noops/lora_train \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio False \
  --batch_size 1 \
  --grad_accumulation_steps 16\
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 200005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 96 \
  --wandb_project vla-libero-Bezierfit \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --shuffle_buffer_size 100000 \
  --use_model use_bezier_regression\
  --finetune_lora True\
  --save_vla False\
  --rnn_in_batch True\
  --run_id_note Bezier4Curve_libero_acc16
  

## libero rnn
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/Model/Libero/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model/lebiro/libero_spatial_no_noops/lora_train \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 3e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 200005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 16 \
  --wandb_project vla-libero_spatial_no_noops \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --shuffle_buffer_size 150000 \
  --use_l1_regression False\
  --use_rnn_regression True\
  --save_vla False\
  --rnn_in_batch True\
  --saved_proprio_path /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model/lebiro/mlp/libero_spatial_no_noops/openvla-7b+libero_spatial_no_noops+b1+lr-0.0002+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--rnn_regression--3rd_person_img--wrist_img--proprio_state--finetune_RNN--20000_chkpt/proprio_projector--20000_checkpoint.pt\
  --saved_action_head_path /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model/lebiro/libero_spatial_no_noops/openvla-7b+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt/action_head--150000_checkpoint.pt\
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--rnn_regression--3rd_person_img--wrist_img--proprio_state--finetune_RNN

## libero mlp
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/org_finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/Model/Libero/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model/lebiro/mlp/libero_spatial_no_noops \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 200005 \
  --save_freq 25000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project vla-libero_spatial_no_noops \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --shuffle_buffer_size 100000 \
  --use_l1_regression True\
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--org_finetune


## For 4090 need more memory
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData \
  --dataset_name bridge_orig \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/openvla-oft/log \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --wandb_project ur5-fintune-ruizhe \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts



# OpenVla For 4090

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData \
  --dataset_name bridge_orig \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/log \
  --adapter_tmp_dir /mnt/disk2/ruizhe/Projects/openvlaData/adapter_tmp_dir \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project ur5-fintune-ruizhe \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --save_steps 50 


# GPU Check
nvidia-smi

## test for rnn
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData \
  --dataset_name bridge_orig \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project ur5-fintune-ruizhe \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --save_freq 10000\
  --shuffle_buffer_size 1 \
  --use_film False\
  --rnn_in_batch True\

## test for rnn lebro
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/Model/Libero/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model/lebiro \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project ur5-fintune-ruizhe \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --save_freq 10000\
  --shuffle_buffer_size 1 \
  --use_film False\
  --rnn_in_batch True\



  ### if val
  --use_val_set True\
  --val_freq 2000\

  --saved_action_head_path /mnt/disk2/ruizhe/Projects/openvlaData/rnn_model/openvla-7b+bridge_orig+b1+lr-0.0002+lora-r32+dropout-0.0--image_aug--60000_chkpt/action_head--60000_checkpoint.pt

  ## use film will exceed 24GB

## test
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData \
  --dataset_name bridge_orig \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/log \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project ur5-fintune-ruizhe \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --save_freq 10000\
  --shuffle_buffer_size 1 \
  --use_film False

  ## use film will exceed 24GB

## get actions
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/GetEmbedded.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/disk2/ruizhe/Projects/openvlaData \
  --dataset_name bridge_orig \
  --run_root_dir /mnt/disk2/ruizhe/Projects/openvlaData/log \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project ur5-fintune-ruizhe \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --save_freq 50



## Origin OpenVLA:
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>


# tmux 远程窗口
tmux new -s mysesion
tmux attach -t mysesion
tmux ls
tmux kill-session -t mysession

# 提交作业


sbatch submit.bash

squeue -u $USER

scancel 4668051