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



## libero rnn
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_RNN.py --vla_path openvla/openvla-7b --data_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/modified_libero_rlds --dataset_name libero_spatial_no_noops --run_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/lebiro/libero_spatial_no_noops/lora_train --use_film False --num_images_in_input 2 --use_proprio True --batch_size 8 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 200005 --save_freq 25000 --save_latest_checkpoint_only False --image_aug True --lora_rank 32 --wandb_project vla-libero_spatial_no_noops --wandb_entity chu2002-kth-royal-institute-of-technology --shuffle_buffer_size 100000 --use_l1_regression False --use_rnn_regression True --save_vla False --rnn_in_batch True --run_id_note parallel_dec--8_acts--continuous_acts--rnn--finetune_RNN

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/lebiro/libero_spatial_no_noops/lora_train \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
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
  --use_l1_regression False\
  --use_rnn_regression True\
  --save_vla False\
  --rnn_in_batch True\
  --run_id_note parallel_dec--8_acts--continuous_acts--rnn--finetune_RNN


## libero mlp
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_RNN.py --vla_path openvla/openvla-7b --data_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/modified_libero_rlds --dataset_name libero_spatial_no_noops --run_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/lebiro/libero_spatial_no_noops/lora_train --use_film False --num_images_in_input 2 --use_proprio True --batch_size 8 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 200005 --save_freq 25000 --save_latest_checkpoint_only False --image_aug True --lora_rank 32 --wandb_project vla-libero_spatial_no_noops --wandb_entity chu2002-kth-royal-institute-of-technology --shuffle_buffer_size 100000 --use_l1_regression False --use_rnn_regression True --save_vla False --rnn_in_batch True --run_id_note parallel_dec--8_acts--continuous_acts--mlp--finetune_RNN

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/lebiro/libero_spatial_no_noops/lora_train \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
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
  --use_rnn_regression False\
  --save_vla False\
  --rnn_in_batch True\
  --run_id_note parallel_dec--8_acts--continuous_acts--mlp--finetune_RNN


