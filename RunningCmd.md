## libero bezier
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_RNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/rnn_model/Bezier/libero/lora_train \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --grad_accumulation_steps 2\
  --learning_rate 5e-4 \
  --num_steps_before_decay 150005 \
  --max_steps 200005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 128 \
  --wandb_project vla-libero-Bezierfit \
  --wandb_entity chu2002-kth-royal-institute-of-technology \
  --shuffle_buffer_size 100000 \
  --use_model use_bezier_regression_continuous\
  --finetune_lora True\
  --save_vla False\
  --rnn_in_batch True\
  --run_id_note Bezier1Curve_libero_acc16 


  # cluster

cd /mimer/NOBACKUP/groups/naiss2024-5-164/Ruizhe/OPENVLA/Openvla-opt

sbatch submit.bash
squeue -u $USER

scancel 4678322