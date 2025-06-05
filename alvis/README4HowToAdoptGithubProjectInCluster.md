# THIS IS A TURTORIAL TO RUN UniVlA in alvis cluster
UniVLA github: https://github.com/OpenDriveLab/UniVLA

## Introduction slides for Alvis
https://www.c3se.chalmers.se/documentation/first_time_users/intro-alvis/slides/#containers

## instruction to Alvis 
Software modules and environments: https://www.c3se.chalmers.se/documentation/module_system/
├── The module system 
├── Python environments
├── Python module example
# commonly use command for alvis 
# 1.0  Alvis uses the Lmod module system. First, list available modules: 
module avail
# 1.1 list module 
module list 

## download dataset from huggingface, often you need to install git lfs install, how to do that 
# Try checking if Git LFS is available as a module:
module avail git
# If it's available, load it:
module load git-lfs
# Then try:
git lfs install

## how to use Alvis cluster
## !!! improtance
Avoid using pip install without first creating a virtual environemnt. It will place user wide packages directly into your ~/.local/ directory. This will use up your disk quota, and will leak into all other containers and environments, likely breaking compatibility. Please only follow the examples below.

# Access Alvis webpage via https://www.c3se.chalmers.se/about/Alvis/
# then, login alvis via https://alvis.c3se.chalmers.se, you will see the following picture

https://kth-my.sharepoint.com/:i:/g/personal/sicliu_ug_kth_se/EfE8n_fVRU5Mj2ZVpwKJkY4BwmaC2ofI_D4LL2hncnvDiQ?e=hsbZnI

# Struture of my Alvis project named 'naiss2024-5-164'
├── Files             # file directory
|   └── Home Directory        # home directory
|   └── /mimer/NOBACKUP/groups/naiss2024-5-164/       # directory to save all of your dataset and codes
|   └── /mimer/NOBACKUP/Datasets       # public datasets
|   └── Check my quota       # you can check your quota
├── Jobs
|   └── Active Jobs       # you submitted jobs to alvis
|   └── Check my project usage
├── Clusters
|   └── Alvis Shell Access       # open terminal (similar terminal in ubuntu)
├── Interactive Apps
|
├── My interactive Sessions

# open termimal via ├── Clusters/Alvis Shell Access, then you will see 
'''
[sichaol@alvis1 ~]$
'''

## :video_game: Getting Started <a name="installation"></a> NAISS 

1. We use virtual environment to manage the environment.

```bash
#To use virtualenv, we need load its module, to see https://www.c3se.chalmers.se/documentation/module_system/modules/#finding-compatible-software
# Reset everything
module purge
# Alvis uses the Lmod module system. First, list available modules: 
module avail
# Load matching toolchain and Python version, given conda environment is conda create -n univla python=3.10 -y
module load GCCcore/12.2.0
module load Python/3.10.8-GCCcore-12.2.0

# create a virtual environment under Python 3.10.8:

python -m venv univla_venv
source univla_venv/bin/activate

# # create a virtual environment, which is similar with conda environment (another way to creat a virtual environment)
# virtualenv --system-site-packages univla_venv 

# # to activate env, you first
# module load virtualenv/20.23.1-GCCcore-12.3.0
# #source your environment
# source univla_venv/bin/activate
# ```

2. Install dependencies.

```bash
# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
# Our experiments are conducted with 'torch 2.2.0 + cuda 12.1'
# pip install torch torchvision
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

pip install xformers==0.0.24 --no-build-isolation # compatable with torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# cd your local folder in NAISS,
cd /mimer/NOBACKUP/groups/naiss2024-5-164/Sichao

# Clone our repo and pip install to download dependencies
git clone https://github.com/OpenDriveLab/UniVLA.git
cd univla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
##  if error is miss CUDA environment, see th below
  # to install flash-attn==2.5.5, you need to have a cuda environment, how to that
  # On Alvis, CUDA is not preloaded in your environment. Use module to load CUDA and other dependencies.
  module load CUDA/12.1.1
  # Verify nvcc and CUDA paths
  which nvcc
  # It should return something like: /sw/SOME_PATH/CUDA/12.1.1/bin/nvcc
  # Then confirm CUDA_HOME is correct:
  export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
  echo $CUDA_HOME
  # Example output: /sw/SOME_PATH/CUDA/12.1.1
  # Re-activate your virtual environment (optional but safe)
  source univla_venv/bin/activate # please make sure your univla_venv is located in the same folder, if it is in the previous directory, 
  source ../univla_venv/bin/activate
  # Install FlashAttention again
  pip install "flash-attn==2.5.5" --no-build-isolation

# when you successfully buid the environment, you can check 
# '(univla_venv) [sichaol@alvis1 Sichao]$ module list', you will see the below 
# Currently Loaded Modules:
#  1) GCCcore/12.3.0                 4) bzip2/1.0.8-GCCcore-12.3.0       7) Tcl/8.6.13-GCCcore-12.3.0     10) libffi/3.4.4-GCCcore-12.3.0   13) virtualenv/20.23.1-GCCcore-12.3.0
#  2) zlib/1.2.13-GCCcore-12.3.0     5) ncurses/6.4-GCCcore-12.3.0       8) SQLite/3.42.0-GCCcore-12.3.0  11) OpenSSL/1.1
# 3) binutils/2.40-GCCcore-12.3.0   6) libreadline/8.2-GCCcore-12.3.0   9) XZ/5.4.2-GCCcore-12.3.0       12) Python/3.11.3-GCCcore-12.3.0


#### 1) LIBERO
> Please first download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/main) that we used in experiments
downloadLIBERO datasets under '/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla/dataset'
Start training with ```torchrun```:
1) You should first set the pretrained UniVLA and latent action model path in ```vla_path``` and ```lam_path``` of the [training config](https://github.com/OpenDriveLab/UniVLA/blob/b502b3eddc05fef9984d34932a41c96e5a9f21a3/vla-scripts/finetune_libero.py#L107).

vla_path point to the directory of univla-7b, download via https://huggingface.co/qwbu/univla-7b save under '/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla/univla-7b'
lam_path point to the directory of univla-latent-action-model, download via https://huggingface.co/qwbu/univla-latent-action-model save under '/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla'
download TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b (https://huggingface.co/TRI-ML/prismatic-vlms/tree/main) under '/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla'

vla_path: str = "/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla/univla-7b/univla-7b"            # Path to your local UniVLA path
lam_path: str = "/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla/univla-latent-action-model/lam-stage-2.ckpt" 

2) Set your local LIBERO dataset path in [```data_root_dir```](https://github.com/OpenDriveLab/UniVLA/blob/b502b3eddc05fef9984d34932a41c96e5a9f21a3/vla-scripts/finetune_libero.py#L110).
# Directory Paths
data_root_dir: Path = "/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla/dataset/modified_libero_rlds"      # Path to Open-X dataset directory
dataset_name: str = "libero_spatial_no_noops" 

3) You can choose ```dataset_name``` from ```libero_spatial_no_noops```, ```libero_object_no_noops```, ```libero_goal_no_noops```, and ```libero_10_no_noops```
> We trained on *'Spatial'*, *'Object'* and *'Goal'* for 30k steps and *'Long'* for 40k steps. Please first modify the ```max_steps``` in training config accordingly for reproduction. Here,I used ```libero_10_no_noops```

## Configure WANDB
you are in the wandb team 'chu2002-kth-royal-institute-of-technology-org', replace name of wandb_project and  wandb_entity in finetune_libero.py (Lines 145 and 146)
you can cutomise the name of 'wandb_project', and 'wandb_entity' is the team name, namely 'chu2002-kth-royal-institute-of-technology-org'

## .bash file used to submit to Alvis cluster
# on terminal of Alvis cluster, you can see it starts with '(univla_venv) [sichaol@alvis1 univla]$ ', where '(univla_venv)' is created virtual enviroment, #sichaol@alvis1' is my account name. 'univla' is the current directory

# How to submit .bash to Alvis cluster
sbatch submit.bash # make sure submit.bash is in the directory of univla, or your current directory 

# check if submission is successful. if returns 
Submitted batch job 4586952 # similar one, it means success

# then, you can logs info to see it has any error. For my own project, the log folder is '/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla/logs'. log info saved in 'libero_err_4586952.txt'

## explain submit.bach file. please remove all of explanination after the second # for each command
```bash
#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-791                             # Project allocation/account under which the job is charged.
#SBATCH -J libero_train                                 # Sets the job name 
#SBATCH --output=logs/libero_out_%j.txt                 # Specifies the standard output log file path. %j gets replaced by the job ID 
#SBATCH --error=logs/libero_err_%j.txt                  # Specifies the standard error log file path (for warnings/errors). 
#SBATCH --nodes=1                                       # Requests 1 compute node.
#SBATCH --gpus-per-node=A100:4                          # Requests 4 A100 GPUs on the node
#SBATCH --ntasks-per-node=4                             # Spawns 4 parallel tasks on the node (usually one per GPU).
#SBATCH --cpus-per-task=6                               # Allocates 6 CPU cores per task, often used for CPU-intensive preprocessing, dataloading, etc. 
#SBATCH --time=24:00:00                                 # Maximum job duration: 24 hours.
#SBATCH --mail-user=sicliu@kth.se                       # Email address to send job notifications. 
#SBATCH --mail-type=ALL                                 # Sends email on job BEGIN, END, FAIL, etc.
#SBATCH -p alvis                                        #Runs the job on the Alvis partition, i.e., the queue/group of nodes you're submitting to.                                  #
                            

# -------------------------------
# Setup environment
# -------------------------------
module purge
module load GCCcore/12.2.0  # load basic GCCcore so that 'module spider' works
#module spider Python   # check what Python versions are available

# Load a valid Python module (check this by running `module avail Python`)
module load Python/3.10.8-GCCcore-12.2.0  # ested version on Alvis

# If your venv does not exist yet, create it (only once). Here, you have created a virtual enviorment with all of the dependencies
# python -m venv $HOME/univla_venv

# Activate venv
source /mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla_venv/bin/activate   #univla_venv is the directory of your virtual enviorment
export WANDB_API_KEY=227e90122b9b1ca8c8428a045fea14583f8c331d                   # WANDB_API_KEY

# Add torchrun to PATH manually if needed
export PATH="/mimer/NOBACKUP/groups/naiss2024-5-164/Sichao/univla_venv/bin:$PATH"

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
torchrun --standalone --nnodes=1 --nproc-per-node=4 vla-scripts/finetune_libero.py \
  --dataset_name "libero_10_no_noops" \
  --run_root_dir "libero_log"

```
## this corresponse to the content of submit.bash
```bash
# Start training on LIBERO-10(long) with 4 GPUs
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_libero.py --dataset_name "libero_10_no_noops" --run_root_dir "libero_log" 
```

Once you finished training and get the action decoder and VLA backbone, you can simply start evaluation with:


## 提交作业
sbatch submit.bash
squeue -u $USER
scontrol show job <作业ID>
# 实时监控输出
tail -f logs/	libero_out_4589348.txt

# 查看完整输出
cat 4589365.out