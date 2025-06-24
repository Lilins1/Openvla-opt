"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from torch.nn.utils.rnn import pad_sequence

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead

from prismatic.models.MLP_RNN_action import MLP_RNN_ActionHead

from prismatic.models.Bezier_MLP_Action import Bezier_MLP_Action
from prismatic.models.Bezier_MLP_Action_b import Bezier_MLP_Action_b
from prismatic.models.Bezier_MLP_Action_continuous import Bezier_MLP_Action_continuous
from prismatic.models.MLP_DCT_Actionhead import MLP_DCT_Actionhead
from prismatic.models.MLP_Action_Actionhead import MLP_Action_Actionhead

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    ACTION_CHUNK_PER_CURVE,
    TOKEN_SEQUENCE_LINE,
    BEZIER_CURVES,
    ACTION_LENGTH,
    Debug
)
from prismatic.vla.datasets.datasetsSequence import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.vla.datasets.DataProcess import BezierProcess

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

forCount = 0
forSum = 0
print_curve_count = 0

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = False                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    use_rnn_regression: bool = True
    use_bezier_regression: bool = False
    use_bezier_regression_onecurve:bool = False
    use_bezier_regression_continuous:bool = False
    use_dct_regression:bool = False
    use_action_regression:bool = False


    use_model: str = 'use_bezier_regression'

    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    saved_proprio_path: str = None
    rnn_type = 'rnn'                                 #  'rnn', 'lstm', 'gru', 'relu'
    rnn_in_batch: bool = False
    saved_action_head_path: str = None
    

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    load_Lora_path: str = None
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    finetune_lora: bool = True                       # add lora fine tune
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!
    save_vla: bool = True                           # if Save Vla model

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on



 

def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if module_name == "mlp_rnn_action_head":
        if cfg.saved_action_head_path:
            module = load_module_from_path(
                saved_path=cfg.saved_action_head_path,
                module_name=module_name,
                module=module,
                device_id=device_id
            )
            
            print(f"✅ Loaded pretrained {module_name} from {cfg.saved_action_head_path}")
        else:
            print(f"⚠️ No saved {module_name} model path provided, initializing new action head ")


    if module_name == "proprio_projector":
        if cfg.saved_proprio_path:
            module = load_module_from_path(
                saved_path=cfg.saved_proprio_path,
                module_name=module_name,
                module=module,
                device_id=device_id
            )
            
            print(f"✅ Loaded pretrained {module_name} from {cfg.saved_action_head_path}")
        else:
            print(f"⚠️ No saved {module_name} model path provided, initializing new action head ")


    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

def load_module_from_path(
    saved_path: str,
    module_name: str,
    module: nn.Module,
    device_id: int,
) -> nn.Module:
    """
    从一个单独的路径加载已训练好的子模块权重。整个流程都在 CPU 上做完：
      1. torch.load(map_location="cpu")
      2. 根据 module.state_dict().keys() 与 checkpoint keys，做前缀 'module.' 的对齐
      3. module.load_state_dict(对齐后的 state_dict)
      4. 删除临时 state_dict

    note: 任何对齐与 load 都在 CPU 上完成，不会触发 GPU 上的冗余拷贝。

    返回：仍然在 CPU 上的 module（只加载了权重，尚未搬到 GPU）。
    """
    # 1) load 到 CPU
    state_dict = torch.load(saved_path, map_location="cpu")

    # 2) 前缀对齐
    model_keys = list(module.state_dict().keys())
    ckpt_keys  = list(state_dict.keys())

    def _all_with_prefix(keys, prefix="module."):
        return all(k.startswith(prefix) for k in keys)

    def _all_without_prefix(keys, prefix="module."):
        return all(not k.startswith(prefix) for k in keys)

    # 如果 module 的 key 不带 "module."，但 ckpt 带了 -> 去掉前缀
    if _all_without_prefix(model_keys) and _all_with_prefix(ckpt_keys):
        new_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_sd[new_key] = v
        state_dict = new_sd

    # 如果 module 的 key 带 "module."，但 ckpt 不带 -> 加上前缀
    elif _all_with_prefix(model_keys) and _all_without_prefix(ckpt_keys):
        new_sd = {}
        for k, v in state_dict.items():
            new_key = "module." + k
            new_sd[new_key] = v
        state_dict = new_sd

    # 3) 在 CPU 上 load 到 module
    module.load_state_dict(state_dict, strict=True)

    # 4) 释放临时变量
    del state_dict

    return module

def pad_or_truncate(actions, target_len=120):
    """
    actions: Tensor shape [B, T, ...]
    返回: Tensor shape [B, target_len, ...]
    """
    batch_size, seq_len = actions.shape[:2]
    if seq_len == target_len:
        return actions
    elif seq_len > target_len:
        return actions[:, :target_len, ...]
    else:
        # pad length
        pad_len = target_len - seq_len
        # padding dims: pad last dim of sequence dimension with zeros
        # F.pad pads last dims in reverse order, so pad=(pad_left, pad_right, ...)
        # For padding sequence dimension at dim=1, pad = (0, 0, 0, pad_len)
        # But F.pad only supports padding last dims, so要用torch.cat来pad sequence维度
        pad_shape = list(actions.shape)
        pad_shape[1] = pad_len
        padding = torch.zeros(pad_shape, dtype=actions.dtype, device=actions.device)
        return torch.cat([actions, padding], dim=1)

def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_model,
    use_proprio,
    use_film,
    num_patches,
    rnn_prev_state=None,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    epsilon = 0.05,
):
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}
    curve_length = 1
    global print_curve_count

    # Get ground-truth action labels
    actions = batch["actions"]  # 假设 shape=[B, T, ...]
    if use_model == 'use_bezier_regression':
        actions = pad_or_truncate(actions, (NUM_ACTIONS_CHUNK//ACTION_CHUNK_PER_CURVE) * TOKEN_SEQUENCE_LINE)
    else:
        actions = pad_or_truncate(actions, ACTION_LENGTH)
    ground_truth_actions = actions.to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_model == 'use_diffusion':
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # print("in labels len: " + str(batch["labels"]))
    # print("in input_ids len: " + str(batch["input_ids"]))
    # print("in attention_mask len: " + str(batch["attention_mask"]))

    def trim_batch_after_eos(labels_batch, input_ids_batch, attention_mask_batch,keep_length):
        new_labels = []
        new_input_ids = []
        new_attention_masks = []
        seq_len = labels_batch.size(1)
        PAD_TOKEN_ID =2

        for labels, input_ids, attn in zip(labels_batch, input_ids_batch, attention_mask_batch):
            L = labels.tolist()
            I = input_ids.tolist()
            M = attn.tolist()

            trimmed_L, trimmed_I, trimmed_M = [], [], []
            started = False
            keep = keep_length

            for tok, iid, mask in zip(L, I, M):
                if not started:
                    if tok == -100:
                        trimmed_L.append(tok)
                        trimmed_I.append(iid)
                        trimmed_M.append(mask)
                    else:
                        started = True
                if started and keep > 0:
                    keep -= 1
                    trimmed_L.append(tok)
                    trimmed_I.append(iid)
                    trimmed_M.append(mask)
                    if tok == 2:  # eos
                        break

            new_labels.append(torch.tensor(trimmed_L, dtype=labels.dtype, device=labels.device))
            new_input_ids.append(torch.tensor(trimmed_I, dtype=input_ids.dtype, device=input_ids.device))
            new_attention_masks.append(torch.tensor(trimmed_M, dtype=attn.dtype, device=attn.device))

        # 堆叠成 (B, seq_len) 的 tensor
        # new_labels = torch.stack(new_labels, dim=0)
        # new_input_ids = torch.stack(new_input_ids, dim=0)
        # new_attention_masks = torch.stack(new_attention_masks, dim=0)
        # 填充值：labels用-100，input_ids用PAD_TOKEN_ID，attention_mask用0
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)
        padded_input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
        padded_attention_masks = pad_sequence(new_attention_masks, batch_first=True, padding_value=False)
        return padded_labels, padded_input_ids, padded_attention_masks


    if use_model == 'use_bezier_regression':
        token_padding_length = NUM_ACTIONS_CHUNK
        batch["labels"], batch["input_ids"], batch["attention_mask"] = trim_batch_after_eos(batch["labels"], batch["input_ids"], batch["attention_mask"],token_padding_length)
        Debug("in labels: " + str(batch["labels"]))
        Debug("in input_ids: " + str(batch["input_ids"]))
        Debug("in attention_mask: " + str(batch["attention_mask"]))
    elif use_model == 'use_bezier_regression_onecurve':
        token_padding_length = ACTION_DIM*(BEZIER_CURVES * ACTION_CHUNK_PER_CURVE + 1)
        batch["labels"], batch["input_ids"], batch["attention_mask"] = trim_batch_after_eos(batch["labels"], batch["input_ids"], batch["attention_mask"],token_padding_length)
        Debug("in labels: " + str(batch["labels"]))
        Debug("in input_ids: " + str(batch["input_ids"]))
        Debug("in attention_mask: " + str(batch["attention_mask"]))
    elif use_model == 'use_bezier_regression_continuous' or use_model == 'use_dct_regression' or use_model == 'use_action_regression':
        # token_padding_length = NUM_ACTIONS_CHUNK + 1
        token_padding_length = 2 + 1
        batch["labels"], batch["input_ids"], batch["attention_mask"] = trim_batch_after_eos(batch["labels"], batch["input_ids"], batch["attention_mask"],token_padding_length)


    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if (use_model == 'use_diffusion') else None,
            noisy_action_projector=noisy_action_projector if (use_model == 'use_diffusion') else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if (use_model == 'use_diffusion') else None,
            use_film=use_film,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    Debug("current_action_mask: "+str(current_action_mask))
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    Debug("next_actions_mask: "+str(next_actions_mask))

    # Compute metrics for discrete action representation (next-token prediction)
    if use_model is None:
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]

        if use_model == 'use_bezier_regression_onecurve':

            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, token_padding_length, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            # Predict action
            out_put_curves = action_head.module.predict_action(actions_hidden_states)
            Debug("out_put_curves: "+str(out_put_curves))
            
            # 再传给 compute_loss
            out_put_curves_loss, batch_errors_org = BezierProcess.fitBezierToolBox.compute_loss(out_put_curves, ground_truth_actions,ACTION_DIM)
            Debug("out_put_curves_loss:", out_put_curves_loss)
            avg_length = BezierProcess.fitBezierToolBox.curves_length(out_put_curves)
            Debug("avg_length:", avg_length)
            ratio = (2 ** (0.5 *(1 - 2 * (avg_length/TOKEN_SEQUENCE_LINE))))
            Debug("ratio:", ratio)
            loss = out_put_curves_loss

        if use_model == 'use_bezier_regression_continuous':

            ground_truth_batch = ground_truth_actions

            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, token_padding_length, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            # Predict action
            out_put_curves = action_head.module.predict_action(actions_hidden_states)
            print_curve_count += 1
            # if print_curve_count % 10001 == 0: 
            #     print("out_put_curves: "+str(out_put_curves))
            #     print("ground_truth_actions: "+str(ground_truth_actions))
            #     print_curve_count = 0

            # 再传给 compute_loss
            out_put_curves_loss, ratio = BezierProcess.fitBezierToolBox.compute_loss(out_put_curves, ground_truth_actions,ACTION_DIM)
            BezierProcess.fitBezierToolBox.avg_update(out_put_curves_loss)

            Debug("out_put_curves_loss:", out_put_curves_loss)
            avg_length = BezierProcess.fitBezierToolBox.curves_length_avg(out_put_curves)
            Debug("avg_length:", avg_length)
            # ratio = (2 ** (0.5 *(1 - 2 * (avg_length/TOKEN_SEQUENCE_LINE))))
            # ratio = 1
            Debug("ratio:", ratio)
            loss = out_put_curves_loss

        if use_model == 'use_dct_regression':
            # Predict action
            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, token_padding_length, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            # print(f'ground_truth_actions: {ground_truth_actions}')
            # print(f'predicted_actions: {predicted_actions}')
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_model == 'use_action_regression':
            # Predict action
            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, token_padding_length, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            # print(f'ground_truth_actions: {ground_truth_actions}')
            # print(f'predicted_actions: {predicted_actions}')
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        
        if use_model == 'use_bezier_regression':

            ground_truth_batch = ground_truth_actions
            # print("ground_truth_batch: "+str(ground_truth_batch))

            seq_len = NUM_ACTIONS_CHUNK // ACTION_CHUNK_PER_CURVE
            max_pts = TOKEN_SEQUENCE_LINE * seq_len

            # for ground_truth in ground_truth_batch:
            #     curve_fit = BezierProcess.fitBezierToolBox.fit_beziers(ground_truth,epsilon)
            #     if len(curve_fit) > NUM_ACTIONS_CHUNK//ACTION_CHUNK_PER_CURVE:
            #         curve_fit = curve_fit[:NUM_ACTIONS_CHUNK//ACTION_CHUNK_PER_CURVE]
            #     ground_truth_curve.append(curve_fit)
            #     curve_length = [curve[3] for curve in curve_fit]
            #     ground_truth_length.append(curve_length)
            #     ground_truth_curve = torch.stack(ground_truth_curve, dim=0)
            #     print("ground_truth_curve: "+str(ground_truth_curve))
            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            # Predict action
            out_put_curves = action_head.module.predict_action(actions_hidden_states)
            Debug("out_put_curves: "+str(out_put_curves))
            
            # 生成 list-of-list
            ground_truth_curves = [
                BezierProcess.fitBezierToolBox
                    .fit_beziers(gt[:max_pts], epsilon)[:seq_len]
                for gt in ground_truth_batch
            ]

            # 整体转换成 (B, seq_len, 4, pt_dim) 张量
            batch_curves = BezierProcess.fitBezierToolBox.make_ground_truth_tensors(
                ground_truth_curves,
                device=actions_hidden_states.device,
                pt_dim=ACTION_DIM,
                seq_len=seq_len
            )
            batch_curves = BezierProcess.fitBezierToolBox.curves_to_combined(batch_curves)
            Debug("batch_curves: "+str(batch_curves))


            # 再传给 compute_loss
            out_put_curves_loss,batch_errors_org = BezierProcess.fitBezierToolBox.compute_loss(out_put_curves, ground_truth_actions,ACTION_DIM)
            Debug("out_put_curves_loss:", out_put_curves_loss)
            loss = out_put_curves_loss


            ratio = 0.2

            # 然后再切片
            predict_curve_length = out_put_curves[:, :,-1]
            ground_truth_length = batch_curves[:, :,-1]
            predict_curve_length_loss = torch.nn.L1Loss()(predict_curve_length, ground_truth_length)/TOKEN_SEQUENCE_LINE * ratio # 归一化
            loss += predict_curve_length_loss
            Debug("predict_curve_length_loss:" + str(predict_curve_length_loss))

        if use_model == 'use_l1_regression':
            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_model == 'use_rnn_regression':
            # Predict action
            input_dim = text_hidden_states.shape[-1]  # 这个是 D
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]
                .reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, input_dim)
                .reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM * input_dim)
                .to(torch.bfloat16)
            )
            predicted_actions,rnn_prev_state = action_head.module.predict_action(actions_hidden_states,rnn_prev_state)
            # Get full L1 loss
            # print(f'ground_truth_actions: {ground_truth_actions}')
            # print(f'predicted_actions: {predicted_actions}')
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)


            # print(f"ground_truth_actions: {ground_truth_actions}")
            # print(f"predicted_actions: {predicted_actions}")
            

        if use_model == 'use_diffusion':
            actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_model == 'use_diffusion' or (use_model == 'use_diffusion' and compute_diffusion_l1)
        if should_log_l1_loss and not (use_model == "use_bezier_regression" or use_model == "use_bezier_regression_onecurve" or use_model == "use_bezier_regression_continuous"):
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        else:
            if use_model == "use_bezier_regression":
                metrics.update(
                    {
                        "out_put_curves_loss": out_put_curves_loss.item(),
                        "predict_curve_length_loss": predict_curve_length_loss.item()
                    }
                )
            
            if use_model == "use_bezier_regression_onecurve" or use_model == "use_bezier_regression_continuous":
                metrics.update(
                    {
                        # "out_put_curves_loss": batch_errors_org.item(),
                        "avg_length": avg_length,
                        "ratio": ratio.item(),
                    }
                )
            # if use_model == 'use_action_regression' or use_model == 'use_dct_regression':
            #     metrics.update(
            #         {
            #             "curr_action_l1_loss": curr_action_l1_loss.item(),
            #             "next_actions_l1_loss": next_actions_l1_loss.item(),
            #         }
            #     )


    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics, rnn_prev_state


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if action_head is not None: #(cfg.use_l1_regression or cfg.use_diffusion or cfg.use_rnn_regression or cfg.use_dct_regression or cfg.use_action_regression or cfg) and
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training and cfg.save_vla:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg: FinetuneConfig,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0
    rnn_prev_state = None

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            if cfg.rnn_in_batch or (cfg.use_rnn_regression and lang_flag != batch["language_instructions"]):# 每batch内训练
                current_instructs = lang_flag
                lang_flag = batch["language_instructions"]
                print(f'language_instructions: {batch["language_instructions"]}')
                rnn_prev_state = None
            # Always compute L1 loss for validation, even for diffusion
            _, metrics,rnn_prev_state = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_model=cfg.use_model,
                num_patches=num_patches,
                rnn_prev_state = rnn_prev_state,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )
            # Detach the RNN state to prevent graph retention across batches
            if cfg.use_rnn_regression:
                if isinstance(rnn_prev_state, tuple):  # For LSTM
                    rnn_prev_state = tuple(s.detach() for s in rnn_prev_state)
                else:  # For RNN or GRU
                    rnn_prev_state = rnn_prev_state.detach()


            # Add the loss value to the metrics
            metrics["Val loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    cfg.use_l1_regression = (cfg.use_model == 'use_l1_regression') 
    cfg.use_diffusion = (cfg.use_model == 'use_diffusion') 
    cfg.use_bezier_regression = (cfg.use_model == 'use_bezier_regression') 
    cfg.use_bezier_regression_onecurve = (cfg.use_model == 'use_bezier_regression_onecurve') 
    cfg.use_bezier_regression_continuous = (cfg.use_model == 'use_bezier_regression_continuous') 
    cfg.use_rnn_regression = (cfg.use_model == 'use_rnn_regression')
    cfg.use_dct_regression = (cfg.use_model == 'use_dct_regression') 
    cfg.use_action_regression = (cfg.use_model == 'use_action_regression')  
    action_head = None

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        if cfg.load_Lora_path:
            # 恢复训练时加载保存的LoRA适配器
            adapter_dir = Path(cfg.load_Lora_path) / "lora_adapter"
            if not adapter_dir.exists():
                raise FileNotFoundError(f"LoRA adapter directory not found at {adapter_dir}")
            vla = PeftModel.from_pretrained(vla, str(adapter_dir))
            print(f"✅ Loaded LoRA adapter from {adapter_dir}")
            vla.print_trainable_parameters()

        else:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    if cfg.use_rnn_regression:
        action_head = init_module(
            MLP_RNN_ActionHead,
            "mlp_rnn_action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM, 'rnn_type': cfg.rnn_type},
            to_bf16=True,
        )

    if cfg.use_bezier_regression:
        action_head = init_module(
            Bezier_MLP_Action,
            "mlp_rnn_action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )
        
    if cfg.use_bezier_regression_onecurve:
        action_head = init_module(
            Bezier_MLP_Action_b,
            "mlp_rnn_action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )
    
    if cfg.use_bezier_regression_continuous:
        action_head = init_module(
            Bezier_MLP_Action_continuous,
            "mlp_rnn_action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    if cfg.use_dct_regression:
        action_head = init_module(
            MLP_DCT_Actionhead,
            "mlp_rnn_action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    if cfg.use_action_regression:
        action_head = init_module(
            MLP_Action_Actionhead,
            "mlp_rnn_action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = []
    if cfg.finetune_lora:
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        print(f"add vla.parameters: {sum(p.numel() for p in trainable_params)}")
    # if cfg.use_l1_regression or cfg.use_diffusion or cfg.use_rnn_regression:
        # trainable_params += [param for param in action_head.parameters() if param.requires_grad]# 训练action_head
        # print(f"add action_head: {sum(p.numel() for p in trainable_params)}")
    action_head_params = [param for param in action_head.parameters() if param.requires_grad]
    trainable_params += action_head_params
    print(f"add action_head: {sum(p.numel() for p in action_head_params)}")
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    # if cfg.use_proprio:
    #     trainable_params = [param for param in proprio_projector.parameters() if param.requires_grad] #只微调use_proprio
        print(f"add use_proprio: {sum(p.numel() for p in trainable_params)}")

    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        curve_fit_mode = cfg.use_bezier_regression,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0, 
        )

    ## Used for oder training
    # dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=cfg.batch_size,
    #     collate_fn=collator,
    #     num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    #     shuffle=False,  # 保持顺序
    # )
    # if cfg.use_val_set:
    #     print("cfg.use_val_set: true")
    #     val_batch_size = cfg.batch_size
    #     val_dataloader = DataLoader(
    #         val_dataset,
    #         batch_size=val_batch_size,
    #         collate_fn=collator,
    #         num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    #         shuffle=False,  # 保持顺序
    #     )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "out_put_curves_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "predict_curve_length_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "out_put_curves_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "avg_length": deque(maxlen=cfg.grad_accumulation_steps),
        "ratio": deque(maxlen=cfg.grad_accumulation_steps),

    }

    ForCount = 0
    normalized_lossAcc = 0
    rnn_prev_state = None

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        lang_flag = ""
        rnn_prev_state = None

        for batch_idx, batch in enumerate(dataloader):
            if cfg.rnn_in_batch or (cfg.use_rnn_regression and lang_flag != batch["language_instructions"]):# 每batch内训练
                current_instructs = lang_flag
                lang_flag = batch["language_instructions"]

                rnn_prev_state = None
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics,rnn_prev_state = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                use_model = cfg.use_model,
                num_patches=NUM_PATCHES,
                rnn_prev_state=rnn_prev_state,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )
            # Detach the RNN state to prevent graph retention across batches
            if cfg.use_rnn_regression:
                if isinstance(rnn_prev_state, tuple):  # For LSTM
                    rnn_prev_state = tuple(s.detach() for s in rnn_prev_state)
                else:  # For RNN or GRU
                    rnn_prev_state = rnn_prev_state.detach()

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            ForCount += 1
            normalized_lossAcc += abs(loss)
            if ForCount % 2000 == 0:
                print(f"normalized_lossAvg: {normalized_lossAcc/ForCount}")
                normalized_lossAcc = 0
                ForCount = 0

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            # for metric_name, value in metrics.items():
            #     if metric_name in recent_metrics:
            #         recent_metrics[metric_name].append(value)
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion or cfg.use_rnn_regression) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
