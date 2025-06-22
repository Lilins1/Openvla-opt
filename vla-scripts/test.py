import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

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
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
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
)
from prismatic.vla.datasets.datasetsSequence import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

forCount = 0
forSum = 0

@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000
    use_l1_regression: bool = False
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    saved_proprio_path: str = None
    use_rnn_regression: bool = True
    rnn_type = 'rnn'
    rnn_in_batch: bool = False
    saved_action_head_path: str = None
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 4  # Default to 4 steps for gradient accumulation
    max_steps: int = 200_000
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    diffusion_sample_freq: int = 50
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True
    save_vla: bool = True
    lora_warmup_steps: int = 5000
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10

def remove_ddp_in_checkpoint(state_dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_run_id(cfg) -> str:
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
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
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)

def count_parameters(module: nn.Module, name: str) -> None:
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
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if module_name == "mlp_rnn_action_head" and cfg.saved_action_head_path:
        module = load_module_from_path(cfg.saved_action_head_path, module_name, module, device_id)
        print(f"✅ Loaded pretrained {module_name} from {cfg.saved_action_head_path}")
    elif module_name == "proprio_projector" and cfg.saved_proprio_path:
        module = load_module_from_path(cfg.saved_proprio_path, module_name, module, device_id)
        print(f"✅ Loaded pretrained {module_name} from {cfg.saved_proprio_path}")

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)

def load_module_from_path(saved_path: str, module_name: str, module: nn.Module, device_id: int) -> nn.Module:
    state_dict = torch.load(saved_path, map_location="cpu")
    model_keys = list(module.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    def _all_with_prefix(keys, prefix="module."):
        return all(k.startswith(prefix) for k in keys)

    def _all_without_prefix(keys, prefix="module."):
        return all(not k.startswith(prefix) for k in keys)

    if _all_without_prefix(model_keys) and _all_with_prefix(ckpt_keys):
        new_sd = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict = new_sd
    elif _all_with_prefix(model_keys) and _all_without_prefix(ckpt_keys):
        new_sd = {"module." + k: v for k, v in state_dict.items()}
        state_dict = new_sd

    module.load_state_dict(state_dict, strict=True)
    del state_dict
    return module

def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    use_rnn_regression,
    rnn_prev_state=None,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
):
    metrics = {}
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    if not (use_l1_regression or use_diffusion or use_rnn_regression):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        curr_action_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        next_actions_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        metrics.update(
            {
                "loss_value": loss.item(),
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    else:
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)

        if use_l1_regression:
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
        elif use_rnn_regression:
            input_dim = text_hidden_states.shape[-1]
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]
                .reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, input_dim)
                .reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM * input_dim)
                .to(torch.bfloat16)
            )
            predicted_actions, rnn_prev_state = action_head.module.predict_action(actions_hidden_states, rnn_prev_state)
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
        elif use_diffusion:
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla, action_head, noisy_action_projector, proprio_projector, batch, batch_size, num_patches,
                        ground_truth_actions.shape, device_id, current_action_mask, next_actions_mask, use_proprio, use_film
                    )

        metrics.update({"loss_value": loss.item()})
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update({"curr_action_l1_loss": curr_action_l1_loss.item(), "next_actions_l1_loss": next_actions_l1_loss.item()})

    return loss, metrics, rnn_prev_state

def run_diffusion_sampling(
    vla, action_head, noisy_action_projector, proprio_projector, batch, batch_size, num_patches,
    actions_shape, device_id, current_action_mask, next_actions_mask, use_proprio, use_film
) -> torch.Tensor:
    noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device_id, dtype=torch.bfloat16)
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)
    curr_noisy_actions = noise

    for t in action_head.module.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device).unsqueeze(1)
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
            last_hidden_states = output.hidden_states[-1]
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)

def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics

def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)

def save_training_checkpoint(
    cfg, run_dir, log_step, vla, processor, proprio_projector, noisy_action_projector, action_head, train_dataset, distributed_state
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    dist.barrier()
    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")
        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}")
        if (cfg.use_l1_regression or cfg.use_diffusion or cfg.use_rnn_regression) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
        if cfg.use_film:
            torch.save(vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}")

    dist.barrier()
    if cfg.use_lora and cfg.merge_lora_during_training and cfg.save_vla:
        base_vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        dist.barrier()

def run_validation(
    vla, action_head, noisy_action_projector, proprio_projector, val_dataloader, action_tokenizer, device_id, cfg, num_patches, log_step, distributed_state, val_time_limit
) -> None:
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0
    rnn_prev_state = None
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            if cfg.rnn_in_batch or (cfg.use_rnn_regression and lang_flag != batch["language_instructions"]):
                lang_flag = batch["language_instructions"]
                rnn_prev_state = None
            _, metrics, rnn_prev_state = run_forward_pass(
                vla, action_head, noisy_action_projector, proprio_projector, batch, action_tokenizer, device_id,
                cfg.use_l1_regression, cfg.use_diffusion, cfg.use_proprio, cfg.use_film, num_patches,
                cfg.use_rnn_regression, rnn_prev_state, True, cfg.num_diffusion_steps_train if cfg.use_diffusion else None
            )
            if cfg.use_rnn_regression:
                if isinstance(rnn_prev_state, tuple):
                    rnn_prev_state = tuple(s.detach() for s in rnn_prev_state)
                else:
                    rnn_prev_state = rnn_prev_state.detach()
            metrics["Val loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1
            if time.time() - val_start_time > val_time_limit:
                break

    avg_val_metrics = {metric_name: sum(m[metric_name] for m in all_val_metrics if metric_name in m) / len([m for m in all_val_metrics if metric_name in m]) 
                       for metric_name in all_val_metrics[0].keys()}
    avg_val_metrics["val_batches_count"] = val_batches_count
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot do both L1 regression and diffusion. Please pick one!"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    print(f"Detected constants:\n\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n\tACTION_DIM: {ACTION_DIM}\n\tPROPRIO_DIM: {PROPRIO_DIM}\n\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        lora_config = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout, target_modules="all-linear", init_lora_weights="gaussian")
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim)
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    vla = wrap_ddp(vla, device_id, find_unused=True)

    if cfg.use_proprio:
        proprio_projector = init_module(ProprioProjector, "proprio_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM})
    if cfg.use_l1_regression:
        action_head = init_module(L1RegressionActionHead, "action_head", cfg, device_id, {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM}, to_bf16=True)
    if cfg.use_rnn_regression:
        action_head = init_module(MLP_RNN_ActionHead, "mlp_rnn_action_head", cfg, device_id, {"input_dim": vla.module.llm_dim, "action_dim": ACTION_DIM, 'rnn_type': cfg.rnn_type}, to_bf16=True)
    if cfg.use_diffusion:
        action_head = init_module(DiffusionActionHead, "action_head", cfg, device_id, {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM, "num_diffusion_steps_train": cfg.num_diffusion_steps_train}, to_bf16=True)
        noisy_action_projector = init_module(NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim})

    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    lora_params = [p for n, p in vla.named_parameters() if "lora" in n and p.requires_grad]
    non_lora_params = [p for n, p in vla.named_parameters() if "lora" not in n and p.requires_grad]
    trainable_params = [
        {"params": non_lora_params, "lr": cfg.learning_rate * 0.1},
        {"params": lora_params, "lr": cfg.learning_rate},
    ]
    if cfg.use_l1_regression or cfg.use_diffusion or cfg.use_rnn_regression:
        trainable_params.append({"params": [p for p in action_head.parameters() if p.requires_grad], "lr": cfg.learning_rate})
    if cfg.use_diffusion:
        trainable_params.append({"params": [p for p in noisy_action_projector.parameters() if p.requires_grad], "lr": cfg.learning_rate})
    if cfg.use_proprio:
        trainable_params.append({"params": [p for p in proprio_projector.parameters() if p.requires_grad], "lr": cfg.learning_rate})

    optimizer = AdamW(trainable_params)
    original_lr = cfg.learning_rate
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer, image_transform=processor.image_processor.apply_transform, prompt_builder_fn=PurePromptBuilder, use_wrist_image=use_wrist_image, use_proprio=cfg.use_proprio)
    train_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, resize_resolution=tuple(vla.module.config.image_sizes), shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug)
    if cfg.use_val_set:
        val_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, resize_resolution=tuple(vla.module.config.image_sizes), shuffle_buffer_size=cfg.shuffle_buffer_size // 10, image_aug=cfg.image_aug, train=False)

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    if cfg.use_val_set:
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Training loop with gradient accumulation
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        lang_flag = ""
        rnn_prev_state = None
        accumulation_counter = 0

        for batch_idx, batch in enumerate(dataloader):
            if cfg.rnn_in_batch or (cfg.use_rnn_regression and lang_flag != batch["language_instructions"]):
                lang_flag = batch["language_instructions"]
                rnn_prev_state = None

            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics, rnn_prev_state = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                use_rnn_regression=cfg.use_rnn_regression,
                num_patches=NUM_PATCHES,
                rnn_prev_state=rnn_prev_state,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            if cfg.use_rnn_regression:
                if isinstance(rnn_prev_state, tuple):
                    rnn_prev_state = tuple(s.detach() for s in rnn_prev_state)
                else:
                    rnn_prev_state = rnn_prev_state.detach()

            # Normalize loss for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            accumulation_counter += 1
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

            # Perform optimization step after accumulating gradients
            if accumulation_counter == cfg.grad_accumulation_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()
                accumulation_counter = 0

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(cfg, run_dir, log_step, vla, processor, proprio_projector, noisy_action_projector, action_head, train_dataset, distributed_state)

            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(vla, action_head, noisy_action_projector, proprio_projector, val_dataloader, action_tokenizer, device_id, cfg, NUM_PATCHES, log_step, distributed_state, cfg.val_time_limit)
                vla.train()

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

if __name__ == "__main__":
    finetune()