import os
import random
import warnings
from typing import Optional

import numpy as np
import torch
from transformers import get_scheduler
from transformers import set_seed as transformers_seed


def seed(seed: Optional[int]):
    """Set random seeds for reproducible training across all libraries.

    Args:
        seed (Optional[int]): Seed value. If None, seeding is skipped.
    """
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clip_gradients(model, args):
    """Clip model gradients by global norm when ``args.max_grad_norm`` is set.

    Args:
        model: The model whose gradients should be clipped.
        args: Argument namespace containing ``max_grad_norm``.
    """
    if args.max_grad_norm is None:
        return
    parameters_to_clip = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=args.max_grad_norm)

def create_scheduler(optimizer, args):
    """Build a learning-rate scheduler based on ``args.schedule_name``.

    Args:
        optimizer: The optimizer to schedule.
        args: Argument namespace with ``schedule_name``, ``warmup_steps``,
            ``max_steps`` and optional scheduler-specific kwargs.

    Returns:
        A Transformers LR scheduler instance.
    """
    if args.schedule_name == 'polynomial':
        specific_kwargs = {'power': args.polynomial_power}
    elif args.schedule_name == 'cosine_with_restarts':
        specific_kwargs = {'num_cycles': args.num_cycles}
    else:
        specific_kwargs = None

    scheduler = get_scheduler(
        args.schedule_name,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        scheduler_specific_kwargs=specific_kwargs
    )
    return scheduler

def save_check_point(model, args, tokenizer, accelerator):
    """Save a model checkpoint in a DDP/DeepSpeed-safe manner.

    Args:
        model: The wrapped model (e.g. DDP / DeepSpeedEngine).
        args: Argument namespace with ``peft_model`` output directory.
        tokenizer: Tokenizer to save alongside the model.
        accelerator: Hugging Face Accelerator instance.
    """
    warnings.simplefilter("ignore")
    # 1. Synchronize: let faster GPUs wait so all ranks finish their training steps
    accelerator.wait_for_everyone()

    # 2. Unwrap DeepSpeedEngine and DDP to get the TKG_Adapter
    unwrapped_model = accelerator.unwrap_model(model)

    # 3. Let the accelerator assemble parameter shards into a full state dict.
    # Must pass the wrapped model, not the unwrapped_model.
    state_dict = accelerator.get_state_dict(model)

    # 4. Only the main process (rank 0) writes to disk to avoid file conflicts
    if accelerator.is_main_process:
        save_directory = args.peft_model
        import os
        os.makedirs(save_directory, exist_ok=True)

        if hasattr(unwrapped_model, "config"):
            # 1. If config stores a real torch.device object, convert it to a string (e.g. "cuda:0")
            if hasattr(unwrapped_model.config, "device") and not isinstance(unwrapped_model.config.device, str):
                unwrapped_model.config.device = str(unwrapped_model.config.device)
        
        # If using the standard HuggingFace/PEFT method
        if hasattr(unwrapped_model, "save_pretrained"):
            # Must pass the assembled state_dict, otherwise saving will fail
            if tokenizer is not None:
                tokenizer.save_pretrained(save_directory)
            unwrapped_model.save_pretrained(save_directory, safe_serialization=False,state_dict=state_dict)
            accelerator.print(f"模型已通过 save_pretrained 成功保存至 {save_directory}")
            
        # The most robust PyTorch native save path (recommended for highly custom architectures)
        else:
            save_path = os.path.join(save_directory, "tkg_adapter_weights.pt")
            torch.save(state_dict, save_path)
            accelerator.print(f"模型权重已通过 torch.save 成功保存至 {save_path}")

    # 5. Signal the accelerator that the job is fully done
    accelerator.end_training()
