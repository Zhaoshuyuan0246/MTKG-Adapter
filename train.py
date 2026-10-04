import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from mydata import TKGAdapterDataset
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from Conv import PretrainedConv1D
from utils.args import Args
from utils.func import clip_gradients, save_check_point
from utils.model import TKG_Adapter

ACCUMULATION_STEPS = 1
transformers.logging.set_verbosity_error()


def collect_layer_insert_loss(model, args):
    """Collect the layer-insertion gate regularization loss.

    Args:
        model: The wrapped TKG-Adapter model.
        args: Argument namespace with ``lambda_binary``, ``lambda_margin`` and
            ``gate_margin`` hyperparameters.

    Returns:
        A scalar tensor. Zero when no layer-insert manager is present.
    """
    if not hasattr(model.model.language_model, "layer_insert_manager"):
        return torch.tensor(0.0, device=next(model.parameters()).device)

    manager = model.model.language_model.layer_insert_manager

    gate_loss = manager.regularization_loss(
        lambda_binary=getattr(args, "lambda_binary", 1e-3),
        lambda_margin=getattr(args, "lambda_margin", 1e-2),
        margin=getattr(args, "gate_margin", 0.2),
    )

    return gate_loss


def print_layer_insert_gates(model, accelerator, print_all: bool = True):
    """Print the KG-Adapter layer-insertion gate states.

    Compatible with the global Top-K gate manager and the legacy independent
    ``KgAdapterLayerInsert`` fallback.

    Args:
        model: The wrapped model to inspect.
        accelerator: Hugging Face Accelerator instance (for distributed printing).
        print_all (bool): If True, print every layer; otherwise a compact summary.
    """
    unwrapped_model = accelerator.unwrap_model(model)

    # ===== Prefer the Global Top-K Gate Manager =====
    try:
        language_model = unwrapped_model.model.language_model

        if hasattr(language_model, "layer_insert_manager"):
            manager = language_model.layer_insert_manager
            probs, hard = manager.gate_stats()

            selected_layers = torch.nonzero(hard > 0).view(-1).tolist()

            accelerator.print("===== Global Top-K KG Adapter Gates =====")
            accelerator.print(f"Selected layers: {selected_layers}")
            accelerator.print(
                f"Active layers: {int(hard.sum().item())}, "
                f"Target active layers: {manager.target_active_layers}"
            )

            if print_all:
                for i in range(len(probs)):
                    accelerator.print(
                        f"Layer {i}: "
                        f"prob={probs[i].item():.4f}, "
                        f"hard_gate={int(hard[i].item())}"
                    )
            else:
                compact = ", ".join(
                    [
                        f"L{i}:{probs[i].item():.3f}/{int(hard[i].item())}"
                        for i in range(len(probs))
                    ]
                )
                accelerator.print(compact)

            return

    except Exception as e:
        accelerator.print(f"[Warning] Failed to print global gate manager: {e}")

    # ===== Fallback: legacy independent KgAdapterLayerInsert =====
    accelerator.print("===== Independent KG Adapter Layer Insert Gates =====")

    for name, module in unwrapped_model.named_modules():
        if module.__class__.__name__ == "KgAdapterLayerInsert":
            if hasattr(module, "gate_info"):
                info = module.gate_info()
                accelerator.print(
                    f"{name}: "
                    f"prob={info['prob']:.4f}, "
                    f"hard_gate={info['hard_gate']}, "
                    f"logit={info['logit']:.4f}"
                )

def train(model: PeftModel,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          tokenizer: PreTrainedTokenizer,
          train_dataloader: DataLoader,
          G_model,
          device: Optional[str],
          args,
          accelerator: Accelerator):
    """Run the main training loop.

    Args:
        model: The PEFT-wrapped model to train.
        optimizer: Optimizer with grouped parameter groups (normal vs speedup LR).
        scheduler: Learning-rate scheduler.
        tokenizer: Tokenizer used for the base model.
        train_dataloader: Training data loader.
        G_model: Pre-trained temporal graph model (RE-GCN) used to produce ``graph_emb``.
        device: Target device.
        args: Argument namespace.
        accelerator: Hugging Face Accelerator instance.
    """

    loss_list = []
    updates_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_steps = updates_per_epoch * args.num_epochs
    bar = tqdm(total=total_steps, ncols=80, disable=not accelerator.is_main_process)
    loss_fn = torch.nn.CrossEntropyLoss(
                    ignore_index=-100,
                    label_smoothing=getattr(args, 'label_smoothing', 0.0)
                )

    def take_step(batched_source):
        clip_gradients(model, args)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        bar.update(1)
        bar.set_postfix(loss=loss_list[-1])

    def loop():
        step = 0
        finished = False

        for epoch in range(args.num_epochs):
            for batch in train_dataloader:
                if batch is None:
                    continue

                model.train()

                with accelerator.accumulate(model):

                    evolve_emb = node_emb = rel_emb = None
                    if batch[0] is not None and batch[1] is not None and batch[2] is not None:
                        evolve_emb = batch[0].to(accelerator.device)
                        node_emb   = batch[1].to(accelerator.device)
                        rel_emb    = batch[2].to(accelerator.device)

                    input_tensors = batch[5].to(accelerator.device)
                    labels = input_tensors.labels

                    graph_emb = None
                    if args.use_modality_graph and node_emb is not None and rel_emb is not None:
                        if args.ConvGraph:
                            graph_emb = PretrainedConv1D(G_model).to(accelerator.device)(
                                torch.cat([node_emb.unsqueeze(1), rel_emb.unsqueeze(1)], dim=1)
                            )
                        else:
                            graph_emb = torch.cat([node_emb, rel_emb], dim=-1)   # [1, 400]
                            graph_emb = graph_emb.view(graph_emb.shape[0], -1, graph_emb.shape[-1])  # [1, 200, 128]

                        if torch.isnan(graph_emb).any() or torch.isinf(graph_emb).any():
                            graph_emb = torch.nan_to_num(graph_emb, nan=0.0, posinf=65500.0, neginf=-65500.0)
                        graph_emb = graph_emb.to(dtype=accelerator.unwrap_model(model).dtype)
                        

                    output = model(
                        input_ids=input_tensors.input_ids,
                        pixel_values=getattr(input_tensors, 'pixel_values', None),
                        image_grid_thw=getattr(input_tensors, 'image_grid_thw', None),
                        attention_mask=input_tensors.attention_mask,
                        labels=None,
                        graph_emb=graph_emb,
                        sg=None,
                        use_cache=False,
                    )

                    logits       = output.logits
                    del output
                    vocab_size   = logits.size(-1)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    task_loss = loss_fn(
                        shift_logits.float().view(-1, vocab_size),
                        shift_labels.view(-1)
                    )

                    unwrapped_model = accelerator.unwrap_model(model)

                    gate_loss = collect_layer_insert_loss(unwrapped_model, args)

                    sparse_warmup_steps = getattr(args, "sparse_warmup_steps", 100)
                    if step < sparse_warmup_steps:
                        gate_loss = gate_loss * 0.0

                    loss = task_loss + gate_loss

                    # loss = accelerator.unwrap_model(model).model.language_model.adapter_router_manager.get_auxiliary_loss(loss, input_tensors.attention_mask)

                    del logits, shift_logits, shift_labels

                    if hasattr(input_tensors, 'pixel_values') and input_tensors.pixel_values is not None:
                        input_tensors.pixel_values = None

                    accelerator.backward(loss)

                if accelerator.sync_gradients:
                    clip_gradients(model, args)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    accelerator.unwrap_model(model).model.language_model.adapter_router_manager.clear()

                    loss_list.append(loss.item())
                    bar.update(1)
                    bar.set_postfix(loss=loss_list[-1], 
                                    ce=f"{task_loss.detach().item():.4f}",
                                    gate=f"{gate_loss.detach().item():.4f}")
                    step += 1

                    if step % 50 == 0:
                        lrs = [g['lr'] for g in optimizer.param_groups]
                        lr_str = ', '.join(f'{lr:.2e}' for lr in lrs)

                        accelerator.print(
                            f"Step {step}, "
                            f"loss={loss_list[-1]:.4f}, "
                            f"ce={task_loss.detach().item():.4f}, "
                            f"gate={gate_loss.detach().item():.4f}, "
                            f"lr=[{lr_str}]"
                        )

                        if accelerator.is_main_process:
                            print_layer_insert_gates(
                                model=model,
                                accelerator=accelerator,
                                print_all=True
                            )

                if args.max_steps is not None and step > args.max_steps:
                    finished = True
                    break

                del loss

            torch.cuda.empty_cache()
            if finished:
                break

    loop()

    save_check_point(model, args, tokenizer, accelerator)

    bar.close()


def main(args):
    """Entry point: fix the seed, build the model/optimizer and start training.

    Args:
        args: Parsed command-line arguments (from ``utils.args.Args``).
    """
    # Step 1: fix the multi-GPU seed.
    set_seed(42)

    # Instantiate the Accelerator inside main(), after the seed is fixed
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs]
    )

    # Print args inside main(), since the accelerator is now initialized here
    if accelerator.is_main_process:
        accelerator.print("ARGS:\n", args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    accelerator.print("正在加载 Processor...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    accelerator.print("正在初始化训练集...")
    train_dataset = TKGAdapterDataset(args, processor, accelerator=accelerator, mode="train")

    _raw_collate = train_dataset.collate_fn
    def filtering_collate(batch):
        result = _raw_collate(batch)
        if result is None:
            accelerator.print("[Warning] Skipping a batch: response completely truncated by prompt length.")
            return None
        return result

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=filtering_collate,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        total_steps = (len(train_dataset) // args.batch_size) * args.num_epochs
        accelerator.print(f"Total training steps expected: {total_steps}")

    args.device = accelerator.device
    model = TKG_Adapter(args, tokenizer, processor, accelerator)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "llm") and hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    speedup_param_name = ['lora_b', 'TKGAdapter', 'LowRankAdapterRouter']
    norm_param, speedup_param = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in speedup_param_name):
            speedup_param.append(param)
        else:
            norm_param.append(param)

    speedup_lr = getattr(args, 'speedup_lr', args.lr * 10.0)

    optimizer_grouped_parameters = [
        g for g in [
            {"params": norm_param,    "lr": args.lr},
            {"params": speedup_param, "lr": speedup_lr},
        ] if len(g["params"]) > 0
    ]

    accelerator.print(f"慢速参数组: {len(norm_param)} 个参数，lr={args.lr:.2e}")
    accelerator.print(f"快速参数组: {len(speedup_param)} 个参数，lr={speedup_lr:.2e}")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_steps = (len(train_dataloader) // ACCUMULATION_STEPS) * args.num_epochs  
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    train(
        model,
        optimizer,
        scheduler,
        tokenizer,
        train_dataloader,
        train_dataset.model,
        args.device,
        args,
        accelerator  # pass the initialized accelerator
    )


if __name__ == '__main__':
    args = Args()
    main(args)