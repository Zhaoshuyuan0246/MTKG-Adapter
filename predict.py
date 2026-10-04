import argparse
import glob
import json
import os
import re

import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen3VLForConditionalGeneration,
)

from Conv import PretrainedConv1D
from lora.model import TKGAdapterModel
from mydata import TKGAdapterDataset

def get_qwen_digit_tokens(tokenizer):
    """Pre-compute all token IDs that start with a digit (executed once).

    Args:
        tokenizer: Tokenizer whose vocabulary is scanned.

    Returns:
        A list of token IDs whose decoded string starts with a digit.
    """
    digit_tokens = set()
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        clean_str = token_str.lstrip("Ġ ▂▃▄▅▆▇█")
        if len(clean_str) > 0 and clean_str[0].isdigit():
            digit_tokens.add(token_id)
    return list(digit_tokens)

class ControlledDigitLogitsProcessor(LogitsProcessor):
    """Logits processor that forces the first generated token to start with a digit.

    Masks out non-digit tokens on the first decoding step and penalizes repeated
    digits when the same digit is generated ``max_retry`` times in a row.
    """

    def __init__(self, tokenizer, input_length, valid_digit_tokens, history=None, max_retry=2):
        """Initialize the processor.

        Args:
            tokenizer: Tokenizer used to encode digits.
            input_length: Prompt length; used to detect the first generation step.
            valid_digit_tokens: Pre-computed digit-leading token IDs.
            history: Shared history list tracking previously generated digits.
            max_retry: Number of repeated digits allowed before penalizing.
        """
        self.tokenizer = tokenizer
        self.input_length = input_length  
        self.history = history if history else []
        self.max_retry = max_retry
        
        # reuse the externally precomputed list (zero cost)
        self.digit_tensor = torch.tensor(valid_digit_tokens, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        current_gen_len = input_ids.shape[-1] - self.input_length
        min_value = -1e4 
        
        # force the first generated token to start with a digit (allow full tokens like 253, 3123)
        if current_gen_len == 0:
            device = scores.device
            digit_tensor = self.digit_tensor.to(device)
            
            mask = torch.ones(scores.shape[1], dtype=torch.bool, device=device)
            mask[digit_tensor] = False 
            scores[:, mask] = min_value
            
            # anti-repetition logic
            if len(self.history) >= self.max_retry:
                last_digit = self.history[-1]
                if self.history[-self.max_retry:] == [last_digit] * self.max_retry:
                    last_digit_tokens = self.tokenizer.encode(str(last_digit), add_special_tokens=False)
                    if last_digit_tokens:
                        last_digit_token_id = last_digit_tokens[0]
                        scores[:, last_digit_token_id] = min_value
        
        return scores

class HistoryTracker:
    """Tracks recently generated digits to detect repeated outputs."""

    def __init__(self, max_history=2):
        self.history = []
        self.max_history = max_history

    def add(self, digit: int):
        """Append a digit, evicting the oldest entry if capacity is exceeded."""
        self.history.append(digit)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def need_adjust(self):
        """Return True when all tracked digits are identical (repetition detected)."""
        if len(self.history) < self.max_history:
            return False
        return all(x == self.history[0] for x in self.history)

def str2bool(v):
    """Parse a string into a boolean, raising on invalid input."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def parse_config():
    """Parse command-line arguments for the inference/evaluation script."""
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="MLLM/Qwen3-VL-8B-Instruct", help='base model path')
    parser.add_argument('--context_size', type=int, default=2048, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default="checkpoint/Qwen3_8B/", help='')
    parser.add_argument('--dataset', type=str, default="GDELT")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--last_token', default=None)
    parser.add_argument('--max_new_tokens', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=20)
    parser.add_argument('--max_retry_attempts', type=int, default=2)
    # ----------------------------ReGCN-------------------------------
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")    # statistic the result
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")   # multi-step inference
    # parser.add_argument("--topk", type=int, default=10,
    #                     help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution") # save model according to relation evaluation

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    # parser.add_argument("--dropout", type=float, default=0.2,
    #                     help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=True,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    # parser.add_argument("--lr", type=float, default=0.001,
    #                     help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=2,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=2,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")    # span of history time steps

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")  # grid search over hyperparameters
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use") # list of hyperparameters to tune
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated") # number of triples to generate
    
    parser.add_argument("--peft_path", type=str, default="RE-GCN_premodel/GDELT", help="peft path")
    parser.add_argument("--dropout", action='store_true', default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ConvGraph", type=bool, default=False)
    parser.add_argument("--use_Conv", type=bool, default=False)

    parser.add_argument("--use_modality_image", type=str2bool, default=True)
    parser.add_argument("--use_modality_graph", type=str2bool, default=True)
    parser.add_argument("--use_text_moe",        type=str2bool, default=True)
    parser.add_argument("--use_image_moe",       type=str2bool, default=True)
    parser.add_argument("--use_graph_moe",       type=str2bool, default=True)
    parser.add_argument("--use_cross_attention", type=str2bool, default=True)
    parser.add_argument("--use_channel_attention",type=str2bool,default=False)
    parser.add_argument("--use_lora",            type=str2bool, default=False)

    args = parser.parse_args()
    return args

def extract_first_digit(sequence, tokenizer):
    # response looks like "253.moselle (département)"; take the first contiguous digit run
    decoded = tokenizer.decode(sequence)
    # find the first contiguous digit run first (does not rely on the '.' separator, more robust)
    m = re.search(r'\d+', decoded)
    if m:
        return int(m.group())
    return None
# digit generation with a retry mechanism
def generate_with_retry(model, tokenizer, inputs, graph_emb, args, history_tracker, valid_digit_tokens,accelerator):
    best_candidates = []
    # use ControlledDigitLogitsProcessor to force digit generation at specific positions
    input_len = inputs.input_ids.shape[-1]
    for attempt in range(args.max_retry_attempts + 1):
        logits_processor = ControlledDigitLogitsProcessor(
            tokenizer=tokenizer,
            input_length=input_len,
            valid_digit_tokens=valid_digit_tokens, # pass it the precomputed list
            history=history_tracker.history,
            max_retry=args.max_retry_attempts
        )
        with torch.no_grad():
            # clear the previous cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            inputs.pop("token_type_ids", None)
            outputs = unwrapped_model.generate(
                **inputs,  
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_return_sequences * 2,
                num_return_sequences=args.num_return_sequences,
                do_sample=False,
                logits_processor=LogitsProcessorList([logits_processor]),
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=False,
                graph_emb=graph_emb,
                sg=None,
            )
            
            text = tokenizer.decode(outputs.sequences[0][len(inputs.input_ids[0]):])

        current_candidates = []
        input_len = inputs.input_ids.shape[1]
        
        for i, seq in enumerate(outputs.sequences):
            # extract the generated part
            generated_seq = seq[input_len:]
            score = outputs.sequences_scores[i].item()
            digit = extract_first_digit(generated_seq, tokenizer)
            if digit is not None:
                current_candidates.append((digit, score, generated_seq))

        del outputs # free memory if needed

        # prefer digits that appear for the first time
        for candidate in sorted(current_candidates, key=lambda x: -x[1]):
            if candidate[0] not in [x[0] for x in best_candidates]:
                best_candidates.append(candidate)
                history_tracker.add(candidate[0])
                if len(best_candidates) >= args.num_return_sequences:
                    return best_candidates

        # if still not enough, add the remaining candidates
        for candidate in sorted(current_candidates, key=lambda x: -x[1]):
            # manually check whether the candidate is already in best_candidates
            is_already_best = False
            current_digit, current_score, current_seq_tensor = candidate # unpack the current candidate

            for best_digit, best_score, best_seq_tensor in best_candidates: # iterate over the collected best candidates
                # compare the non-tensor parts for equality and the tensor contents
                if current_digit == best_digit and current_score == best_score and torch.equal(current_seq_tensor, best_seq_tensor):
                    is_already_best = True
                    break # match found, no need to keep checking

            if not is_already_best: # if the current candidate is not among the collected best candidates
                best_candidates.append(candidate)
                history_tracker.add(candidate[0])
                if len(best_candidates) >= args.num_return_sequences:
                    return best_candidates

    return best_candidates[:args.num_return_sequences]# digit: first extracted digit (int), score: sequence confidence (float), sequence_tensor: full generated sequence (torch.Tensor)

def main(args):
    """Entry point for inference/evaluation under DDP.

    Args:
        args: Parsed command-line arguments (from ``parse_config``).
    """
    set_seed(42)

    # initialize the Accelerator
    accelerator = Accelerator(mixed_precision="bf16")

    base_model = args.base_model
    peft_path = args.peft_model
    
    if accelerator.is_main_process:
        print("base model: ", base_model)
        print("peft model:", peft_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=2048,
        # padding_side="right",
        use_fast=True,
        truncation=True # enable truncation
    )
    if tokenizer.padding_side == 'right':
        tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print("正在缓存数字 Token 词表...")
    valid_digit_tokens = get_qwen_digit_tokens(tokenizer)

    # Load model and tokenizer
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map={"": accelerator.device},
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        
    if accelerator.is_main_process:
        print("Success1...base_model")

    base_model.config.use_modality_image = args.use_modality_image
    base_model.config.use_modality_graph = args.use_modality_graph
    base_model.config.use_text_moe = args.use_text_moe
    base_model.config.use_image_moe = args.use_image_moe
    base_model.config.use_graph_moe = args.use_graph_moe
    base_model.config.use_cross_attention = args.use_cross_attention
    base_model.config.use_channel_attention = args.use_channel_attention
    base_model.config.use_lora = args.use_lora
    
    peft_model = TKGAdapterModel.from_pretrained(
            base_model,
            args.peft_model  # path must end with '/', because "config.json" is appended internally
        )
    
    if accelerator.is_main_process:
        print("Success2...peft_model")


    adapter_dir = args.peft_model
    sf_files  = sorted(glob.glob(os.path.join(adapter_dir, "*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(adapter_dir, "*.bin")))
    pt_files  = sorted(glob.glob(os.path.join(adapter_dir, "*.pt")))

    state_dict = {}
    if sf_files:
        from safetensors.torch import load_file
        for f in sf_files:
            state_dict.update(load_file(f, device="cpu"))
    elif bin_files:
        for f in bin_files:
            state_dict.update(torch.load(f, map_location="cpu"))
    elif pt_files:
        for f in pt_files:
            state_dict.update(torch.load(f, map_location="cpu"))

    model_keys = set(peft_model.state_dict().keys())

    # -- Frozen-param key set (requires_grad=False, never saved during training; missing is normal) --
    frozen_keys = {
        name for name, param in peft_model.named_parameters()
        if not param.requires_grad
    }

    # -- Key remapping: try in priority order, stop at the first match --
    def remap_key(k: str, model_keys: set) -> str:
        if k in model_keys:
            return k  # exact match, return directly

        # Rule 1: strip the DDP prefix 'module.'
        k1 = k.replace("module.", "", 1) if k.startswith("module.") else k
        if k1 in model_keys:
            return k1

        # Rule 2: add the top-level 'model.' prefix (older saved paths lack the outer wrapper)
        k2 = f"model.{k1}"
        if k2 in model_keys:
            return k2

        # Rule 3: model.layers.* / model.embed_tokens.* lack the language_model layer
        # saved path: model.layers.0.TKGAdapter.*
        # expected path: model.language_model.layers.0.TKGAdapter.*
        if (k1.startswith("model.")
                and not k1.startswith("model.language_model.")
                and not k1.startswith("model.visual.")
                and not k1.startswith("model.lm_head.")):
            suffix = k1[len("model."):]          # take the part after 'model.'
            k3 = f"model.language_model.{suffix}"
            if k3 in model_keys:
                return k3

        # Rule 4: the key has no top-level wrapper at all, fill in the path directly
        if not k1.startswith("model.language_model."):
            k4 = f"model.language_model.{k1}"
            if k4 in model_keys:
                return k4

        return k1  # fallback: return the original key after stripping the DDP prefix

    new_state_dict = {}
    remap_log = []  # record keys that were remapped (for diagnostics)
    for k, v in state_dict.items():
        new_k = remap_key(k, model_keys)
        if new_k != k:
            remap_log.append((k, new_k))
        new_state_dict[new_k] = v

    if accelerator.is_main_process and remap_log:
        print(f"[load] 发生 key 重映射 {len(remap_log)} 条，示例：")
        for old_k, new_k in remap_log[:3]:
            print(f"  {old_k}  ->  {new_k}")

    # -- Load weights --
    missing, unexpected = peft_model.load_state_dict(new_state_dict, strict=False)

    # -- Diagnostic output --
    # Missing entries that belong to frozen params are normal (pretrained weights were never saved);
    # what really matters is "trainable params missing" and any unexpected keys
    trainable_missing = [k for k in missing if k not in frozen_keys]

    accelerator.print(
        f"Success3...权重加载完成 | "
        f"missing(冻结/正常): {len(missing) - len(trainable_missing)} | "
        f"missing(可训练/异常): {len(trainable_missing)} | "
        f"unexpected: {len(unexpected)}"
    )

    if accelerator.is_main_process:
        if trainable_missing:
            print(f"⚠️ [异常] 以下可训练参数未能加载（共 {len(trainable_missing)} 个）：")
            for k in trainable_missing[:5]:
                print(f"  {k}")
        if unexpected:
            print(f"⚠️ [异常] 以下 key 在模型中找不到（共 {len(unexpected)} 个）：")
            for k in list(unexpected)[:5]:
                print(f"  {k}")
        if not trainable_missing and not unexpected:
            print("✅ 所有可训练 Adapter 权重已正确加载，冻结参数沿用预训练权重（正常）")

    peft_model = peft_model.to(accelerator.device)
    peft_model = peft_model.to(torch.bfloat16)

    if accelerator.is_main_process:
        print("----------------------------")
        print("alpha: ", args.alpha)
        print("----------------------------")
    
    # load the training set
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    train_dataset = TKGAdapterDataset(args, processor, accelerator=accelerator, mode="test")
    

    if accelerator.is_main_process:
        print("len: ", len(train_dataset))
        print("test_dataset: \n", train_dataset.data.test[0])
        print("test_text: \n", train_dataset.data.test_text[0])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn_test,  # bind the custom collate function
        num_workers=0,
    )
    # distributed training preparation and optimization
    # train_dataloader, peft_model = accelerator.prepare(train_dataloader, peft_model)
    train_dataloader = accelerator.prepare(train_dataloader)
    
    Hit1, Hit3, Hit10 = 0, 0, 0
    MRR = 0.0
    peft_model.eval()
    history_tracker = HistoryTracker(max_history=2)
    
    for batch in tqdm(train_dataloader, total=len(train_dataloader), disable=not accelerator.is_local_main_process):
        
        evolve_emb = batch[0].to(accelerator.device) if batch[0] is not None else None
        node_emb   = batch[1].to(accelerator.device) if batch[1] is not None else None
        rel_emb    = batch[2].to(accelerator.device) if batch[2] is not None else None
        instructions = batch[3]
        label = int(batch[4][0].split('.')[0])
        scores = batch[5]
        filter_set = batch[6] 
        input_tensors = batch[7].to(accelerator.device)   

        if isinstance(input_tensors, dict):
            for k, v in input_tensors.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    input_tensors[k] = v.to(dtype=torch.bfloat16)             

        history_tracker.history.clear()
        graph_emb = None
        if args.use_modality_graph and node_emb is not None and rel_emb is not None:
            if args.ConvGraph: # use the convolution method
                node_emb_ = node_emb.unsqueeze(1)
                rel_emb_ = rel_emb.unsqueeze(1)
                graph_emb = torch.cat([node_emb_, rel_emb_], dim=1) # shape: [bsz, 2, h_dim]
                conv = PretrainedConv1D(train_dataset.model).to(accelerator.device) # load the convolution layer
                graph_emb = conv(graph_emb) # shape: [bsz, h_dim]
            else: # use the concatenation method
                graph_emb = torch.cat([node_emb, rel_emb], dim=-1) # shape: [bsz, 2*h_dim]

            if torch.isnan(graph_emb).any() or torch.isinf(graph_emb).any():
                accelerator.print("Graph Embedding contains NaN/Inf, clipping...")
                graph_emb = torch.nan_to_num(
                    graph_emb, nan=0.0, posinf=65500.0, neginf=-65500.0
                )  
            graph_emb = graph_emb.to(dtype=torch.bfloat16)

        # generate candidate answers
        candidates = generate_with_retry(peft_model, tokenizer, input_tensors, graph_emb, args, history_tracker, valid_digit_tokens,accelerator)

        # optional filter set: Top-K entities predicted by the graph model
        filter_set = set()
        if scores is not None and len(scores) > 0 and "topk_entity_ids" in scores[0]:
            filter_set = set(scores[0]["topk_entity_ids"])

        # initialize the unique-candidate records
        unique_candidates = {}

        # iterate the candidate list: filter + dedupe + keep scores
        for digit, score, _ in candidates:
            # filter illegal or out-of-range digits
            if digit is None or digit < 0 or digit >= train_dataset.num_nodes:
                continue

            # filter entities that may have appeared in training, keep gold (optional)
            if digit in filter_set and digit != label:
                continue

            # dedup logic: keep the higher score
            if digit not in unique_candidates or score > unique_candidates[digit]:
                unique_candidates[digit] = score

        # keep only the Top-K (15 by default) predictions
        sorted_candidates = sorted(unique_candidates.items(), key=lambda x: -x[1])[:args.num_return_sequences]
        top_answers = [x[0] for x in sorted_candidates]

        if sorted_candidates:
            if args.use_Conv: # use the convolution method
                node_emb_ = node_emb.unsqueeze(1)
                rel_emb_ = rel_emb.unsqueeze(1)
                graph_emb = torch.cat([node_emb_, rel_emb_], dim=1) # shape: [bsz, 2, h_dim]
                
                conv = PretrainedConv1D(train_dataset.model).to(accelerator.device) # load the convolution layer
                graph_emb = conv(graph_emb) # shape: [bsz, h_dim]
                
                scores = graph_emb @ evolve_emb.transpose(2, 1)
                scores = scores.squeeze(1)
                scores = torch.softmax(scores, dim=-1)
                
                scores_vector = torch.zeros(scores.size(1)).unsqueeze(0)

                for digit, score in sorted_candidates:
                    if digit <= scores.size(1):  # only handle valid indices
                        scores_vector[0][digit] = score
                    
                nonzero_mask = scores_vector != 0
                nonzero_values = scores_vector[nonzero_mask].clone()
                min_val = torch.min(nonzero_values)
                max_val = torch.max(nonzero_values)
                normalized_nonzero_values = (nonzero_values - min_val) / (max_val - min_val)
                normalized_tensor = torch.zeros_like(scores_vector)
                normalized_tensor[nonzero_mask] = normalized_nonzero_values
                normalized_tensor = torch.softmax(normalized_tensor, dim=-1).to(accelerator.device)
    
                # scores_vector = torch.softmax(scores_vector, dim=-1).to(accelerator.device)
                
                final_scores = args.alpha * normalized_tensor + (1 - args.alpha) * scores
                final_scores = torch.softmax(final_scores, dim=-1)
            else: # directly use the pretrained score table
                # --- Step 1: always handle the LLM (Adapter) generation scores first ---
                scores_vector = torch.zeros(train_dataset.num_nodes).unsqueeze(0)
                for digit, score in sorted_candidates:
                    if digit <= train_dataset.num_nodes:  # only handle valid indices
                        scores_vector[0][digit] = score
                
                nonzero_mask = scores_vector != 0
                normalized_tensor = torch.zeros_like(scores_vector)
                
                if nonzero_mask.any():
                    nonzero_values = scores_vector[nonzero_mask].clone()
                    min_val = torch.min(nonzero_values)
                    max_val = torch.max(nonzero_values)
                    # prevent division by zero when max == min
                    if max_val > min_val:
                        normalized_nonzero_values = (nonzero_values - min_val) / (max_val - min_val)
                    else:
                        normalized_nonzero_values = torch.ones_like(nonzero_values)
                        
                    normalized_nonzero_values_ = torch.softmax(normalized_nonzero_values, dim=-1)
                    normalized_tensor[nonzero_mask] = normalized_nonzero_values_
                
                normalized_tensor = normalized_tensor.to(accelerator.device)

                # --- Step 2: only when alpha < 1 and scores exist, extract and fuse the graph-model scores ---
                if args.alpha < 1.0 and scores is not None and len(scores) > 0:
                    graph_scores = torch.zeros(train_dataset.num_nodes).unsqueeze(0).to(accelerator.device)
                    
                    # safely get entity ID and score, using .get to avoid KeyError
                    entity_ids = scores[0].get("topk_entity_ids", [])[:10]
                    entity_scores = scores[0].get("topk_entity_scores", [])[:10]
                    
                    for entity_id, entity_score in zip(entity_ids, entity_scores):
                        graph_scores[0][entity_id] = entity_score
                        
                    graph_mask = graph_scores != 0
                    if graph_mask.any():
                        nonzero_values = graph_scores[graph_mask].clone()
                        normalized_graph_tensor = torch.softmax(nonzero_values, dim=-1).to(accelerator.device)
                        graph_scores[graph_mask] = normalized_graph_tensor 
                        
                    final_scores = args.alpha * normalized_tensor + (1 - args.alpha) * graph_scores
                else:
                    # when alpha == 1 or no scores, trust the LLM result alone
                    final_scores = normalized_tensor
        else:
            print("Warning: Skipping batch due to error. Fallback to graph predictions.")
            final_scores = torch.zeros(train_dataset.num_nodes).unsqueeze(0).to(accelerator.device)
            
            # only fall back when scores actually exist
            if scores is not None and len(scores) > 0:
                entity_ids = scores[0].get("topk_entity_ids", [])[:10]
                entity_scores = scores[0].get("topk_entity_scores", [])[:10]
                
                for entity_id, entity_score in zip(entity_ids, entity_scores):
                    final_scores[0][entity_id] = entity_score
                    
                # score normalization    
                graph_mask = final_scores != 0
                if graph_mask.any():
                    nonzero_values = final_scores[graph_mask].clone()
                    normalized_graph_tensor = torch.softmax(nonzero_values, dim=-1).to(accelerator.device)
                    final_scores[graph_mask] = normalized_graph_tensor  

        top_scores, top_answers = torch.topk(
            final_scores,
            k=10,          
            dim=-1,        
            largest=True,  
            sorted=True    
        )        
        
        if accelerator.is_main_process:
            print(f"Ground Truth ID: {label}")
            print(f"LLM Top-3: {top_answers[0][:3].tolist()}")
            # print(f"Graph Top-3: {entity_ids[:3]}")
        
        # evaluation logic
        top_list = top_answers[0].tolist()
        if label in top_list:
            if accelerator.is_main_process:
                print("yes...")
            rank = top_list.index(label) + 1   # 1-based rank
            MRR += 1.0 / rank
            Hit10 += 1
            if rank <= 3:
                Hit3 += 1
                if rank == 1:
                    Hit1 += 1
        
        del input_tensors, candidates, graph_emb, node_emb, rel_emb
        if 'evolve_emb' in dir():
            del evolve_emb
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # convert Python numbers to tensors
    hit1_tensor  = torch.tensor(Hit1,  device=accelerator.device)
    hit3_tensor  = torch.tensor(Hit3,  device=accelerator.device)
    hit10_tensor = torch.tensor(Hit10, device=accelerator.device)
    mrr_tensor   = torch.tensor(MRR,   device=accelerator.device)   # added

    Hit1_total  = accelerator.reduce(hit1_tensor,  reduction="sum").item()
    Hit3_total  = accelerator.reduce(hit3_tensor,  reduction="sum").item()
    Hit10_total = accelerator.reduce(hit10_tensor, reduction="sum").item()
    MRR_total   = accelerator.reduce(mrr_tensor,   reduction="sum").item()  # added
    # only print results on the main process
    if accelerator.is_main_process:
        total_samples = len(train_dataset)
        hit1_score  = Hit1_total  / total_samples
        hit3_score  = Hit3_total  / total_samples
        hit10_score = Hit10_total / total_samples
        mrr_score   = MRR_total   / total_samples

        print("--------------")
        print(f'MRR:   {mrr_score:.8f}')
        print(f'Hit@1: {hit1_score:.8f}')
        print(f'Hit@3: {hit3_score:.8f}')
        print(f'Hit@10:{hit10_score:.8f}')
        print("--------------")

        with open("results.txt", "w") as f:
            f.write("--------------\n")
            f.write(f'MRR:   {mrr_score:.8f}\n')
            f.write(f'Hit@1: {hit1_score:.8f}\n')
            f.write(f'Hit@3: {hit3_score:.8f}\n')
            f.write(f'Hit@10:{hit10_score:.8f}\n')
            f.write("--------------\n")

if __name__ == "__main__":
    args = parse_config()
    main(args)