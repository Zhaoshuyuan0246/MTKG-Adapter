import torch
import numpy as np
from torch.utils.data import Dataset
from pre_trained_regcn.rgcn import utils
from pre_trained_regcn.rgcn.utils import build_sub_graph
import random
from PIL import Image
from transformers import AutoProcessor
from pre_trained_regcn.load import load, graph_model

# Global threshold: only skip samples whose response is fully truncated (0 tokens).
MIN_VALID_LABEL_TOKENS = 1


class TKGAdapterDataset(Dataset):
    """PyTorch Dataset for the multi-modal temporal KG reasoning task.

    Loads the RE-GCN graph data, extracts temporal subgraph embeddings on the
    fly, and tokenizes the (image, text) instruction/response pairs.
    """

    def __init__(self, args, processor, accelerator, mode="train"):
        """Initialize the dataset.

        Args:
            args: Argument namespace (dataset, modality flags, history length...).
            processor: Qwen3-VL processor used for image/text tokenization.
            accelerator: Hugging Face Accelerator instance.
            mode (str): Either ``"train"`` or ``"test"``.
        """
        self.args = args
        self.processor = processor
        self.accelerator = accelerator
        self.mode = mode
        
        if self.mode not in ["train", "test"]:
            raise ValueError("mode wrong, should be in ['train', 'test']")
        
        if not hasattr(self.args, 'device'):
            self.args.device = self.accelerator.device

        accelerator.print("loading graph data and model...")
        self.data = utils.load_data(args.dataset)  
        self.model = graph_model(args, self.data)
        
        if self.mode == "train":
            self.current_data = self.data.train
            self.current_text = self.data.train_text
        else:
            self.current_data = self.data.test
            self.current_text = self.data.test_text
            self.pretrained_test_scores = self.data.pretrained_test_scores
            self.test_tail_sets = self.data.test_tail_sets
            
        self.num_nodes = self.data.num_nodes
        self.num_rels = self.data.num_rels
        self.train_list = utils.split_by_time(np.concatenate([self.data.train, self.data.test], axis=0))
        
        self.entity_image_paths = {}
        if getattr(self.args, 'use_modality_image', False):
            if hasattr(self.data, 'entity_images'):
                self.entity_image_paths = self.data.entity_images
                accelerator.print(f"成功加载 {len(self.entity_image_paths)} 个实体的图像数据")
            else:
                accelerator.print("警告: 未找到图像数据")
        
        accelerator.print(f"{self.mode} 数据加载与预处理完成")

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        """Return a single sample (triple, text, optional head/tail images).

        In test mode also returns the pre-computed graph scores and the filter
        tail set used for metric computation.
        """
        trip = self.current_data[idx]
        text_data = self.current_text[idx]

        head_image = None
        tail_image = None

        if getattr(self.args, 'use_modality_image', False):
            FIXED_SIZE = 56

            # head entity image trip[0]
            head_paths = self.entity_image_paths.get(trip[0], [])
            if head_paths:
                try:
                    img = Image.open(random.choice(head_paths)).convert('RGB')
                    img = img.resize((FIXED_SIZE, FIXED_SIZE), Image.Resampling.LANCZOS)
                    head_image = img
                except Exception:
                    head_image = None

            # tail entity image trip[2]
            tail_paths = self.entity_image_paths.get(trip[2], [])
            if tail_paths:
                try:
                    img = Image.open(random.choice(tail_paths)).convert('RGB')
                    img = img.resize((FIXED_SIZE, FIXED_SIZE), Image.Resampling.LANCZOS)
                    tail_image = img
                except Exception:
                    tail_image = None

        if self.mode == "train":
            return trip, text_data, head_image, tail_image
        # return trip, text_data, self.pretrained_test_scores[idx], self.test_tail_sets[idx], head_image, tail_image
        return trip, text_data, None, self.test_tail_sets[idx], head_image, tail_image

    def collate_fn(self, batch):
        """Training collate function (forced ``batch_size=1``).

        Builds prompt/full chat templates, masks the prompt region in ``labels``,
        and extracts the temporal graph embeddings (evolve/node/relation).
        """
        trip, text_data, head_image, tail_image = batch[0]

        # Build user_content: head image first, then tail image, then text
        user_content = []
        if head_image is not None:
            user_content.append({"type": "image", "image": head_image})
        if tail_image is not None:
            user_content.append({"type": "image", "image": tail_image})
        user_content.append({"type": "text", "text": text_data['instruction']})

        text_full = self.processor.apply_chat_template(
            [{"role": "user", "content": user_content},
            {"role": "assistant", "content": text_data['response']}],
            tokenize=False, add_generation_prompt=False
        )
        text_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True
        )

        MAX_SEQ_LEN     = 1024
        MIN_RESP_TOKENS = 64
        max_prompt_len  = MAX_SEQ_LEN - MIN_RESP_TOKENS

        min_pixels = 4  * 14 * 14   # 784
        max_pixels = 16 * 14 * 14   # 3136

        # Assemble the list of actually present images (0, 1 or 2)
        images = [img for img in [head_image, tail_image] if img is not None]
        images_arg = images if images else None  # processor rejects an empty list

        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "right"

        try:
            inputs_prompt = self.processor(
                text=[text_prompt],
                images=images_arg,
                padding="longest",
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            prompt_len = inputs_prompt.input_ids.shape[1]

            inputs = self.processor(
                text=[text_full],
                images=images_arg,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
                truncation=True,
                return_tensors="pt",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            labels = inputs.input_ids.clone()
            labels[:, :prompt_len] = -100
            labels.masked_fill_(inputs.attention_mask.eq(0), -100)

            valid_label_num = (labels != -100).sum().item()
            if valid_label_num < MIN_VALID_LABEL_TOKENS:
                self.accelerator.print(
                    f"[Skip] valid_label_num=0，response 完全截断，跳过此样本"
                )
                return None

            inputs['labels'] = labels

        finally:
            self.processor.tokenizer.padding_side = orig_padding_side

        if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
            thw = inputs.image_grid_thw
            total_vision_tokens = (thw[:, 1] * thw[:, 2]).sum().item()
            if total_vision_tokens > 128:  # two-image cap raised from 64 to 128
                self.accelerator.print(
                    f"[Warning] vision tokens={total_vision_tokens} > 128, "
                    f"image_grid_thw={thw}. Consider reducing image size further."
                )

        # --- Extract TKG graph features on demand (unchanged) ---
        evolve_emb = node_emb = rel_emb = None
        START_TIME = 0

        if getattr(self.args, 'use_modality_graph', True):
            subject, relation, target, time = trip
            if time != START_TIME:
                time_idx = time - START_TIME
                if time >= 32:
                    time_idx = time_idx - 1 
                history_len = min(time_idx, self.args.train_history_len)
                history_glists = [
                    build_sub_graph(self.num_nodes, self.num_rels, snap, [trip], True, 'cpu')
                    for snap in self.train_list[time_idx - history_len : time_idx]
                ]
                evolve_emb, node_emb, rel_emb = load(
                    self.model, history_glists, [trip], static_graph=None, args=self.args
                )
                evolve_emb = (torch.cat(evolve_emb, dim=0) if evolve_emb[0].dim() == 2
                              else torch.stack(evolve_emb)).unsqueeze(0)
                node_emb = (torch.stack(node_emb) if isinstance(node_emb, list) else node_emb)
                node_emb = node_emb[-1].unsqueeze(0)
                rel_emb  = (torch.stack(rel_emb)  if isinstance(rel_emb,  list) else rel_emb)
                rel_emb  = rel_emb[-1].unsqueeze(0)
            else:
                sub_idx = [subject] if isinstance(subject, (int, np.integer)) else list(subject)
                rel_idx = [relation] if isinstance(relation, (int, np.integer)) else list(relation)
                evolve_emb = self.model.get_parameter('dynamic_emb').data.unsqueeze(0).unsqueeze(0)
                node_emb   = self.model.get_parameter('dynamic_emb').data[sub_idx].unsqueeze(0)
                rel_emb    = self.model.get_parameter('emb_rel').data[rel_idx].unsqueeze(0)

        return evolve_emb, node_emb, rel_emb, [text_data['instruction']], [text_data['response']], inputs

    def collate_fn_test(self, batch):
        """Test collate function; like ``collate_fn`` but also returns the
        graph filter sets used for evaluation."""
        trips, texts, scores, filter_sets, head_images, tail_images = zip(*batch)
        current_head_image = head_images[0]
        current_tail_image = tail_images[0]
        current_instruction = texts[0]['instruction']

        images = [img for img in [current_head_image, current_tail_image] if img is not None]
        images_arg = images if images else None  # filter out None; processor rejects an empty list

        evolve_emb = node_emb = rel_emb = None
        START_TIME = 0

        if getattr(self.args, 'use_modality_graph', True):
            subject, relation, target, time = trips[0]
            if time != START_TIME:
                time_idx = time - START_TIME
                if time >= 32:
                    time_idx = time_idx - 1 
                history_len = min(time_idx, self.args.train_history_len)
                history_glists = [
                    build_sub_graph(self.num_nodes, self.num_rels, snap, trips, True, 'cpu')
                    for snap in self.train_list[time_idx - history_len : time_idx]
                ]
                evolve_emb, node_emb, rel_emb = load(
                    self.model, history_glists, trips, static_graph=None, args=self.args
                )
                evolve_emb = (torch.cat(evolve_emb, dim=0) if evolve_emb[0].dim() == 2
                            else torch.stack(evolve_emb)).unsqueeze(0)
                node_emb = torch.stack(node_emb) if isinstance(node_emb, list) else node_emb
                node_emb = node_emb[-1].unsqueeze(0) if node_emb.dim() > 1 else node_emb.unsqueeze(0)
                rel_emb  = torch.stack(rel_emb)  if isinstance(rel_emb,  list) else rel_emb
                rel_emb  = rel_emb[-1].unsqueeze(0) if rel_emb.dim() > 1 else rel_emb.unsqueeze(0)
            else:
                evolve_emb = self.model.get_parameter('dynamic_emb').data.unsqueeze(0)
                subject_idx = int(subject) if isinstance(subject, (int, np.integer)) else list(subject)
                node_emb = self.model.get_parameter('dynamic_emb').data[
                    [subject_idx] if isinstance(subject_idx, int) else subject_idx].unsqueeze(0)
                rel_emb    = self.model.get_parameter('emb_rel').data[
                    [relation] if isinstance(relation, int) else list(relation)].unsqueeze(0)

        content_list = []
        if current_head_image is not None:
            content_list.append({"type": "image", "image": current_head_image})
        if current_tail_image is not None:
            content_list.append({"type": "image", "image": current_tail_image})
        content_list.append({"type": "text", "text": current_instruction})

        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content_list}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images_arg,       
            padding="max_length", max_length=1024,
            truncation=True, return_tensors="pt"
        )

        return evolve_emb, node_emb, rel_emb, [current_instruction], [texts[0]['response']], None, filter_sets, inputs

    def select(self, data_range):
        """Slice the dataset to the given inclusive range ``[start, end]``."""
        if data_range is not None:
            start, end = data_range[0], data_range[-1]
            self.current_data = self.current_data[start:end]
            self.current_text = self.current_text[start:end]
            if self.mode == "test":
                self.pretrained_test_scores = self.pretrained_test_scores[start:end]
                self.test_tail_sets = self.test_tail_sets[start:end]