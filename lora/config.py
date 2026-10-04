from dataclasses import dataclass, asdict, field
from typing import Dict, List
import torch
from transformers.modeling_utils import PreTrainedModel
import dataclasses
#from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import Qwen3VLConfig

VALID_TREND = ("EQUAL", "INCREASE", "DECREASE")
SUPPORTED_CAUSAL_MODELS = ('llama', 'bloom', 'qwen3', 'qwen2_5', 'qwen3_vl')
TARGET_MODULE_TYPE = {
    'qwen3_vl': {
        'q': ['q_proj'],
        'k': ['k_proj'],
        'v': ['v_proj'],
        'o': ['o_proj'],
        'wi': ['gate_proj', 'up_proj'],  # Qwen's MLP also uses SwiGLU with gate and up projections
        'wo': ['down_proj'],
        'atte': 'self_attn',
        'ffn': 'mlp',
        'embed': 'embed_tokens',
        'decoders': 'model.layers'       # Qwen's main layer container name is usually model.layers
    },
}

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    Attributes:
        target_modules: Names of modules to apply LoRA to.
        peft_type: PEFT method name (``"lora"``).
        hidden_size: Hidden size of the base model.
        model_type: Base model type (e.g. ``"qwen3_vl"``).
        torch_dtype: Torch dtype for LoRA parameters.
        dropout: Dropout probability.
        max_llm_layer: Maximum LLM layer index to apply LoRA (0 = all).
        target_modules_lora: LoRA-specific target modules.
        use_rs_scaling: Whether to use rank-stabilized scaling.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling alpha.
    """
    target_modules: List[str] = None
    peft_type: str = "lora"
    hidden_size: int = None
    model_type: str = None
    torch_dtype: torch.dtype = torch.float32
    dropout: float = 0.1
    max_llm_layer: int = 0

    # lora
    target_modules_lora: List[str] = None
    use_rs_scaling: bool = False
    lora_r: int = 8
    lora_alpha: int = 16

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoRAConfig":
        config = LoRAConfig(**config)
        return config
    
    # Convert the dataclass instance to a dict
    def export(self) -> Dict[str, any]:
        config = asdict(self)
        return config


@dataclass(init=False)
class TKGAdapterConfig(Qwen3VLConfig):
    """Configuration for the TKG-Adapter (extends ``Qwen3VLConfig``).

    Holds all LLM, LoRA, TKG-Adapter, KG node/edge, and GNN-related settings.
    The ``__init__`` method overwrites ``text_config``/``vision_config`` with the
    real values read from the loaded base model.
    """
    # LLM-related class attributes.
    target_modules: List[str] = None
    peft_type: str = "TKG_Adapter"
    hidden_size: int = 2560
    graph_hiden_size: int = 400
    model_type: str = None
    torch_dtype: torch.dtype = torch.float32
    hidden_act: str = None
    dropout: float = 0.1
    max_llm_layer: int = 0
    use_prefix: bool = True # whether to use prefix-tuning
    align_mask: bool = False # whether to use the alignment mask
    train_lm_head: bool = True # whether to train the LM head

    # lora
    add_lora: bool = False # whether to add LoRA
    target_modules_lora: List[str] = None
    use_rs_scaling: bool = False
    lora_r: int = 8
    lora_alpha: int = 16

    # TKGAdapter position
    kg_adapter_dec_range: list[int] = field(default_factory=lambda: [0, 36])    # decoder range, adjusted per model
    info_merge_pos: str = 'before' # before: merge_info -> SA; mid: SA->merge_info->FFN; after: FNN-> merge_info -> ...

    # TKGAdapter parameters
    kg_adapter_hidden_size: int = 128  # hidden dim of the adapter
    kg_adapter_intermediate_size: int = kg_adapter_hidden_size * 4
    enc_interact_with_LLM: bool = False
    
    # TKGAdapter text-related
    kg_adapter_info_merge: str = 'gate'    # fusion method (gating); choose from [gate, linear, sum]
    scaling_rate: float = 1.0  # scaling ratio for node-infused text; how much info to keep
    linear_scale: bool = True # whether to apply linear scaling to text

    # TKGAdapter KG-node related
    use_node_emb: bool = True # whether to use node embeddings
    mix_emb: bool = False # whether to encode nodes with LLMs
    node_emb_path: str = None # whether to use pretrained node embeddings
    kg_adapter_node_emb_size: int = 100  # entity node embedding size

    # TKGAdapter KG-edge related
    use_edge_emb: bool = True # whether to use edge embeddings

    # TKGAdapter KG node-and-edge related
    linear_emb: bool = True # related to node and edge embeddings

    # TKGAdapter KG-triple related
    use_trips: bool = True # whether to use KG triple info

    # TKGAdapter GNN related
    use_gnn: bool = True # whether to use GNN on the KG
    use_SRGAT: bool = True # whether to use SRGAT for KG message passing
    output_sg: bool = False # whether to output the subgraph
    keep_ratio: float = 0.5 # SRGAT keep ratio
    dev: bool = False

    # TKGAdapter text-node fusion
    kg_adapter_cross_attention: bool = False # whether to use cross-attention

    # residual connection related
    no_res: bool = False # whether to disable the residual connection
    fuse_rate: float = 1.0 # residual connection ratio

    exp_set: str = 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb'
    
    # Qwen2.5-specific config
    pad_token_id: int = 151643  # Qwen2.5 pad_token_id
    bos_token_id: int = 151643  # Qwen2.5 bos_token_id
    eos_token_id: int = 151645  # Qwen2.5 eos_token_id
    padding_side: str = 'right'
    
    vocab_size: int = 152064  # Qwen2.5 vocab size
    
    # MoE-related config
    graph2text: bool = True
    text2graph: bool = True
    kg_adapter_moe: bool = True
    num_experts: int = 3
    topk: int = 2
    
    # -------------------- Vision-modality-specific config ------------------------
    use_vision: bool = True  # whether to use the vision modality
    vision_embed_dim: int = 1024  # vision embedding dim
    vision_feature_select_strategy: str = 'default'  # feature selection strategy
    vision_num_features: int = 256  # number of vision features
    
    # vision adapter config
    use_visual_adapter: bool = False  # whether to add an adapter to the vision module
    visual_adapter_hidden_size: int = 64  # vision adapter hidden size

    def __init__(self, **kwargs):
        # Save the real passed-in values first
        _hidden_size = kwargs.get('hidden_size', 2560)
        _text_config = kwargs.get('text_config', None)
        _vision_config = kwargs.get('vision_config', None)
        
        super().__init__(**kwargs)  # the parent class creates a default text_config
        
        # After the parent class finishes, overwrite with the correct values
        for key, value in kwargs.items():
            setattr(self, key, value)  # force-overwrite (removed the if not hasattr guard)
        
        # If a real text_config was passed in, force-replace the parent default
        if _text_config is not None:
            self.text_config = _text_config
        if _vision_config is not None:
            self.vision_config = _vision_config

    @staticmethod
    def from_config(config: Dict[str, any]) -> "TKGAdapterConfig":
        adapter_config = TKGAdapterConfig(**config) 
        return adapter_config

    def export(self) -> Dict[str, any]:
        config = asdict(self)
        return config

    
@dataclass
class PretrainedConfig(PreTrainedModel):
    """Legacy pretrained config class for the TKG-Adapter model.

    Exposes the same set of attributes as ``TKGAdapterConfig`` but implemented
    as a plain class for backward compatibility.
    """
    def __init__(self, 
                 config = None, 
                 target_modules=None, 
                 target_modules_lora=None, 
                 dropout=0.1, 
                 lora_r=8, 
                 lora_alpha=32,
                 hidden_size=2560, 
                 kg_adapter_hidden_size=128, 
                 kg_adapter_node_emb_size=100, 
                 kg_adapter_info_merge="gate",
                 use_gnn=True,
                 **kwargs):
        super().__init__(config, **kwargs)
        
        # LLM-related
        self.target_modules: List[str] = target_modules
        self.peft_type: str = "TKG_Adapter"
        self.hidden_size: int = hidden_size
        self.model_type: str = None
        self.torch_dtype: torch.dtype = torch.float32
        self.dropout: float = dropout
        self.max_llm_layer: int = 0
        self.train_lm_head: bool = True # whether to train the LM head
        self.use_prefix: bool = True # whether to use prefix-tuning
        self.align_mask: bool = True # whether to use the alignment mask
        self.padding_side: str = "right"

        # lora
        self.add_lora: bool = False # whether to add LoRA
        self.target_modules_lora: List[str] = target_modules_lora
        self.use_rs_scaling: bool = False
        self.lora_r: int = lora_r
        self.lora_alpha: int = lora_alpha


        # TKGAdapter position
        self.kg_adapter_enc_range = [0, 0]   # which layers host the encoder
        self.kg_adapter_dec_range = [0, 32]   # which layers host the decoder
        self.info_merge_pos = 'before' # before: merge_info -> SA; mid: SA->merge_info->FFN; after: FNN-> merge_info -> ...

        # TKGAdapter parameters
        self.kg_adapter_hidden_size: int = 128  # tkgadapter internal dim
        self.kg_adapter_intermediate_size: int = kg_adapter_hidden_size * 4 # dim after the FFN
        self.enc_interact_with_LLM: bool = True # encoder-related; currently unused
        
        # TKGAdapter text-related
        self.kg_adapter_info_merge: str = kg_adapter_info_merge  # fusion method (gating); choose from [gate, linear, sum]
        self.scaling_rate: float = 1.0 # scaling ratio for node-infused text
        self.linear_scale: bool = True # whether to apply linear scaling to text

        # TKGAdapter KG-node related
        self.use_node_emb: bool = False # whether to use node embeddings
        self.mix_emb: bool = True # whether to encode nodes with LLMs
        self.node_emb_path: str = None # whether to use pretrained node embeddings
        self.kg_adapter_node_emb_size: int = kg_adapter_node_emb_size  # node embedding

        self.linear_emb: bool = True # related to node and edge embeddings
        # TKGAdapter KG-edge related
        self.use_edge_emb: bool = True # whether to use edge embeddings
        self.num_relations: int = 500  # number of relations

        # TKGAdapter KG-triple related
        self.use_trips: bool = True # whether to use KG triple info

        # TKGAdapter GNN related
        self.use_gnn: bool = True # whether to use GNN on the KG
        self.use_SRGAT: bool = True # whether to use SRGAT for KG message passing
        self.output_sg: bool = False # whether to output the subgraph
        self.keep_ratio: float = 0.5 # SRGAT keep ratio
        self.dev: bool = False

        # TKGAdapter text-node fusion
        self.kg_adapter_cross_attention: bool = False # whether to use cross-attention

        # residual connection related
        self.no_res: bool = False # whether to disable the residual connection
        self.fuse_rate: float = 1.0 # residual connection ratio




    @staticmethod
    def from_config(config: Dict[str, any]) -> "PretrainedConfig":
        config = PretrainedConfig(**config)
        return config

    def export(self) -> Dict[str, any]:
        config = asdict(self)
        return config
