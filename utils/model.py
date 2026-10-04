import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from lora.model import get_peft_model, get_model, get_tkg_adapter_model
from lora.config import LoRAConfig, TKGAdapterConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers import  Qwen3VLConfig 

DEFAULT_TARGET_MODULES = {
      'qwen2': ['q_proj', 'k_proj', 'v_proj'],
      'qwen3_vl': ['q_proj', 'k_proj', 'v_proj'],
    # 'Qwen3vl': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj','vision_model.encoder.layers.*.self_attn.*_proj', 'visual_projection'],
}

def TKG_Adapter(args, tokenizer, processor, accelerator):
    """Build the Qwen3-VL model equipped with the TKG-Adapter (and optionally LoRA).

    Args:
        args: Argument namespace (base model, modality flags, adapter hyperparams).
        tokenizer: Tokenizer for the base model.
        processor: Qwen3-VL processor (unused here but kept for API symmetry).
        accelerator: Hugging Face Accelerator instance.

    Returns:
        The assembled model with adapter/LoRA/router parameters left trainable.
    """
    # ==================== Load the base pre-trained model ====================
    current_device = accelerator.device 

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=args.torch_dtype,
        device_map={"": current_device},
        attn_implementation="sdpa",
    )  

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   
    
    model_config = model.config

    # ==================== Read the real nested config up front (baseline for everything below) ====================
    _tc = model.config.text_config    # Qwen3VLTextConfig, hidden_size=2560
    _vc = model.config.vision_config  # Qwen3VLVisionConfig, hidden_size=1024

    # ==================== Tokenizer setup ====================
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side == 'right':
        tokenizer.padding_side = 'left'

    # ==================== Target module config ====================
    if args.target_modules is None:
        args.target_modules = DEFAULT_TARGET_MODULES[model_config.model_type]
          
    # ==================== LoRA fine-tuning config ====================
    if args.add_lora == True: 
        peft_config = LoRAConfig(
            target_modules=args.target_modules,      
            target_modules_lora=getattr(args, 'target_modules_lora', None), 
            dropout=args.dropout,                    
            lora_r=args.lora_r,                      
            lora_alpha=args.lora_alpha               
        )
        peft_config.torch_dtype = torch.float16      
        peft_config.padding_side = tokenizer.padding_side
        model = get_peft_model(model, peft_config)

    # ==================== TKGAdapter config ====================
    if args.add_adapter == True:  
        tkg_config = TKGAdapterConfig(
            target_modules=args.target_modules,              
            peft_type=args.peft_type,                        
            model_type='qwen3_vl',                         
            torch_dtype=args.torch_dtype,                    
            # All read from the real text_config / vision_config, not args or getattr(model_config)
            hidden_size=_tc.hidden_size,                     # 2560
            image_hidden_size=_vc.hidden_size,               # 1024
            num_hidden_layers=_tc.num_hidden_layers,         # 36 (critical fix)
            intermediate_size=_tc.intermediate_size,
            num_attention_heads=_tc.num_attention_heads,
            num_key_value_heads=_tc.num_key_value_heads,
            rms_norm_eps=_tc.rms_norm_eps,
            hidden_act=_tc.hidden_act,
            vocab_size=_tc.vocab_size,
            max_position_embeddings=_tc.max_position_embeddings,
            rope_theta=_tc.rope_theta,
            tie_word_embeddings=_tc.tie_word_embeddings,
            # The remaining args-derived parameters stay unchanged
            graph_hidden_size=args.graph_hidden_size,        
            dropout=args.dropout,                            
            attention_dropout=args.dropout,                  
            use_vision=True,                                 
            vision_embed_dim=_vc.hidden_size,
            vision_feature_select_strategy=getattr(model_config, 'vision_feature_select_strategy', 'default'), 
            vision_num_features=getattr(model_config, 'vision_num_features', 256), 
            image_token_id=getattr(model_config, 'image_token_id', 151857), 
            train_lm_head=args.train_lm_head,                
            use_prefix=args.use_prefix,                      
            align_mask=args.align_mask,                      
            add_lora=args.add_lora,                          
            lora_r=args.lora_r,                              
            lora_alpha=args.lora_alpha,                      
            mix_emb=args.mix_emb,                            
            kg_adapter_dec_range=args.kg_adapter_dec_range,  
            info_merge_pos=args.info_merge_pos,              
            kg_adapter_hidden_size=args.kg_adapter_hidden_size, 
            kg_adapter_intermediate_size=args.kg_adapter_hidden_size * 4, 
            kg_adapter_info_merge=args.kg_adapter_info_merge, 
            scaling_rate=args.scaling_rate,                  
            linear_scale=args.linear_scale,                  
            use_node_emb=args.use_node_emb,                  
            use_edge_emb=args.use_edge_emb,                  
            node_emb_path=args.node_emb_path,                
            kg_adapter_node_emb_size=args.kg_adapter_node_emb_size, 
            linear_emb=args.linear_emb,                      
            use_trips=args.use_trips,                        
            use_gnn=args.use_gnn,                            
            use_SRGAT=args.use_SRGAT,                        
            output_sg=args.output_sg,                        
            keep_ratio=args.keep_ratio,                      
            dev=args.dev,                                    
            kg_adapter_cross_attention=args.kg_adapter_cross_attention, 
            no_res=args.no_res,                              
            fuse_rate=args.fuse_rate,                        
            exp_set=args.exp_set,                            
            kg_adapter_moe=args.kg_adapter_moe,              
            text2graph=args.text2graph,                      
            graph2text=args.graph2text,                      
            num_experts=args.num_experts,                    
            topk=args.topk,                                  
            last_token=args.last_token,                      
            top_k_routing_strategy=args.top_k_routing_strategy, 
            num_router_mlp_layers=args.num_router_mlp_layers,   
            router_hidden_dim=args.router_hidden_dim,          
            use_load_balancing_loss=args.use_load_balancing_loss, 
            use_div_loss=args.use_div_loss,                  
            gamma_div_certain_t=args.gamma_div_certain_t,    
            gamma_div_balance_t=args.gamma_div_balance_t,    
            gamma_div_certain_s=args.gamma_div_certain_s,    
            gamma_div_balance_s=args.gamma_div_balance_s,    
            lambda_lm=args.lambda_lm,                        
            lambda_auxiliary=args.lambda_auxiliary,          
            pad_token_id=getattr(model_config, 'pad_token_id', 151643),
            bos_token_id=getattr(model_config, 'bos_token_id', 151643),
            eos_token_id=getattr(model_config, 'eos_token_id', 151645),
            use_visual_adapter=getattr(args, 'use_visual_adapter', False), 
            visual_adapter_hidden_size=getattr(args, 'visual_adapter_hidden_size', 64), 
            attn_implementation="sdpa",
            use_modality_image=getattr(args, 'use_modality_image', False),
            use_modality_graph=getattr(args, 'use_modality_graph', True),
            use_text_moe=getattr(args, 'use_text_moe', True),
            use_image_moe=getattr(args, 'use_image_moe', False),
            use_graph_moe=getattr(args, 'use_graph_moe', False),
            use_cross_attention=getattr(args, 'use_cross_attention', False),
            use_channel_attention=getattr(args, 'use_channel_attention', False),
            use_pure_lora=getattr(args, 'use_lora', False),
        )

        # Immediately replace the parent class's auto-generated (wrong) text_config and vision_config
        tkg_config.text_config   = _tc
        tkg_config.vision_config = _vc

        # setattr loop: write tkg_config fields into model_config
        for key, value in vars(tkg_config).items():
            setattr(model_config, key, value)

        # Force-restore the real values after the loop (prevent tkg_config default pollution)
        model_config.text_config      = _tc
        model_config.vision_config    = _vc
        model_config.hidden_size      = _tc.hidden_size         # 2560
        model_config.num_hidden_layers = _tc.num_hidden_layers  # 36
        model_config.intermediate_size = _tc.intermediate_size
        model_config.num_attention_heads = _tc.num_attention_heads
        model_config.num_key_value_heads = _tc.num_key_value_heads
        model_config.rms_norm_eps      = _tc.rms_norm_eps
        model_config.hidden_act        = _tc.hidden_act
        model_config.vocab_size        = _tc.vocab_size
        model_config.image_hidden_size = _vc.hidden_size         # 1024

        if getattr(model_config, 'use_node_emb', False):
            model_config.node_num = 10778  
        if getattr(model_config, 'use_edge_emb', False):
            model_config.num_relations = 24  

        model_config.torch_dtype  = torch.float16     
        model_config.padding_side = tokenizer.padding_side  
        model_config.pad_token_id = tokenizer.pad_token_id  
        model_config.kg_adapter_intermediate_size = model_config.kg_adapter_hidden_size * 4  

        if accelerator is not None:
            model_config.device = accelerator.device 
        else:
            model_config.device = args.device            

        model = get_model(model, model_config)        
        model = get_tkg_adapter_model(model, model_config)  

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model.model, "enable_input_require_grads"):
        model.model.enable_input_require_grads()

    vanilla_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if accelerator is None or accelerator.is_local_main_process:
        print(f'模型总参数量: {vanilla_params}'
              f' | 可训练参数量: {trainable_params}'
              f' | 可训练参数比例: {trainable_params / vanilla_params:.4f}')

    return model