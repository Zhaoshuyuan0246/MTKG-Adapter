import argparse
import torch

def str2bool(v):
    """Parse a string into a boolean, raising on invalid input."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def Args():
    """Build and return the command-line argument parser for training."""
    parser = argparse.ArgumentParser(description='Training')
    
    parser.add_argument('--base_model', type=str, required=False, default='MLLM/Qwen3-VL-8B-Instruct', help="base model path")
    parser.add_argument('--peft_model', type=str, default='checkpoint/Qwen3_8B/', help="fine-tuned model path")
    parser.add_argument('--dataset', type=str, required=False, default="GDELT", help="dataset name")
    parser.add_argument('--max_steps', type=int, default=None, help="max steps")
    parser.add_argument('--num_epochs', type=int, default=1, help="max training epochs")
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8, help="random seed")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--accumulation_steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="weight decay")
    parser.add_argument('--lr', type=float, default=2e-5 , help="learning rate")
    parser.add_argument('--warmup_steps', type=int, default=20, help="learning-rate warmup steps")
    parser.add_argument(
        '--schedule_name', 
        type=str, 
        default='constant_with_warmup', 
        choices=['constant', 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup'],
        help="learning-rate schedule"
    )
    
    parser.add_argument('--target_modules', nargs='+', type=str, default=None)
    parser.add_argument('--target_modules_lora', nargs='+', type=str, default=None)
    parser.add_argument('--polynomial_power', type=int, default=2) 
    parser.add_argument('--num_cycles', type=int, default=5) 

    parser.add_argument('--label_smoothing', type=float, default=0) # TODO
    parser.add_argument('--max_grad_norm', type=float, default=1.0) # TODO

    parser.add_argument('--peft_type', type=str, default="TKG_Adapter")
    parser.add_argument('--hidden_size', type=int, default=2560)
    parser.add_argument('--graph_hidden_size', type=int, default=400)
    parser.add_argument('--image_hidden_size', type=int, default=1024)
    parser.add_argument('--model_type', type=str, default='qwen3_vl')
    parser.add_argument('--torch_dtype', type=torch.dtype, default=torch.bfloat16)
    parser.add_argument('--max_llm_layer', type=int, default=0)

    # TKG_Adapter
    parser.add_argument('--kg_adapter_dec_range', type=list[int], default=[0, 36])
    parser.add_argument('--kg_adapter_node_emb_size', type=int, default=100)
    parser.add_argument('--kg_adapter_hidden_size', type=int, default=128)
    parser.add_argument('--kg_adapter_info_merge', type=str, default='gate')
    parser.add_argument('--align_mask', type=bool, default=False)
    parser.add_argument('--use_gnn', type=bool, default=False)
    parser.add_argument('--enc_interact_with_LLM', type=bool, default=False)

    parser.add_argument('--use_node_emb', type=bool, default=False)
    parser.add_argument('--use_edge_emb', type=bool, default=False)
    parser.add_argument('--mix_emb', type=bool, default=False)
    parser.add_argument('--use_trips', type=bool, default=False)
    parser.add_argument('--use_SRGAT', type=bool, default=False)
    parser.add_argument('--output_sg', type=bool, default=False)
    parser.add_argument('--node_emb_path', type=str, default=None)
    parser.add_argument('--keep_ratio', type=float, default=0.5)
    parser.add_argument('--exp_set', type=str, default='loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb')

    parser.add_argument('--dev', type=bool, default=True)
    parser.add_argument('--fuse_rate', type=float, default=1.0)
    parser.add_argument('--scaling_rate', type=float, default=1.0)
    parser.add_argument('--add_lora', type=bool, default=False)#*****
    parser.add_argument('--train_lm_head', type=bool, default=True)
    parser.add_argument('--use_prefix', type=bool, default=True)
    # parser.add_argument('--use_kg_encoder', type=bool, default=False)
    parser.add_argument('--no_res', type=bool, default=False)
    parser.add_argument('--linear_scale', type=bool, default=True)
    parser.add_argument('--linear_emb', type=bool, default=True)
    parser.add_argument('--info_merge_pos', type=str, default='before')

    parser.add_argument('--add_adapter', type=bool, default=True) #*****
    parser.add_argument('--kg_adapter_cross_attention', type=bool, default=False)
    parser.add_argument('--kg_adapter_moe', type=bool, default=True)
    parser.add_argument('--text2graph', type=bool, default=False)
    parser.add_argument('--graph2text', type=bool, default=True)
    parser.add_argument('--num_experts', type=int, default=2)
    
    parser.add_argument('--last_token', type=bool, default=None)
    parser.add_argument('--eta_b', type=float, default=1.2)
    parser.add_argument('--ConvGraph', type=bool, default=False)
    
    # MoE
    parser.add_argument('--top_k_routing_strategy', action='store_true', default=True)
    parser.add_argument('--topk', type=int, default=1)# number of activated experts
    # parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--num_router_mlp_layers', type=int, default=1)
    parser.add_argument('--router_hidden_dim', type=int, default=32)
    # hmora loss
    parser.add_argument('--use_load_balancing_loss', action='store_true', default=False)
    parser.add_argument('--use_div_loss', action='store_true', default=False)
    parser.add_argument('--gamma_div_certain_t', type=float, default=0.5)
    parser.add_argument('--gamma_div_balance_t', type=float, default=1)
    parser.add_argument('--gamma_div_certain_s', type=float, default=0.5)
    parser.add_argument('--gamma_div_balance_s', type=float, default=1)
    parser.add_argument('--lambda_auxiliary', type=float, default=0.01)
    parser.add_argument('--lambda_lm', type=float, default=1.0)

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
    # parser.add_argument("--gpu", type=int, default=0, help="gpu id")

    parser.add_argument("--use_modality_image", type=str2bool, default=True)
    parser.add_argument("--use_modality_graph", type=str2bool, default=True)
    parser.add_argument("--use_text_moe",        type=str2bool, default=True)
    parser.add_argument("--use_image_moe",       type=str2bool, default=True)
    parser.add_argument("--use_graph_moe",       type=str2bool, default=True)
    parser.add_argument("--use_cross_attention", type=str2bool, default=True)
    parser.add_argument("--use_channel_attention",type=str2bool,default=False)
    parser.add_argument("--use_lora",            type=str2bool, default=False)


    # ==================== Layer-wise Adapter Insertion Gate ====================
    parser.add_argument("--target_active_layers", type=int, default=12)

    parser.add_argument("--kg_adapter_insert_init_prob", type=float, default=0.5)
    parser.add_argument("--kg_adapter_insert_init_noise_std", type=float, default=0.2)

    parser.add_argument("--kg_adapter_use_hard_topk", type=str2bool, default=True)

    parser.add_argument("--lambda_binary", type=float, default=1e-3)
    parser.add_argument("--lambda_margin", type=float, default=1e-2)
    parser.add_argument("--gate_margin", type=float, default=0.2)

    parser.add_argument("--sparse_warmup_steps", type=int, default=100)

    args = parser.parse_args()
    
    return args 

