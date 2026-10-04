import os
# fix the duplicated OpenMP library loading issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# import the RGCN model
from .rgcn.rrgcn import RecurrentRGCN
import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_model(args, data, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    """Create and load the pre-trained RecurrentRGCN graph model.

    Args:
        args: Argument namespace with RE-GCN hyperparameters and ``peft_path``.
        data: Dataset object providing ``num_nodes`` and ``num_rels``.
        n_hidden, n_layers, dropout, n_bases: Optional overrides for the model.

    Returns:
        The loaded RecurrentRGCN model moved to CPU (for DDP safety).
    """
    
    # ============ Parameter config ============
    if n_hidden: args.n_hidden = n_hidden
    if n_layers: args.n_layers = n_layers
    if dropout: args.dropout = dropout
    if n_bases: args.n_bases = n_bases
    
    num_nodes = data.num_nodes          
    num_rels = data.num_rels            
    num_static_rels = 0                 
    num_words = 0                       
    
    # get the device; if args has no device, fall back to the current default device
    current_device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # dynamically decide whether to use CUDA
    is_cuda = current_device.type == 'cuda' if hasattr(current_device, 'type') else 'cuda' in str(current_device)

    # ============ Model creation ============
    model = RecurrentRGCN(
        args.decoder,                    
        args.encoder,                    
        num_nodes,                       
        num_rels,                        
        num_static_rels,                 
        num_words,                       
        args.n_hidden,                   
        args.opn,                        
        sequence_len=args.train_history_len,  
        num_bases=args.n_bases,          
        num_basis=args.n_basis,          
        num_hidden_layers=args.n_layers, 
        dropout=args.dropout,            
        input_dropout=args.input_dropout,    
        hidden_dropout=args.hidden_dropout,  
        feat_dropout=args.feat_dropout,      
        self_loop=args.self_loop,        
        skip_connect=args.skip_connect,  
        layer_norm=args.layer_norm,      
        aggregation=args.aggregation,    
        weight=args.weight,              
        discount=args.discount,          
        angle=args.angle,                
        use_static=args.add_static_graph,        
        entity_prediction=args.entity_prediction,    
        relation_prediction=args.relation_prediction, 
        gpu=args.device.index if hasattr(args.device, 'index') and args.device.index is not None else 0, # safely get the GPU id
        analysis=args.run_analysis,      
        use_cuda=is_cuda # set dynamically
    )
    
    # ============ Load pretrained weights ============
    # map_location ensures the model is loaded onto the right device (critical for DDP)
    checkpoint = torch.load(args.peft_path, map_location=current_device)
    model.load_state_dict(checkpoint['state_dict'])
    # model.to(current_device)
    model.to('cpu') 
    
    return model

def load(model, history_glists, trips, static_graph, args):
    """Extract temporal graph embeddings for a batch of triples.

    Args:
        model: RecurrentRGCN model (already loaded).
        history_glists: Historical subgraph list.
        trips: Triple tensor (subject, relation, object, time).
        static_graph: Unused placeholder (kept for API compatibility).
        args: Argument namespace with ``use_modality_graph``.

    Returns:
        Tuple of (evolve_embs, node_embs, rel_embs), each a list per triple.
    """
    if not getattr(args, 'use_modality_graph', True):
        return None, None, None

    model.eval()
    
    # collate_fn runs in a CPU worker; inference always uses cpu
    # the result is moved to GPU later in train.py via .to(accelerator.device)
    infer_device = torch.device('cpu')

    evolve_embs = []
    node_embs = []
    rel_embs = []

    for trip in trips:
        trip_input = torch.LongTensor(trip).to(infer_device)  # cpu
        evolve_emb, node_emb, rel_emb = process(
            model, history_glists, trip_input, args, infer_device  # pass infer_device
        )
        evolve_embs.append(evolve_emb)
        node_embs.append(node_emb)
        rel_embs.append(rel_emb)

    return evolve_embs, node_embs, rel_embs


def process(model, history_glists, trip_input, args, infer_device=None):
    """Run a single inference step and return subject/relation embeddings.

    Args:
        model: RecurrentRGCN model.
        history_glists: Historical subgraph list.
        trip_input: Single triple tensor (subject, relation, object, time).
        args: Argument namespace with ``layer_norm``.
        infer_device: Device to run inference on (defaults to CPU).

    Returns:
        Tuple of (evolve_emb, node_emb, rel_emb).
    """
    if infer_device is None:
        infer_device = torch.device('cpu')

    # temporarily move the model to cpu before inference to avoid device conflict with cpu graph data
    model_device = next(model.parameters()).device
    if model_device != infer_device:
        model.to(infer_device)

    is_cuda = infer_device.type == 'cuda'

    with torch.no_grad():
        evolve_embs, _, rel_emb, _, _ = model(
            history_glists,
            static_graph=None,
            use_cuda=is_cuda
        )
        node_emb = F.normalize(evolve_embs[-1]) if args.layer_norm else evolve_embs[-1]

    subject, relation, object_, time = trip_input

    evolve_emb = node_emb
    node_emb   = node_emb[subject]
    rel_emb    = rel_emb[relation]

    return evolve_emb, node_emb, rel_emb