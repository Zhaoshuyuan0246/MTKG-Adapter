import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from transformers import  Qwen3VLConfig

# ==========================================
# TokenRouter, RouterManager, LowRankAdapterRouter
# ==========================================

class SharedLoRAExperts(nn.Module):
    """LoRA-style experts sharing a single down-projection.

    All experts share one down-projection followed by independent up-projections;
    the routing weights blend the expert outputs.
    """

    def __init__(self, hidden_size, adapter_hidden_size, num_experts, dropout_prob=0.1):
        super().__init__()
        
        # 0. Add the standard LoRA dropout
        self.lora_dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0. else nn.Identity()
        
        # 1. Shared down-projection (no bias)
        self.down_proj = nn.Linear(hidden_size, adapter_hidden_size, bias=False)
        # activation layer
        self.activation = nn.ReLU()
        
        # 2. Independent up-projections (no bias)
        self.up_projs = nn.ModuleList([
            nn.Linear(adapter_hidden_size, hidden_size, bias=False)
            for _ in range(num_experts)
        ])
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down_proj.weight)
        for up_proj in self.up_projs:
            nn.init.zeros_(up_proj.weight)

    def forward(self, x: torch.Tensor, routing_weight: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        
        # 0. Add noise during training to prevent overfitting; transparent at eval
        x = self.lora_dropout(x)
        
        # 1. All tokens share one down-projection
        down_x = self.down_proj(x)  # -> [B, S, r]

        down_x = self.activation(down_x)
        
        # 2. Each expert up-projects separately
        adapter_outputs = torch.stack(
            [up_proj(down_x) for up_proj in self.up_projs], 
            dim=-2
        )  # -> [B, S, num_experts, H]
        
        # 3. Routing-weighted fusion
        routing_weight_ = routing_weight.unsqueeze(-1)  
        out = (routing_weight_ * adapter_outputs).sum(dim=-2) 
        
        return out

class TokenRouter(nn.Module):
    """Token-level router that dynamically selects experts for each token.

    Supports Top-K routing and load-balancing/confidence regularization losses.
    """
    def __init__(self, num_experts: int, 
                 input_dim: int, 
                 layer_id: int, 
                 dropout_prob: float = 0.1,
                 top_k: int = 2, 
                 gamma_div_balance: float = 0.8, 
                 gamma_div_certain: float = 0.2,
                 router_hidden_dim: int = 128,
                 num_router_mlp_layers: int = 1,
                 top_k_routing_strategy: bool = True, 
                 torch_dtype = torch.float32, 
                 tag: Optional[str] = None):
        super().__init__()
        self.num_experts = num_experts
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.torch_dtype = torch_dtype
        self.tag = tag

        # Dropout and routing strategy
        self.dropout_prop = dropout_prob
        self.top_k_routing_strategy = top_k_routing_strategy
        self.top_k = top_k

        # Regularization parameters
        self.gamma_div_balance = gamma_div_balance
        self.gamma_div_certain = gamma_div_certain
        
        if num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                nn.Dropout(self.dropout_prop),
                nn.Linear(input_dim, self.num_experts, dtype=torch_dtype)
            )
        else:
            layers = [
                nn.Dropout(self.dropout_prop),
                nn.Linear(input_dim, router_hidden_dim, dtype=torch_dtype),
                nn.ReLU()
            ]
            for _ in range(num_router_mlp_layers - 2):
                layers.extend([
                    nn.Dropout(self.dropout_prop),
                    nn.Linear(router_hidden_dim, router_hidden_dim, dtype=torch_dtype),
                    nn.ReLU()
                ])
            layers.extend([
                nn.Dropout(self.dropout_prop),
                nn.Linear(router_hidden_dim, self.num_experts, dtype=torch_dtype)
            ])
            self.mlp = nn.Sequential(*layers)
        
        self.routing_weight: Optional[torch.Tensor] = None
        self.token_routing_weight: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor):
        routing_weight = F.softmax(self.mlp(hidden_states), dim=-1)
        self.token_routing_weight = routing_weight
        self.routing_weight = routing_weight
        
        if self.top_k_routing_strategy:
            top_k_values, top_k_indices = torch.topk(self.routing_weight, self.top_k, dim=-1)
            routing_weight = torch.zeros_like(self.routing_weight)
            routing_weight.scatter_(-1, top_k_indices, top_k_values)
            routing_weight = routing_weight / (routing_weight.sum(dim=-1, keepdim=True) + 1e-10)
            self.routing_weight = routing_weight
            
        return self.routing_weight


    def divergence_loss(self, attention_mask):
        """Compute the generalized Jensen-Shannon divergence loss."""
        token_routing_weight = self.token_routing_weight
        # changed to unsqueeze(-1) to keep 3D
        mask = attention_mask.unsqueeze(-1).to(token_routing_weight.dtype)  # [B, S, 1]
        token_routing_weight = token_routing_weight * mask                   # ✅ [B, S, num_experts]
        
        max_entropy = torch.log(torch.tensor(
            self.num_experts, dtype=token_routing_weight.dtype, device=token_routing_weight.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p
        
        num_token = torch.sum(mask)
        m = torch.sum(token_routing_weight.view(-1, self.num_experts), dim=0) / num_token
        entropy_m = -torch.sum(m * torch.log(m + 1e-9), dim=-1)
        entropy_m = torch.clamp(entropy_m, max=max_entropy_m)
        
        entropy_p = -torch.sum(token_routing_weight * torch.log(token_routing_weight + 1e-9), dim=-1)
        entropy_p = torch.clamp(entropy_p, min=min_entropy_p) * mask.squeeze(-1)
        entropy_p = torch.sum(entropy_p) / num_token
        
        loss = torch.relu(max_div - (entropy_m - entropy_p)) / max_entropy
        return loss

    
    def load_balancing_loss(self, attention_mask):
        """Compute the load-balancing loss."""
        routing_weight = self.token_routing_weight
        routing_weight_ = self.routing_weight
        mask = attention_mask.view(-1, 1).to(routing_weight.dtype)       
        num_token = mask.sum()
        
        routing_weight  = routing_weight.view(-1, self.num_experts) * mask    # ✅ [B*S, num_experts]
        routing_weight_ = routing_weight_.view(-1, self.num_experts)          # unified reshape

        freq = torch.sum(torch.sign(routing_weight_), dim=0) / (num_token * self.top_k)
        prop = torch.sum(routing_weight, dim=0) / num_token
        
        loss = torch.sum(prop * freq) * self.num_experts
        return loss.unsqueeze(0)

    def get_routing_weight(self):
        """Return the routing weights."""
        return self.routing_weight

    def clear(self):
        """Clear the cached weights."""
        self.routing_weight = None
        self.token_routing_weight = None

class RouterManager(nn.Module):
    """Manages a collection of ``TokenRouter`` instances and their auxiliary losses."""

    def __init__(self, config, token_routers):
        super().__init__()
        self.token_routers = token_routers
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.use_load_balancing_loss = getattr(config, 'use_load_balancing_loss', False)
        self.use_div_loss = getattr(config, 'use_div_loss', False)
        self.lambda_auxiliary = getattr(config, 'lambda_auxiliary', 0.01)
        self.lambda_lm = getattr(config, 'lambda_lm', 1.0)

    def clear(self):
        for router in self.token_routers:
            router.clear()


    def backward_auxiliary_loss_for_seq_router(self, reduce='sum'):
        if self.use_load_balancing_loss:
            return 0

        auxiliary_loss = []
        if len(auxiliary_loss) == 0:
            return 0

        loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == 'sum':
            loss = torch.sum(loss)
        elif reduce == 'mean':
            loss = torch.mean(loss)
        else:
            raise ValueError(f'reduce must be sum or mean, but got {reduce}')
        loss = loss * self.lambda_auxiliary
        loss.backward()
        return loss.item()

    def get_auxiliary_loss(self, loss, attention_mask, reduce='sum'):
        auxiliary_loss = []
        for router in self.token_routers:
            if self.use_load_balancing_loss:
                auxiliary_loss.append(router.load_balancing_loss(attention_mask))
            elif self.use_div_loss:
                auxiliary_loss.append(router.divergence_loss(attention_mask))
            else:
                break
        if len(auxiliary_loss) == 0:
            return loss
        auxiliary_loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == 'sum':
            auxiliary_loss = torch.sum(auxiliary_loss)
        elif reduce == 'mean':
            auxiliary_loss = torch.mean(auxiliary_loss)
        else:
            raise ValueError(f'reduce must be sum or mean, but got {reduce}')
        loss = self.lambda_lm * loss + self.lambda_auxiliary * auxiliary_loss
        return loss


class KgAdapterLayerInsertManager(nn.Module):
    """Global Top-K gate manager deciding which layers insert a KG adapter.

    Learns one gate logit per layer and selects the top-K layers with the
    highest probabilities, so adapters are inserted adaptively.
    """

    def __init__(
        self,
        num_layers: int,
        target_active_layers: int = 8,
        candidate_layer_ids=None,
        init_prob: float = 0.5,
        init_noise_std: float = 0.2,
        use_hard_topk: bool = True,
    ):
        super().__init__()

        assert 0.0 < init_prob < 1.0

        base_logit = torch.log(
            torch.tensor(init_prob) / (1.0 - torch.tensor(init_prob))
        )

        # Adding noise is important; otherwise all layers are fully symmetric and learn identically
        gate_logits = base_logit + init_noise_std * torch.randn(num_layers)
        self.gate_logits = nn.Parameter(gate_logits)

        self.num_layers = num_layers
        self.target_active_layers = target_active_layers
        self.use_hard_topk = use_hard_topk

        if candidate_layer_ids is None:
            candidate_layer_ids = list(range(num_layers))

        candidate_mask = torch.zeros(num_layers, dtype=torch.bool)
        candidate_mask[candidate_layer_ids] = True
        self.register_buffer("candidate_mask", candidate_mask)

    def gate_probs(self):
        return torch.sigmoid(self.gate_logits)

    def topk_gates(self):
        probs = self.gate_probs()

        # Non-candidate layers do not participate in Top-K
        masked_probs = probs.masked_fill(~self.candidate_mask, -1.0)

        k = min(self.target_active_layers, int(self.candidate_mask.sum().item()))
        topk_idx = torch.topk(masked_probs, k=k).indices

        hard = torch.zeros_like(probs)
        hard[topk_idx] = 1.0

        if self.training:
            # STE: forward uses hard top-k, backward follows the soft probability
            gates = hard.detach() - probs.detach() + probs
        else:
            gates = hard

        return gates

    def gate_for_layer(self, layer_id: int):
        if self.use_hard_topk:
            return self.topk_gates()[layer_id]
        else:
            return self.gate_probs()[layer_id]

    def regularization_loss(
        self,
        lambda_binary: float = 1e-3,
        lambda_margin: float = 1e-2,
        margin: float = 0.2,
    ):
        probs = self.gate_probs()
        candidate_probs = probs[self.candidate_mask]

        # 1. Push gates toward 0 or 1
        binary_loss = (candidate_probs * (1.0 - candidate_probs)).mean()

        # 2. Widen the gap between the K-th and (K+1)-th layers
        sorted_probs = torch.sort(candidate_probs, descending=True).values
        k = min(self.target_active_layers, sorted_probs.numel())

        if k < sorted_probs.numel():
            topk_min = sorted_probs[k - 1]
            rest_max = sorted_probs[k]
            margin_loss = torch.relu(margin - (topk_min - rest_max))
        else:
            margin_loss = torch.tensor(
                0.0,
                device=probs.device,
                dtype=probs.dtype
            )

        return lambda_binary * binary_loss + lambda_margin * margin_loss

    def gate_stats(self):
        with torch.no_grad():
            probs = self.gate_probs()
            hard = torch.zeros_like(probs)

            masked_probs = probs.masked_fill(~self.candidate_mask, -1.0)
            k = min(self.target_active_layers, int(self.candidate_mask.sum().item()))
            topk_idx = torch.topk(masked_probs, k=k).indices
            hard[topk_idx] = 1.0

        return probs.detach(), hard.detach()




class CrossAttnAdapterExpert(nn.Module):
    """Cross-attention expert: graph-aware cross-attention + FFN with residual."""

    def __init__(self, KgAdapterCrossAttention: nn.Module, Norm_Module: nn.Module, KgAdapterMLP: nn.Module, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.kg_adapter_t2n_cross_attn = KgAdapterCrossAttention(config=config)
        self.kg_adapter_ffn_layernorm = Norm_Module(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)
        self.kg_adapter_ffn = KgAdapterMLP(
            hidden_size=config.kg_adapter_hidden_size,
            intermediate_size=config.kg_adapter_intermediate_size,
            hidden_act=config.hidden_act,
        )
            
    def forward(self, q_hidden_states, k_hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        text_rep_residual = q_hidden_states
        text_hidden, _, _ = self.kg_adapter_t2n_cross_attn(
            q_hidden_states=q_hidden_states,
            k_hidden_states=k_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        text_hidden = text_hidden + text_rep_residual
        text_rep_residual_ffn = text_hidden
        text_hidden_ffn_input = self.kg_adapter_ffn_layernorm(text_hidden)
        text_hidden_ffn_output = self.kg_adapter_ffn(text_hidden_ffn_input)
        text_hidden = text_hidden_ffn_output + text_rep_residual_ffn
        return text_hidden
    

class LowRankAdapterRouter(nn.Module):
    """Per-layer multi-modal router combining text/image/graph MoE and cross-attention.

    Routes text, image and graph hidden states through modality-specific expert
    groups, and fuses the outputs with an optional cross-attention branch.
    """

    def __init__(
        self,
        KgAdapterCrossAttention: nn.Module,
        Qwen3RMSNorm: nn.Module,
        KgAdapterMLP: nn.Module,
        config:  Qwen3VLConfig,
        num_experts: int = 3,
        text_hidden_size: int = 2560,
        image_hidden_size: int = 1024,
        graph_hidden_size: int = 400,
        topk: int = 2,
        kg_adapter_hidden_size: int = 128,
        layer_id: int = 0,
    ):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.text_hidden_size = text_hidden_size
        self.graph_hidden_size = graph_hidden_size
        self.image_hidden_size = image_hidden_size
        self.topk = topk
        self.kg_adapter_hidden_size = kg_adapter_hidden_size
        
        # ================= Modality switches =================
        self.use_text_moe = getattr(config, 'use_text_moe', True)
        self.use_graph_moe = getattr(config, 'use_graph_moe', False)
        self.use_image_moe = getattr(config, 'use_image_moe', False)
        self.use_cross_attention = getattr(config, 'use_cross_attention', False)
        

        # ================= 1. Text Router module =================
        if self.use_text_moe:
            self.router_text = TokenRouter(num_experts=num_experts, input_dim=text_hidden_size, layer_id=layer_id, top_k=topk, tag='text')
            # replaced with a shared down-projection expert group
            self.text_experts = SharedLoRAExperts(text_hidden_size, kg_adapter_hidden_size, num_experts)

        # ================= 2. Graph Router module =================
        if self.use_graph_moe:
            self.router_graph = TokenRouter(num_experts=num_experts, input_dim=graph_hidden_size, layer_id=layer_id, top_k=topk, tag='graph')
            self.graph_experts = SharedLoRAExperts(graph_hidden_size, kg_adapter_hidden_size, num_experts)
            self.graph_to_text_proj = nn.Linear(graph_hidden_size, text_hidden_size)

        # ================= 3. Image Router module =================
        if self.use_image_moe:
            self.router_image = TokenRouter(num_experts=num_experts,input_dim=text_hidden_size,layer_id=layer_id,top_k=topk,tag='image')
            self.image_experts = SharedLoRAExperts(text_hidden_size, kg_adapter_hidden_size, num_experts)

        # ================= 4. Cross Attention module =================
        if self.use_cross_attention:
            self.text_downscale = nn.Linear(text_hidden_size, kg_adapter_hidden_size)    
            self.graph_downscale = nn.Linear(graph_hidden_size, kg_adapter_hidden_size)
            self.cross_upscale = nn.Linear(kg_adapter_hidden_size, text_hidden_size)
            
            self.router_cross = TokenRouter(num_experts=num_experts, input_dim=kg_adapter_hidden_size, layer_id=layer_id, top_k=topk, tag='cross')
            self.cross_adapters = nn.ModuleList([
                CrossAttnAdapterExpert(KgAdapterCrossAttention, Qwen3RMSNorm, KgAdapterMLP, config)
                for _ in range(num_experts)
            ])
            
    # ---------------- Tensor computation logic ----------------
    def _text_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_weight = self.router_text(hidden_states)
        return self.text_experts(hidden_states, routing_weight)
    
    def _graph_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_weight = self.router_graph(hidden_states)
        return self.graph_experts(hidden_states, routing_weight)  
    
    def _image_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_weight = self.router_image(hidden_states)
        return self.image_experts(hidden_states, routing_weight)
    
    def _cross_forward(self, text_states, graph_states, attention_mask, position_ids):
        text_hidden = self.text_downscale(text_states)   # [B, S, kg_adapter_hidden_size]
        graph_hidden = self.graph_downscale(graph_states) # [B, ?, kg_adapter_hidden_size]

        B = text_hidden.shape[0]
        if graph_hidden.dim() == 2:
            graph_hidden = graph_hidden.unsqueeze(1)
        elif graph_hidden.dim() > 3:
            graph_hidden = graph_hidden.reshape(B, -1, graph_hidden.shape[-1])

        adapter_outputs = torch.stack([
            adapter(
                q_hidden_states=text_hidden,
                k_hidden_states=graph_hidden,
                attention_mask=None,          
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,     
                use_cache=False,
            ) for adapter in self.cross_adapters
        ], dim=1)                             

        adapter_outputs = adapter_outputs.permute(0, 2, 1, 3) 

        routing_weight = self.router_cross(text_hidden)         
        routing_weight_ = routing_weight.unsqueeze(-1)          
        cross_hidden_states = (routing_weight_ * adapter_outputs).sum(dim=-2)  

        return self.cross_upscale(cross_hidden_states)          

    # ---------------- Overall forward ----------------
    def forward(self, attention_mask, 
                position_ids,
                text_states=None, 
                image_states=None,
                hidden_states=None, 
                graph_states=None
                ) -> dict[str, torch.Tensor]:
        outputs = {}

        if text_states is not None and self.use_text_moe:
            outputs['text'] = self._text_forward(text_states)

        if image_states is not None and self.use_image_moe:
            outputs['image'] = self._image_forward(image_states)

        if graph_states is not None and self.use_graph_moe:
            outputs['graph'] = self._graph_forward(graph_states)

        if self.use_cross_attention and hidden_states is not None and graph_states is not None:
            outputs['cross'] = self._cross_forward(
                hidden_states, graph_states, attention_mask, position_ids
            )

        return outputs