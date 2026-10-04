import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import related components from the local modules
from .layers import RGCNBlockLayer as RGCNLayer
from .layers import UnionRGCNLayer, RGCNBlockLayer
from .model import BaseRGCN
from .decoder import ConvTransE, ConvTransR


class RGCNCell(BaseRGCN):
    """
    RGCN cell, inheriting from BaseRGCN.
    Performs relational graph convolution on a graph; a graph-convolution unit.
    """
    
    def build_hidden_layer(self, idx):
        """
        Build a hidden layer.

        Args:
            idx: layer index, used to determine which layer this is

        Returns:
            the built RGCN layer
        """
        # use RReLU as the activation function
        act = F.rrelu
        
        # if this is not the first layer, do not use basis decomposition
        # Basis Decomposition is a parameter-sharing technique that reduces the number of
        # relation embedding matrix parameters.
        # Each relation matrix is decomposed into a linear combination of shared basis matrices:
        # relation matrix = sum(basis matrix x combination coefficient)
        if idx:
            self.num_basis = 0
            
        # decide whether to use skip connection
        if self.skip_connect:
            sc = False if idx == 0 else True  # no skip connection for the first layer
        else:
            sc = False
            
        # build the corresponding layer based on the encoder type
        if self.encoder_name == "uvrgcn":
            # use a Union RGCN layer
            return UnionRGCNLayer(
                self.h_dim,              # input dim
                self.h_dim,              # output dim
                self.num_rels,           # number of relation types
                self.num_bases,          # number of bases
                activation=act,          # activation function
                dropout=self.dropout,    # dropout rate
                self_loop=self.self_loop,  # whether to use self-loop
                skip_connect=sc,         # skip connection
                rel_emb=self.rel_emb     # relation embeddings
            )
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        """
        Forward pass.

        Args:
            g: graph structure
            init_ent_emb: initial entity embeddings
            init_rel_emb: initial relation embeddings

        Returns:
            updated node representations
        """
        if self.encoder_name == "uvrgcn":
            # get node ids
            node_id = g.n_id.squeeze()
            
            # initialize node features from the node ids
            g.x = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            
            # propagate through each RGCN layer
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
                
            return g.x
        else:
            # if features exist, use them for initialization
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.n_id = self.features
                
            # get node ids and initialize node features
            node_id = g.n_id.squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            
            # propagate differently depending on whether skip connection is used
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
                    
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    """
    Recurrent RGCN.
    Models temporal knowledge graphs and captures the evolution of entities and relations over time.
    """
    
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, 
                 num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, 
                 layer_norm=False, input_dropout=0, hidden_dropout=0, feat_dropout=0, 
                 aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        """
        Initialize the recurrent RGCN model.

        Args:
            decoder_name: decoder name (e.g. "convtranse")
            encoder_name: encoder name (e.g. "uvrgcn")
            num_ents: number of entities
            num_rels: number of relations
            num_static_rels: number of static relations
            num_words: number of words
            h_dim: hidden dimension
            opn: operation type
            sequence_len: sequence length
            num_bases: number of bases
            num_basis: number of bases (spare)
            num_hidden_layers: number of hidden layers
            dropout: dropout rate
            self_loop: whether to use self-loop
            skip_connect: whether to use skip connections
            layer_norm: whether to use layer normalization
            input_dropout/hidden_dropout/feat_dropout: various dropout parameters
            aggregation: aggregation method
            weight: static-loss weight
            discount: discount factor
            angle: angle parameter (for the static loss)
            use_static: whether to use the static-graph constraint
            entity_prediction: whether to do entity prediction
            relation_prediction: whether to do relation prediction
            use_cuda: whether to use CUDA
            gpu: GPU device id
            analysis: whether to run analysis
        """
        super(RecurrentRGCN, self).__init__()# call the parent class's initializer

        # save the basic parameters
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None  # entity representation at the current time
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        # initialize the weight matrices w1 and w2 (possibly for feature transformation)
        self.w1 = torch.nn.Parameter(
            torch.Tensor(self.h_dim, self.h_dim), 
            requires_grad=True
        ).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(
            torch.Tensor(self.h_dim, self.h_dim), 
            requires_grad=True
        ).float()
        torch.nn.init.xavier_normal_(self.w2)# use the Xavier normal method to initialize layer weights

        # initialize relation embeddings (forward and inverse, hence num_rels * 2)
        self.emb_rel = torch.nn.Parameter(
            torch.Tensor(self.num_rels * 2, self.h_dim), 
            requires_grad=True
        ).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        # initialize the dynamic entity embeddings
        self.dynamic_emb = torch.nn.Parameter(  # 1. defined as a learnable parameter
            torch.Tensor(num_ents, h_dim),      # 2. embedding matrix shape [num_entities, hidden_dim]
            requires_grad=True                  # 3. update these embeddings during training
        ).float()                               # 4. make sure it is floating point
        torch.nn.init.normal_(self.dynamic_emb) # 5. initialize embeddings with a normal distribution

        # if the static-graph constraint is used
        if self.use_static:
            # initialize the word embeddings
            self.words_emb = torch.nn.Parameter(
                torch.Tensor(self.num_words, h_dim), 
                requires_grad=True
            ).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            
            # build the static RGCN layer
            self.statci_rgcn_layer = RGCNBlockLayer(
                self.h_dim, 
                self.h_dim, 
                self.num_static_rels * 2,  # static relations also include inverse ones
                num_bases,
                activation=F.rrelu, 
                dropout=dropout, 
                self_loop=False, 
                skip_connect=False
            )
            
            # use MSE for the static loss
            self.static_loss = torch.nn.MSELoss()

        # define the loss functions, combining Softmax and negative log-likelihood.
        self.loss_r = torch.nn.CrossEntropyLoss()  # relation prediction loss
        self.loss_e = torch.nn.CrossEntropyLoss()  # entity prediction loss

        # initialize the RGCN cell
        self.rgcn = RGCNCell(
            num_ents,
            h_dim,
            h_dim,
            num_rels * 2,  # including inverse relations
            num_bases,
            num_basis,
            num_hidden_layers,
            dropout,
            self_loop,
            skip_connect,
            encoder_name,
            self.opn,
            self.emb_rel,
            use_cuda,
            analysis
        )

        # weight and bias of the time-gating mechanism
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                                 

        # GRU cell for relation evolution
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        # initialize the decoders
        if decoder_name == "convtranse":
            # entity prediction decoder (based on ConvTransE)
            self.decoder_ob = ConvTransE(
                num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout
            )
            # relation prediction decoder (based on ConvTransR)
            self.rdecoder = ConvTransR(
                num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout
            )
        else:
            raise NotImplementedError 

    def forward(self, g_list, static_graph, use_cuda):
        """
        Forward pass, processing a temporal graph sequence.

        Args:
            g_list: list of temporal graphs
            static_graph: static knowledge graph
            use_cuda: whether to use CUDA

        Returns:
            history_embs: list of historical entity embeddings
            static_emb: static entity embeddings
            self.h_0: relation embeddings
            gate_list: gate list (for analysis)
            degree_list: degree list (for analysis)
        """
        gate_list = []
        degree_list = []

        # if the static-graph constraint is used
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            
            # concatenate the dynamic and word embeddings as the node features of the static graph
            static_graph.ndata['h'] = torch.cat(
                (self.dynamic_emb, self.words_emb), dim=0
            )
            
            # pass through the static RGCN layer
            self.statci_rgcn_layer(static_graph, [])
            
            # extract the static embeddings of the entity part
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            
            # initialize the current entity representation as the static embedding
            self.h = static_emb
        else:
            # without the static constraint, directly use the dynamic embedding
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        # store the historical embeddings
        history_embs = []

        # iterate over the graph of each time step
        for i, g in enumerate(g_list):
            # get the entity representations related to relations
            temp_e = self.h[g.r_to_e]
            
            # initialize the input features of relations
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda \
                      else torch.zeros(self.num_rels * 2, self.h_dim).float()
            
            # for each relation, compute the average representation of its related entities
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]  # get the entities of this relation
                x_mean = torch.mean(x, dim=0, keepdim=True)  # compute the mean
                x_input[r_idx] = x_mean  # store the relation's average representation
            
            # use GRU to update the relation embeddings
            if i == 0:
                # first time step, initialize the relation hidden state
                x_input = torch.cat((self.emb_rel, x_input.to(self.emb_rel.device)), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                # later time steps, update based on the previous moment
                x_input = torch.cat((self.emb_rel, x_input.to(self.emb_rel.device)), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            # update the entity representation via RGCN
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            
            # time-gating mechanism: fuse the current and historical representations
            time_weight = F.sigmoid(
                torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias
            )
            # weighted fusion
            self.h = time_weight * current_h.to(self.h.device) + (1 - time_weight) * self.h
            
            # save the entity representation of the current time
            history_embs.append(self.h)
        
        # return the results
        if len(history_embs):
            return history_embs, static_emb, self.h_0, gate_list, degree_list
        else:
            # if there are no historical embeddings, return the current h
            return history_embs, static_emb, self.h, gate_list, degree_list

    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        """
        Prediction function, used in the test stage.

        Args:
            test_graph: test graph
            num_rels: number of relations
            static_graph: static graph
            test_triplets: test triples
            use_cuda: whether to use CUDA

        Returns:
            all_triples: all triples (including inverse ones)
            score: entity prediction scores
            score_rel: relation prediction scores
        """
        with torch.no_grad():
            # build inverse triples (h, r, t) -> (t, r_inv, h)
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            
            # merge forward and inverse triples
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            # forward pass to get the evolved embeddings
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            
            # use the embedding of the last time step
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # compute the entity prediction scores
            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            
            # compute the relation prediction scores
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            
            return all_triples, score, score_rel

    def get_loss(self, glist, triples, static_graph, use_cuda):
        """
        Compute the training loss.

        Args:
            glist: graph list
            triples: triples
            static_graph: static graph
            use_cuda: whether to use CUDA

        Returns:
            loss_ent: entity prediction loss
            loss_rel: relation prediction loss
            loss_static: static-constraint loss
        """
        # initialize the losses
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # build inverse triples
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        # forward pass
        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        
        # get the embeddings used for prediction (last time step)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        # compute the entity prediction loss
        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        # compute the relation prediction loss
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(
                pre_emb, r_emb, all_triples, mode="train"
            ).view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        # compute the static-constraint loss
        if self.use_static:
            # discount=1: the angle constraint grows with the time step
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    # compute the angle of the current time step
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    
                    # compute the cosine similarity between static and dynamic embeddings
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    
                    # compute the loss: hope the cosine similarity is not less than cos(step)
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(
                        torch.masked_select(math.cos(step) - sim_matrix, mask)
                    )
                    
            # discount=0: all time steps use the same angle constraint
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(
                        torch.masked_select(math.cos(step) - sim_matrix, mask)
                    )
                    
        return loss_ent, loss_rel, loss_static