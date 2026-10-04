import torch.nn as nn

class BaseRGCN(nn.Module):
    """
    Base RGCN model class.
    Implements the basic framework of a relational graph convolutional network,
    supporting multiple RGCN layer variants.
    """
    
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, 
                 encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        """
        Initialize the base RGCN model.

        Args:
        num_nodes: number of nodes in the graph
        h_dim: hidden dimension
        out_dim: output dimension
        num_rels: number of relation types
        num_bases: number of basis decompositions (for reducing parameter count)
        num_basis: another basis-decomposition parameter (possibly redundant)
        num_hidden_layers: number of hidden layers
        dropout: dropout rate
        self_loop: whether to use self-loop connections
        skip_connect: whether to use skip connections
        encoder_name: encoder type name
        opn: operation type, e.g. "sub" (possibly for relation handling)
        rel_emb: pretrained relation embeddings
        use_cuda: whether to use GPU
        analysis: whether to run in analysis mode
        """
        super(BaseRGCN, self).__init__()
        
        # basic model parameters
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        
        # relation-related parameters
        self.rel_emb = rel_emb  # relation embeddings
        self.opn = opn  # operation type
        
        # create RGCN layers
        self.build_model()
        
        # create the initial node features
        self.features = self.create_features()

    def build_model(self):
        """
        Build the model architecture:
        - input layer (i2h)
        - hidden layers (h2h) x num_hidden_layers
        - output layer (h2o)
        """
        self.layers = nn.ModuleList()
        
        # build the input layer (input to hidden)
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
            
        # build the hidden layers (hidden to hidden)
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
            
        # build the output layer (hidden to output)
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def create_features(self):
        """
        Initialize each node's features.
        Subclasses can override this to provide specific feature initialization.

        Returns:
        None or a node-feature tensor
        """
        return None

    def build_input_layer(self):
        """
        Build the input layer.
        Subclasses can override this to implement a specific input layer.

        Returns:
        None or an input-layer module
        """
        return None

    def build_hidden_layer(self, idx):
        """
        Build a hidden layer - abstract method that must be implemented by subclasses.

        Args:
        idx: index of the hidden layer

        Returns:
        a hidden-layer module
        """
        raise NotImplementedError

    def build_output_layer(self):
        """
        Build the output layer.
        Subclasses can override this to implement a specific output layer.

        Returns:
        None or an output-layer module
        """
        return None

    def forward(self, g):
        # If predefined node features exist, assign them as the node ID feature.
        if self.features is not None:
            g.ndata['id'] = self.features

        # Perform message passing layer by layer; each layer updates node features in-place.
        for layer in self.layers:
            layer(g)

        # Return the final node representations and remove the 'h' field from the graph.
        return g.ndata.pop('h')