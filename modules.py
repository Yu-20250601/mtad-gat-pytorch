import torch
import torch.nn as nn


class Sparsemax(nn.Module):
    """Sparsemax activation.

    Projects logits onto the probability simplex and produces exact zeros for
    low-score entries, unlike softmax.
    """

    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        dim = self.dim
        z = input_tensor - input_tensor.max(dim=dim, keepdim=True)[0]
        z_sorted, _ = torch.sort(z, dim=dim, descending=True)
        z_cumsum = torch.cumsum(z_sorted, dim=dim)

        dim_size = z.size(dim)
        rhos = torch.arange(1, dim_size + 1, device=z.device, dtype=z.dtype)
        view_shape = [1] * z.dim()
        view_shape[dim] = dim_size
        rhos = rhos.view(view_shape)

        support = 1 + rhos * z_sorted > z_cumsum
        k = support.sum(dim=dim, keepdim=True).clamp(min=1)

        # tau is the data-dependent threshold of simplex projection:
        # only entries with z_i > tau remain positive; the rest become exact 0.
        tau = (torch.gather(z_cumsum, dim, k.long() - 1) - 1) / k
        output = torch.clamp(z - tau, min=0)
        return output


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    :param attention_type: type of attention mechanism to use (softmax, sparsemax)
    :param use_node_embedding: whether to use learnable node embeddings
    :param node_embed_dim: dimension of node embeddings
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True, attention_type='softmax', use_node_embedding=False, node_embed_dim=16):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.attention_type = attention_type
        self.use_node_embedding = use_node_embedding
        self.node_embed_dim = node_embed_dim

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        # Node embedding for learnable embeddings
        if self.use_node_embedding:
            self.node_embedding = nn.Embedding(n_features, node_embed_dim)
            self.node_pair_proj = nn.Linear(2 * node_embed_dim, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()
        self.sparsemax = Sparsemax(dim=2)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        # Add node embedding contribution if enabled
        if self.use_node_embedding:
            emb_pair = self._make_node_embedding_pair_input(x.device)   # (1, k, k, 2*node_embed_dim)
            emb_score = self.node_pair_proj(emb_pair).squeeze(3)        # (1, k, k)
            e = e + emb_score

        if self.use_bias:
            e += self.bias

        # Attention weights
        if self.attention_type == 'softmax':
            attention = torch.softmax(e, dim=2)
        elif self.attention_type == 'sparsemax':
            attention = self.sparsemax(e)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)

    def _make_node_embedding_pair_input(self, device):
        node_ids = torch.arange(self.num_nodes, device=device)
        emb = self.node_embedding(node_ids).unsqueeze(0)  # (1, k, d)
        K = self.num_nodes
        blocks_repeating = emb.repeat_interleave(K, dim=1)
        blocks_alternating = emb.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        return combined.view(1, K, K, 2 * self.node_embed_dim)


class AdaptiveSparseGAT(nn.Module):
    """Adaptive sparse feature-oriented GAT layer for MTAD-GAT.

    Differences from FeatureAttentionLayer:
    1) uses Sparsemax instead of Softmax for sparse graph structure learning
    2) introduces learnable node embedding to model static sensor properties
    3) optionally returns sparse attention matrix for RCA visualization
    """

    def __init__(
        self,
        n_features,
        window_size,
        dropout,
        alpha,
        embed_dim=None,
        use_gatv2=True,
        use_bias=True,
        node_embed_dim=16,
    ):
        super(AdaptiveSparseGAT, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.node_embed_dim = node_embed_dim

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.node_embedding = nn.Embedding(n_features, node_embed_dim)
        self.node_pair_proj = nn.Linear(2 * node_embed_dim, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sparsemax = Sparsemax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_sparse_attention=False):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention a node is all values of one sensor over the window.
        x = x.permute(0, 2, 1)  # (b, k, n)

        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k)
        else:
            Wx = self.lin(x)                                        # (b, k, embed_dim)
            a_input = self._make_attention_input(Wx)                # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)

        emb_pair = self._make_node_embedding_pair_input(x.device)   # (1, k, k, 2*node_embed_dim)
        emb_score = self.node_pair_proj(emb_pair).squeeze(3)        # (1, k, k)
        e = e + emb_score

        if self.use_bias:
            e = e + self.bias

        sparse_attention = self.sparsemax(e)                        # (b, k, k)
        attention = torch.dropout(sparse_attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))                # (b, k, n)
        h = h.permute(0, 2, 1)                                      # (b, n, k)

        if return_sparse_attention:
            return h, sparse_attention
        return h

    def _make_attention_input(self, v):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        return combined.view(v.size(0), K, K, 2 * self.embed_dim)

    def _make_node_embedding_pair_input(self, device):
        node_ids = torch.arange(self.num_nodes, device=device)
        emb = self.node_embedding(node_ids).unsqueeze(0)  # (1, k, d)
        K = self.num_nodes
        blocks_repeating = emb.repeat_interleave(K, dim=1)
        blocks_alternating = emb.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        return combined.view(1, K, K, 2 * self.node_embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


class VAEReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, latent_dim, out_dim, n_layers, dropout):
        super(VAEReconstructionModel, self).__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.to_mu = nn.Linear(in_dim, latent_dim)
        self.to_logvar = nn.Linear(in_dim, latent_dim)

        self.decoder = RNNDecoder(latent_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h_end):
        mu = self.to_mu(h_end)
        logvar = self.to_logvar(h_end)
        z = self.reparameterize(mu, logvar)

        z_rep = z.repeat_interleave(self.window_size, dim=1).view(h_end.size(0), self.window_size, -1)
        decoder_out = self.decoder(z_rep)
        out = self.fc(decoder_out)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return out, kl
