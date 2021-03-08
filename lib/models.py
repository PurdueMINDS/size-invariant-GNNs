import torch
from torch import nn
from typing import Union, Tuple, Optional, Callable

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, GCNConv, PNAConv


def build_mlp(
        shapes: Union[Tuple[int, int], Tuple[int, int, int]],
        act: Optional[Callable[[], nn.Module]] = None,
        n_hidden: int = 0,
        batch_norm=False,
        dropout: float = 0.0,
) -> nn.Sequential:
    """Returns a Multi Layer Perceptron (MLP)

    Args:
        shapes: either (in_dim, o_dim) or (in_dim, h_dim, o_dim)
        act: activation function such as nn.ReLU
        n_hidden: number of hidden layers

    Returns:
        nn.Sequential: the MLP or the linear layer

    Example:
        >>> x = torch.randn(32,10)
        >>> mlp = build_mlp(in_dim=10, h_dim=128, o_dim=10, act=nn.ReLU, n_hidden=4)
        >>> mlp(x).shape
        torch.Size([batch_size, *, 10])
    """
    assert (len(shapes) == 2 and act is None and n_hidden == 0) or (
            len(shapes) == 3 and act is not None and n_hidden > 0
    )
    if len(shapes) == 2:
        linear = nn.Linear(shapes[0], shapes[1])
        return nn.Sequential(nn.Dropout(p=dropout), linear) if dropout > 0.0 else linear

    in_dim, h_dim, o_dim = shapes
    h_layers = [nn.Linear(in_dim, h_dim)]
    for i in range(n_hidden):
        if batch_norm:
            h_layers.append(nn.BatchNorm1d(h_dim))
        h_layers.append(act())

        if dropout > 0.0:
            h_layers.append(nn.Dropout(p=dropout))

        h_layers.append(nn.Linear(h_dim, h_dim if i < n_hidden - 1 else o_dim))

    return nn.Sequential(*h_layers)


class GCNNet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_layers: int,
            vertex_embed_dim: int,
            act,
            jk=True,
    ):
        super(GCNNet, self).__init__()
        self.act = act()

        gcn_layers_list = []
        batch_norms_list = []
        for i in range(num_layers):
            gcn_layers_list.append(
                GCNConv(vertex_embed_dim if i > 0 else in_dim, vertex_embed_dim)
            )
            batch_norms_list.append(nn.BatchNorm1d(vertex_embed_dim))

        self.gcn_layers = nn.ModuleList(gcn_layers_list)
        self.batch_norms = nn.ModuleList(batch_norms_list)

        self.jk = jk
        if self.jk:
            self.out_dim = in_dim + vertex_embed_dim * num_layers
        else:
            self.out_dim = vertex_embed_dim

    def forward(self, x, edge_index):
        o = x
        h_v = [o]

        for bn, layer in zip(self.batch_norms, self.gcn_layers):
            o = self.act(bn(layer(o, edge_index)))
            h_v.append(o)

        h_v = torch.cat(h_v, dim=1)
        return h_v if self.jk else o


class PNANet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_layers: int,
            vertex_embed_dim: int,
            deg: torch.Tensor,
            act,
            jk=True,
    ):
        super(PNANet, self).__init__()
        self.act = act()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        pna_layers_list = []
        batch_norms_list = []
        for i in range(num_layers):
            pna_layers_list.append(
                PNAConv(
                    in_channels=vertex_embed_dim if i > 0 else in_dim,
                    out_channels=vertex_embed_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                )
            )
            batch_norms_list.append(nn.BatchNorm1d(vertex_embed_dim))

        self.pna_layers = nn.ModuleList(pna_layers_list)
        self.batch_norms = nn.ModuleList(batch_norms_list)

        self.jk = jk
        if self.jk:
            self.out_dim = in_dim + vertex_embed_dim * num_layers
        else:
            self.out_dim = vertex_embed_dim

    def forward(self, x, edge_index):
        o = x
        h_v = [o]

        for bn, layer in zip(self.batch_norms, self.pna_layers):
            o = self.act(bn(layer(o, edge_index)))
            h_v.append(o)

        return torch.cat(h_v, dim=1) if self.jk else o


class GINNet(nn.Module):
    """Module implementing the GIN model [https://arxiv.org/abs/1810.00826]
        Returns for each vertex, the concatenation of the embeddings after
        each GINConv layer

    Args:
        in_dim_or_pre_mlp: input dimension, that is number of vertex features.
            It assumes vertex features are 1D tensors. If it is a
            torch.nn.Sequential model, then in the first iteration applies
            a MLP before summation, as described in the paper.
        num_layers: number of GINConv layers
        vertex_embed_dim: vertex dimension after each GINConv layer
        mlp_num_hidden: number of hidden layers for each MLP
        mlp_hidden_dim: dimension of the hidden layers of the MLP
        act: activation function such as nn.ReLU
    """

    def __init__(
            self,
            in_dim_or_pre_mlp: Union[int, nn.Sequential],
            num_layers: int,
            vertex_embed_dim: int,
            mlp_num_hidden: int,
            mlp_hidden_dim: int,
            act,
            jk=True,
    ):
        super(GINNet, self).__init__()
        self.act = act()

        if isinstance(in_dim_or_pre_mlp, nn.Sequential):
            self.pre_mlp = in_dim_or_pre_mlp
            first_layer_dim = self.pre_mlp[-1].out_features
        elif isinstance(in_dim_or_pre_mlp, int):
            self.pre_mlp = None
            first_layer_dim = in_dim_or_pre_mlp
        else:
            raise TypeError(
                f"Expected int or nn.Sequential as in_dim_or_pre_mlp but found {type(in_dim_or_pre_mlp)}"
            )

        gin_layers_list = []
        batch_norms_list = []  # List of batchnorms applied to the output of MLP
        for i in range(num_layers):
            gin_layers_list.append(
                GINConv(
                    build_mlp(
                        shapes=(
                            vertex_embed_dim if i > 0 else first_layer_dim,
                            mlp_hidden_dim,
                            vertex_embed_dim,
                        ),
                        act=act,
                        n_hidden=mlp_num_hidden,
                        batch_norm=True,
                    ),
                    train_eps=True,
                )
            )
            batch_norms_list.append(nn.BatchNorm1d(vertex_embed_dim))

        self.gin_layers = nn.ModuleList(gin_layers_list)
        self.batch_norms = nn.ModuleList(batch_norms_list)

        self.jk = jk
        if self.jk:
            self.out_dim = first_layer_dim + vertex_embed_dim * num_layers
            if self.pre_mlp is not None:
                self.out_dim += self.pre_mlp[0].in_features
        else:
            self.out_dim = vertex_embed_dim

    def forward(self, x, edge_index):
        o = x
        h_v = [o]
        if self.pre_mlp is not None:
            o = self.act(self.pre_mlp(x))
            h_v.append(o)

        for bn, layer in zip(self.batch_norms, self.gin_layers):
            o = self.act(bn(layer(o, edge_index)))
            h_v.append(o)

        h_v = torch.cat(h_v, dim=1)
        return h_v if self.jk else o


class FinalLayers(nn.Module):
    def __init__(
            self,
            graph_embedder,
            num_out,
            h_dim: Optional[int] = None,
            act: Optional[Callable[[], nn.Module]] = None,
            n_hidden_layers: int = 0,
            batch_norm: Optional[nn.Module] = None,
            dropout: float = 0.0,
    ):
        super(FinalLayers, self).__init__()
        self.graph_embedder = graph_embedder
        self.batch_norm = batch_norm
        shapes = (
            (graph_embedder.out_dim, num_out)
            if h_dim is None
            else (graph_embedder.out_dim, h_dim, num_out)
        )
        self.layers = build_mlp(
            shapes=shapes, act=act, n_hidden=n_hidden_layers, dropout=dropout
        )

    def forward(self, batch):
        out = self.graph_embedder(batch)
        if self.batch_norm is not None:
            out = self.batch_norm(out)

        out = self.layers(out)
        return out


class GNN(nn.Module):
    def __init__(
            self,
            node_embedder: nn.Module,
            graph_pooling: str = "sum"
    ):
        super(GNN, self).__init__()

        self.node_embedder = node_embedder

        if graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise TypeError(f"Expected sum or mean but found {graph_pooling}")

        self.out_dim = node_embedder.out_dim

    def forward(self, batch):
        x, edge_index, batch_ids = batch.x, batch.edge_index, batch.batch
        h_v = self.node_embedder.forward(x, edge_index)
        h_g = self.pool(h_v, batch_ids)
        return h_g


class RPGNN(nn.Module):
    def __init__(
            self,
            node_embedder: nn.Module,
            num_perm: int = 1
    ):
        super(RPGNN, self).__init__()

        self.fixed_size = 10  # FIXME
        self.register_buffer("node_ids", torch.eye(self.fixed_size))
        self.num_perm = num_perm

        self.node_embedder = node_embedder
        self.out_dim = node_embedder.out_dim

    def forward(self, batch):
        x, edge_index, batch_ids = batch.x, batch.edge_index, batch.batch

        out = None
        for _ in range(self.num_perm):
            new_x = torch.empty(x.size(0), x.size(1) + self.fixed_size).to(x.device)
            for graph in range(torch.max(batch_ids).item() + 1):
                node_indices = (batch_ids == graph).nonzero().squeeze(1)

                graph_size = node_indices.size(0)
                perm = torch.randperm(graph_size)

                node_ids = self.__getattr__(f"node_ids").repeat(
                    graph_size // self.fixed_size + 1, 1
                )[:graph_size]
                permuted_node_ids = node_ids.to(x.device)[perm, :]
                new_x[node_indices] = torch.cat([x[node_indices], permuted_node_ids], dim=1)

            h_v = self.node_embedder.forward(new_x, edge_index)
            h_g = global_add_pool(h_v, batch_ids)

            if out is None:
                out = h_g / self.num_perm
            else:
                out += h_g / self.num_perm
        return out
