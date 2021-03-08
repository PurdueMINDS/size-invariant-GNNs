import torch
from torch import nn
from torch.nn import Parameter

from lib.data import Batch
from lib.models import GINNet


class KaryGNN(nn.Module):
    """Graph Classification model that applies a GNN to the graph of
        k-sized graphlets and represents a graph as weighted sum of
        the nodes of the graphlets, where the weight is the number of
        occurrences of the graphlet within the graph

    Args:
        gnn: GNN model used for vertex representation
        num_output: Number of classes
        graphlet_sz
    """

    def __init__(
            self,
            gnn: GINNet,
            graphlet_sz: int,
    ):
        super(KaryGNN, self).__init__()
        self.graphlet_sz = graphlet_sz
        self.gnn = gnn
        self.out_dim = gnn.out_dim
        self.graphlets_repr = None

    def forward(self, batch: Batch) -> torch.Tensor:
        # Pass once the graph of graphlets to obtain the representation of each
        # graphlet's node
        out = self.gnn(batch.x, batch.edge_index)

        # Represent a graphlet as the sum of its nodes
        graphlets_repr = (
            out.reshape(-1, self.graphlet_sz, out.size(-1)).sum(dim=1)
        )
        self.graphlets_repr = graphlets_repr  # save for regularization

        eps = 0.0001
        normalized_estimates = batch.graph_has_graphlet / (batch.graph_has_graphlet.sum(dim=-1).unsqueeze(-1) + eps)
        return normalized_estimates.matmul(graphlets_repr)


class KaryRPGNN(KaryGNN):
    def __init__(
            self,
            gnn,
            graphlet_sz: int,
            num_perm: int = 1
    ):
        super(KaryRPGNN, self).__init__(
            gnn, graphlet_sz
        )
        self.num_perm = num_perm
        self.register_buffer("node_ids_" + str(graphlet_sz), torch.eye(graphlet_sz))

    def forward(self, batch: Batch) -> torch.Tensor:
        """The graph representation is the mean of the representations
            obtained in each permutation where each representation is
            the weighted sum of graphlets' representations.
        """
        out = None

        for _ in range(self.num_perm):
            # take one permutation for each graphlet
            num_graphlets = batch.graph_has_graphlet.size(-1)
            perms = [torch.randperm(self.graphlet_sz) for _ in range(num_graphlets)]

            # Permute one-hot-encodings
            permuted_node_ids = self.__getattr__(
                f"node_ids_{self.graphlet_sz}"
            )[torch.cat(perms), :]

            permuted_batch = Batch(
                torch.cat([batch.x, permuted_node_ids], dim=1),
                batch.edge_index,
                batch.graph_has_graphlet,
                batch.graphlet_ids,
                batch.y
            )

            if out is None:
                out = super().forward(permuted_batch) / self.num_perm
            else:
                out += super().forward(permuted_batch) / self.num_perm
        return out


class GraphletCounting(nn.Module):
    """Graph Classification model that uses Graphlet Counting to represent
        a graph
    """

    def __init__(self, o_dim: int, order_dict):
        super(GraphletCounting, self).__init__()

        self.order_dict = order_dict

        self.weight = Parameter(
            torch.empty(len(order_dict), o_dim)
        )
        self.bias = Parameter(torch.empty(o_dim))
        self.reset_parameters()
        self.out_dim = o_dim

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** (1 / 2),
                                 mode='fan_out')
        bound = 1 / (self.weight.size(0) ** (1 / 2))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, batch: Batch):
        eps = 0.0001
        normalized_estimates = batch.graph_has_graphlet / (batch.graph_has_graphlet.sum(dim=-1).unsqueeze(-1) + eps)
        graphs_repr = normalized_estimates.matmul(
            self.weight[[self.order_dict[k] for k in batch.graphlet_ids]]
        ) + self.bias

        # The following line is commented because it is unnecessary
        # to scale the result of the linear layer above since the weights
        # of the layer will adjust accordingly

        # out *= self.graphlet_sz

        # There isn't an activation function since batch normalization
        # is applied by the classifier which provides the non-linearity

        return graphs_repr
