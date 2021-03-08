from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch

from lib.data import Batch


class Loss(object):
    def on_epoch_start(self, **context):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SubgraphRegularizedLoss(Loss):
    def __init__(self, lam):
        self.model = None
        self.lam = lam

    def on_epoch_start(self, **context):
        assert 'model' in context
        self.model = context['model']

    def perturbe(self, x):
        new_x = torch.zeros(x.shape).to(x.device)

        # randomize the node features
        feat = np.random.choice(x.size(-1), x.size(0))
        new_x[torch.arange(x.size(0)), feat] = 1
        return new_x

    def __call__(self, batch: Batch, out: torch.Tensor):
        assert isinstance(batch, Batch)

        graphlets_repr = self.model.graph_embedder.graphlets_repr
        new_batch = Batch(
            self.perturbe(batch.x),
            batch.edge_index,
            batch.graph_has_graphlet,
            batch.graphlet_ids,
            batch.y
        )
        _ = self.model(new_batch)
        perturbed_graphlets_repr = self.model.graph_embedder.graphlets_repr
        reg_loss = torch.norm(graphlets_repr - perturbed_graphlets_repr, dim=-1, p=2).mean()
        return F.cross_entropy(out, batch.y) + self.lam * reg_loss


class CELoss(Loss):
    def __init__(self, dataset_name):
        weight = {
            "NCI1": [1 / 0.6230, 1 / 0.3770],
            "NCI109": [1 / 0.6204, 1 / 0.3796],
            "PROTEINS": [1 / 0.4197, 1 / 0.5803],
            "DD": [1 / 0.3547, 1 / 0.6453],
            "deezer_ego_nets": [1 / 0.5521, 1 / 0.4479],
            "twitch_egos": [1 / 0.3905, 1 / 0.6095],
            "IMDB-BINARY": [1 / 0.4899, 1 / 0.5101]
        }
        self.weight = weight.get(dataset_name, None)

    def on_epoch_start(self, **context):
        pass

    def __call__(self, batch: Union[Batch, PyGBatch], out: torch.Tensor):
        weight = torch.tensor(self.weight).to(out.device) if self.weight is not None else None
        return F.cross_entropy(out, batch.y, weight=weight)


class LabelSmoothingLoss(Loss):
    def __init__(self, classes: int, smoothing: float = 0.0, dim: int = -1):
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def on_epoch_start(self, **context):
        pass

    def __call__(self, batch: Batch, out: torch.Tensor):
        pred = out.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, batch.y.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class IRMLoss(Loss):
    def __init__(self, lam: float, dataset_name: str, cutoff: int = None):
        self.lam = lam
        self.cutoff = cutoff
        self.epoch = 0

        weight = {
            "NCI1": [1 / 0.6230, 1 / 0.3770],
            "NCI109": [1 / 0.6204, 1 / 0.3796],
            "PROTEINS": [1 / 0.4197, 1 / 0.5803],
            "DD": [1 / 0.3547, 1 / 0.6453],
            "deezer_ego_nets": [1 / 0.5521, 1 / 0.4479],
            "twitch_egos": [1 / 0.3905, 1 / 0.6095],
            "IMDB-BINARY": [1 / 0.4899, 1 / 0.5101]
        }
        self.weight = weight.get(dataset_name, None)

    def on_epoch_start(self, **context: Dict[str, Any]):
        assert 'epoch' in context
        self.epoch = context['epoch']

    @classmethod
    def irm_penalty(cls, out, target, weight=None):
        with torch.enable_grad():
            scale = torch.tensor(1., device=out.device, requires_grad=True)
            loss = F.cross_entropy(out * scale, target, weight=weight)
            grad = torch.autograd.grad(loss, [scale], retain_graph=True, create_graph=True)[0]
        return torch.sum(grad ** 2).item()

    def __call__(self, batch: PyGBatch, out: torch.Tensor):
        assert isinstance(batch, PyGBatch)
        _, sizes = torch.unique(batch.batch, return_counts=True)
        envs = sizes > self.cutoff if self.cutoff is not None else sizes

        weight = torch.tensor(self.weight).to(out.device) if self.weight is not None else None

        lam = self.lam if self.epoch > 100 else 1
        penalties = []
        losses = []
        for curr_env in torch.unique(envs):
            has_env = envs == curr_env
            penalties.append(IRMLoss.irm_penalty(out[has_env], batch.y[has_env], weight=weight))
            losses.append(F.cross_entropy(out[has_env], batch.y[has_env], weight=weight))

        return (sum(losses) / len(losses) + lam * sum(penalties) / len(penalties)) / lam
