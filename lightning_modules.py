import copy
from dataclasses import asdict
from enum import Enum
from typing import Dict, Union, Any, List, Callable, Tuple, Optional

import dacite
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.core.memory import ModelSummary
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import DataLoader as PyGDataloader, Batch as PyGBatch
from torch_geometric.utils import degree

from lib.config import Config, ModelName
from lib.data import load_data, Batcher, Batch, Subgraph, stratified_ksplit
from lib.models import GINNet, GCNNet, build_mlp, FinalLayers, GNN, RPGNN, PNANet
from lib.subgraph_models import KaryGNN, KaryRPGNN, GraphletCounting
from losses import Loss, IRMLoss, CELoss, SubgraphRegularizedLoss, LabelSmoothingLoss

PyGBatch.__len__ = lambda self: len(self.y)


class MatthewsCoef(pl.metrics.ConfusionMatrix):
    def compute(self):
        C = super().compute()
        C = C.cpu().numpy()

        # Code taken from sklearn
        # https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/metrics/_classification.py#L862-L875
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_correct = np.trace(C, dtype=np.float64)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

        if np.isnan(mcc):
            return 0.
        else:
            return mcc


class AbstractLightningModule(pl.LightningModule):
    def __init__(self, conf: Config, loss: Loss, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.save_hyperparameters(OmegaConf.create(asdict(conf)))
        self.conf = conf

        self.train_ds, self.val_ds_list, self.test_ds_list, self.id2graphlet_list = load_data(self.conf)

        if conf.dataset_name == "brain-net":
            if self.conf.num_splits > 1:
                self.train_ds, val_ds = stratified_ksplit(
                    self.train_ds, self.conf.num_splits, self.conf.split
                )
                self.val_ds_list = [val_ds]
                self.test_ds_list = []
            else:
                self.val_ds_list = [self.train_ds]

        if self.conf.model in [ModelName.RPGNN, ModelName.GNN]:
            self.train_id2graphlet = None
            self.id2graphlet_list = None
        else:
            self.train_id2graphlet = self.id2graphlet_list[0]

        seed_everything(self.conf.seed)
        np.random.seed(self.conf.seed)
        torch.manual_seed(self.conf.seed)

        model = self.build_model(conf, (self.train_ds, self.train_id2graphlet))

        if not (conf.model == ModelName.GraphletCounting and conf.num_layers == 1):

            h_dim, act = None, None
            if conf.classifier_num_hidden > 0:
                h_dim = conf.classifier_h_dim
                act = nn.ReLU

            batch_norm = None
            if conf.batch_norm.presence and conf.model not in [ModelName.GNN, ModelName.RPGNN]:
                batch_norm = nn.BatchNorm1d(model.out_dim, affine=conf.batch_norm.affine)

            if conf.model is ModelName.GraphletCounting:
                assert conf.classifier_num_hidden == conf.num_layers - 2

            model = FinalLayers(
                model,
                num_out=conf.num_out,
                h_dim=h_dim,
                act=act,
                n_hidden_layers=conf.classifier_num_hidden,
                batch_norm=batch_norm,
                dropout=conf.classifier_dropout,
            )

        self.model = model
        loss.on_epoch_start(epoch=0, model=model)

        with_mcc = conf.dataset_name in ["NCI1", "NCI109", "PROTEINS", "DD"]
        self.train_acc = MatthewsCoef(num_classes=conf.num_out) if with_mcc else pl.metrics.Accuracy()
        self.val_acc = MatthewsCoef(num_classes=conf.num_out) if with_mcc else pl.metrics.Accuracy()
        self.test_acc = MatthewsCoef(num_classes=conf.num_out) if with_mcc else pl.metrics.Accuracy()

    def setup(self, stage):
        logger = None
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            logger = self.logger
        elif isinstance(self.logger, pl_loggers.LoggerCollection):
            for l in self.logger:
                if isinstance(l, pl_loggers.TensorBoardLogger):
                    logger = l
                    break

        if logger:
            if stage == 'fit':
                logger.experiment.add_text("model_summary", f"<pre>{ModelSummary(self, ModelSummary.MODE_FULL)}</pre>")
            elif stage == 'test':
                logger._default_hp_metric = False

    def forward(self, batch: Batch):
        return self.model(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.conf.lr)

    def one_step(self, batch: Union[Batch, PyGBatch],
                 accuracy: Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor
                 ]) -> Tuple[torch.Tensor, torch.Tensor]:

        out = self.forward(batch)
        loss = self.loss(batch, out)
        acc = accuracy(out.argmax(dim=1), batch.y)
        return loss, acc

    def training_step(self, batch: Union[Batch, PyGBatch], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, acc = self.one_step(batch, self.train_acc)
        return {'loss': loss, 'accuracy': acc}

    def training_step_end(self, output):
        self.log('train/batch_run_accuracy', output['accuracy'], prog_bar=True)
        self.log('train/loss', output['loss'], prog_bar=True)
        return output['loss']

    def on_train_epoch_start(self):
        self.loss.on_epoch_start(epoch=self.current_epoch, model=self.model)

    def on_train_epoch_end(self, _) -> None:
        accuracy = self.train_acc.compute()
        self.train_acc.reset()
        self.log('train/accuracy', accuracy)

    def validation_step(self, batch: Union[Batch, PyGBatch], batch_idx: int,
                        dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        loss, _ = self.one_step(batch, self.val_acc)
        return {'val/loss': loss}

    def validation_step_end(self, output):
        self.log('val/batch_loss', output['val/loss'], prog_bar=True)

    def validation_epoch_end(self, _) -> None:
        accuracy = self.val_acc.compute()
        self.val_acc.reset()
        self.log('val/accuracy', accuracy, prog_bar=True)

    def test_step(self, batch: Union[Batch, PyGBatch], batch_idx: int,
                  dataloader_idx: int = 0) -> None:
        self.one_step(batch, self.test_acc)

    def test_epoch_end(
            self, outputs
    ):
        res = self.test_acc.compute()
        self.test_acc.reset()
        self.log('test/accuracy', res)

    def on_epoch_end(self) -> None:
        score = self.trainer.checkpoint_callback.best_model_score
        if score is not None:
            self.log('hp_metric', score, prog_bar=True)

    def train_dataloader(self, epoch: Optional[int] = None) -> DataLoader:
        epoch = epoch or self.current_epoch
        if "SYNTHETIC3" in self.conf.dataset_name and self.conf.model not in [ModelName.GNN, ModelName.RPGNN]:
            self.train_ds, _, _, id2graphlet_list = load_data(self.conf, epoch)
            self.train_id2graphlet = id2graphlet_list[0]

            seed_everything(self.conf.seed + epoch)
            np.random.seed(self.conf.seed + epoch)
            torch.manual_seed(self.conf.seed + epoch)

        common_file = None
        if self.conf.only_common and self.conf.dataset_name != "DD":
            common_file = self.conf.data_path_complete / "common.txt"
            assert common_file.exists()

        return DataLoader(self.train_ds, batch_size=self.conf.batch_size, shuffle=True,
                          collate_fn=Batcher(self.train_id2graphlet,
                                             common_file=common_file),
                          num_workers=0, pin_memory=True)

    def val_dataloader(self) -> List[DataLoader]:
        common_file = None
        if self.conf.only_common and self.conf.dataset_name != "DD":
            common_file = self.conf.data_path_complete / "common.txt"
            assert common_file.exists()

        return [DataLoader(val_ds, batch_size=self.conf.batch_size, shuffle=False,
                           collate_fn=Batcher(id2graphlet, common_file=common_file),
                           num_workers=0, pin_memory=True) for val_ds, id2graphlet in
                zip(self.val_ds_list, self.id2graphlet_list)]

    def test_dataloader(self) -> List[DataLoader]:
        common_file = None
        if self.conf.only_common and self.conf.dataset_name != "DD":
            common_file = self.conf.data_path_complete / "common.txt"
            assert common_file.exists()

        return [DataLoader(test_ds, batch_size=self.conf.batch_size, shuffle=False,
                           collate_fn=Batcher(id2graphlet, common_file=common_file),
                           num_workers=0, pin_memory=True) for test_ds, id2graphlet in
                zip(self.test_ds_list, self.id2graphlet_list)]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["conf"] = self.conf
        if hasattr(self.loss, 'model'):
            model = self.loss.model
            self.loss.model = None
            checkpoint["loss"] = copy.copy(self.loss)
            self.loss.model = model
        else:
            checkpoint["loss"] = self.loss

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs_new):
        conf = checkpoint.pop("conf")
        checkpoint.pop("hyper_parameters")
        super()._load_model_state(checkpoint, strict, conf=conf, loss=checkpoint["loss"])


class BuildGNN(object):
    @classmethod
    def build_gnn(cls, conf: Config, in_dim_or_pre_mlp: Union[int, nn.Sequential], deg=None, act=nn.ReLU):
        if conf.gnn_type == "gin":
            gnn = GINNet(
                in_dim_or_pre_mlp=in_dim_or_pre_mlp,
                num_layers=conf.num_layers,
                vertex_embed_dim=conf.vertex_embed_dim,
                mlp_num_hidden=conf.mlp_num_hidden,
                mlp_hidden_dim=conf.mlp_hidden_dim,
                act=act,
                jk=conf.jk
            )
        elif conf.gnn_type == "gcn":
            gnn = GCNNet(
                in_dim=in_dim_or_pre_mlp,
                num_layers=conf.num_layers,
                vertex_embed_dim=conf.vertex_embed_dim,
                act=act,
                jk=conf.jk
            )
        else:
            gnn = PNANet(
                in_dim=in_dim_or_pre_mlp,
                num_layers=conf.num_layers,
                vertex_embed_dim=conf.vertex_embed_dim,
                deg=deg,
                act=act,
                jk=conf.jk
            )
        return gnn


class KaryGNNModule(AbstractLightningModule, BuildGNN):
    @classmethod
    def build_model(cls, conf: Config, data_context: (Dataset, Dict[int, Subgraph])):
        _, id2graphlet = data_context
        num_node_features = next(iter(id2graphlet.values())).x.size(-1)
        return KaryGNN(
            gnn=super(KaryGNNModule, cls).build_gnn(conf, num_node_features),
            graphlet_sz=conf.graphlet_size
        )


class KaryRPGNNModule(AbstractLightningModule, BuildGNN):
    @classmethod
    def build_model(cls, conf: Config, data_context: (Dataset, Dict[int, Subgraph])):
        _, id2graphlet = data_context
        num_node_features = next(iter(id2graphlet.values())).x.size(-1)

        mlp = build_mlp(
            shapes=(
                conf.graphlet_size + num_node_features,
                conf.mlp_hidden_dim,
                conf.vertex_embed_dim,
            ),
            act=nn.ReLU,
            n_hidden=conf.mlp_num_hidden,
        )
        return KaryRPGNN(
            gnn=super(KaryRPGNNModule, cls).build_gnn(conf=conf, in_dim_or_pre_mlp=mlp),
            graphlet_sz=conf.graphlet_size,
            num_perm=3,
        )


class GraphletCountingModule(AbstractLightningModule):
    @classmethod
    def build_model(cls, conf: Config, data_context: (Dataset, Dict[int, Subgraph])):
        _, id2graphlet = data_context
        graphlets_order = sorted(id2graphlet.keys())
        order_dict = {_id: i for i, _id in enumerate(graphlets_order)}
        return GraphletCounting(
            o_dim=conf.graph_embed_dim if conf.num_layers > 1 else conf.num_out,
            order_dict=order_dict,
        )


def compute_deg_hist(ds):
    max_deg = max([
        max(degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)).item()
        for data in ds
    ])
    deg = torch.zeros(max_deg + 1, dtype=torch.long)
    for data in ds:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


class GNNModule(AbstractLightningModule, BuildGNN):
    @classmethod
    def build_model(cls, conf: Config, data_context: (Dataset, Dict[int, Subgraph])):
        ds, _ = data_context
        num_node_features = ds[0].num_node_features
        node_embedder = super(
            GNNModule, cls
        ).build_gnn(
            conf, num_node_features,
            deg=compute_deg_hist(ds) if conf.gnn_type == "pna" else None
        )
        return GNN(node_embedder=node_embedder, graph_pooling=conf.graph_pooling)

    def train_dataloader(self, epoch: Optional[int] = None) -> DataLoader:
        return PyGDataloader(self.train_ds, batch_size=self.conf.batch_size, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [PyGDataloader(
            val_ds, batch_size=self.conf.batch_size, shuffle=False,
        ) for val_ds in self.val_ds_list]

    def test_dataloader(self) -> List[DataLoader]:
        return [PyGDataloader(
            test_ds, batch_size=self.conf.batch_size, shuffle=False,
        ) for test_ds in self.test_ds_list]


class RPGNNModule(GNNModule):
    @classmethod
    def build_model(cls, conf: Config, data_context: (Dataset, Dict[int, Subgraph])):
        ds, _ = data_context
        num_node_features = ds[0].num_node_features

        mlp = build_mlp(
            shapes=(
                10 + num_node_features,  # FIXME
                conf.mlp_hidden_dim,
                conf.vertex_embed_dim,
            ),
            act=nn.ReLU,
            n_hidden=conf.mlp_num_hidden,
        )

        return RPGNN(
            node_embedder=super(RPGNNModule, cls).build_gnn(conf, mlp),
            num_perm=3,
        )


def model_name_to_cons(name: ModelName) -> Callable[[Config, Loss], AbstractLightningModule]:
    return {
        ModelName.GraphletCounting: GraphletCountingModule,
        ModelName.KaryGNN: KaryGNNModule,
        ModelName.KaryRPGNN: KaryRPGNNModule,
        ModelName.GNN: GNNModule,
        ModelName.RPGNN: RPGNNModule,
    }[name]


def build_trainer_and_model(conf, dirpath='.', progress=False) -> Tuple[pl.Trainer, AbstractLightningModule]:
    cons = model_name_to_cons(conf.model)
    if conf.irm is not None:
        assert conf.irm > 0
        if conf.dataset_name not in ["SYNTHETIC2", "NCI1", "NCI109", "PROTEINS", "DD"]:
            assert conf.cutoff is None
        model = cons(conf, IRMLoss(conf.irm, dataset_name=conf.dataset_name, cutoff=conf.cutoff))
    elif conf.reg_const is not None:
        assert conf.reg_const > 0
        assert conf.dataset_name in ["SYNTHETIC3", "NCI1", "NCI109", "PROTEINS", "DD"] and conf.model in [
            ModelName.KaryGNN, ModelName.KaryRPGNN]
        model = cons(conf, SubgraphRegularizedLoss(conf.reg_const))
    elif conf.label_smooth is not None:
        assert conf.label_smooth > 0
        assert conf.dataset_name in ["NCI1", "NCI109", "PROTEINS", "DD"] and conf.model in [ModelName.KaryGNN,
                                                                                            ModelName.KaryRPGNN]
        model = cons(conf, LabelSmoothingLoss(conf.num_out, conf.label_smooth))
    else:
        model = cons(conf, CELoss(conf.dataset_name))

    kwgs = {}

    if len(model.val_ds_list) > 0:
        kwgs["monitor"] = 'val/accuracy'
        kwgs["mode"] = 'max'
    else:
        print("No validation present")

    chkp = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename="model.ckpt", **kwgs)

    csv_logger = pl_loggers.CSVLogger(dirpath, name='csv_logs')
    tb_logger = pl_loggers.TensorBoardLogger(dirpath, name="tb_logs")
    trainer = pl.Trainer(weights_summary='full', max_epochs=conf.num_epochs, callbacks=[chkp],
                         reload_dataloaders_every_epoch=(
                                 conf.dataset_name in ["SYNTHETIC3", "NCI1", "NCI109", "PROTEINS"]
                                 and conf.model not in [ModelName.GNN, ModelName.RPGNN]),
                         logger=[tb_logger, csv_logger], gpus=1, progress_bar_refresh_rate=0 if not progress else 1)

    return trainer, model


def train_and_test(config: Config, dirpath='.', progress=False):
    trainer, model = build_trainer_and_model(config, dirpath=dirpath, progress=progress)
    # disable signal registering because of tune
    trainer.slurm_connector.register_slurm_signal_handlers = lambda: None

    trainer.fit(model)

    # The following loads the best model
    test_score = trainer.test(ckpt_path='best')
    if isinstance(test_score, int) and test_score == 1:
        print("No test dataset available, skipping")
        test_score = [{'test/accuracy': 0}]
    # Now trainer.model is the best

    train_testing = pl.Trainer(logger=False, gpus=1, checkpoint_callback=False,
                               progress_bar_refresh_rate=0 if not progress else 1)
    train_testing.slurm_connector.register_slurm_signal_handlers = lambda: None

    best_ckpt = torch.load(trainer.checkpoint_callback.best_model_path, map_location='cpu')
    best_epoch = best_ckpt['epoch']
    del best_ckpt
    # Test the best model on training
    print(f"Best model @ epoch {best_epoch}, evaluating on training set at that epoch")
    train_score = train_testing.test(trainer.model, test_dataloaders=[
        trainer.model.train_dataloader(best_epoch)], ckpt_path='best')

    # Dummy line just to check if load_from_checkpoint works
    # as we change the model
    _ = model_name_to_cons(config.model).load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Scores with best model (on validation)
    val_metric = 0
    if trainer.checkpoint_callback.best_model_score is not None:
        val_metric = trainer.checkpoint_callback.best_model_score.item()

    return {
        'train': train_score[0]['test/accuracy'],
        'val': val_metric,
        'test': test_score[0]['test/accuracy'],
    }


def main():
    dacite_conf = dacite.Config(cast=[Enum])
    with open("base_config.yaml") as f:
        conf: Dict = yaml.load(f, Loader=yaml.FullLoader)

    conf: Config = dacite.from_dict(data_class=Config, data=conf,
                                    config=dacite_conf)

    print(train_and_test(conf, dirpath='run', progress=True))


if __name__ == '__main__':
    main()
