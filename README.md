# Size-Invariant Graph Representations for Graph Classification Extrapolations

### Manual dependencies (CUDA)
- PyTorch 1.7.1
- `torch-cluster` 1.5.8
- `torch-geometric` 1.6.3
- `torch-scatter` 2.0.5
- `torch-sparse` 0.6.8
- `torch-spline-conv` 1.2.0
- `torch-geometric` 1.6.3
- `ray[tune]` 1.1.0

Install the additional dependencies
as follows:

```shell
$ pip install -r requirements.txt
```

### Download Data
Please, run the following commands to download and set up the data folder.

```shell
$ wget https://www.dropbox.com/s/38eg3twe4dd1hbt/data.zip
$ unzip data.zip
```

The command above will place the data _already sampled_ in the folder `data/`.
Please specify its absolute path in `base_config.yaml`.

### Hypertune:
The provided configurations allow you to run the hyperparameter tuning of $\Gamma_\text{GIN}$ on `NCI1`.

To tune for other datasets and/or models:
- In `hyper_config.yaml` specify the hyperparameters values. For details on the range of the hyperparameter refer to the Appendix.
- In `base_config.yaml` set `dataset_name` to `NCI1`, `NCI109`, `PROTEINS` or `brain-net` (i.e. schizophrenia).
- In `base_config.yaml` set the `model` to `KaryGNN` (i.e. $\Gamma_\text{GNN}$), `KaryRPGNN` (i.e. $\Gamma_\text{RPGNN}$), `GraphletCounting` (i.e. $\Gamma_\text{1-hot}$), `GNN` or `RPGNN`. You can specify the GNN in `gnn_type` as `pna`, `gcn` or `gin` and the XU-READOUT in `graph_pooling`
as `mean`, `max` or `sum`.

Run
```shell
$ python hypertuning.py
```


### Run a single configuration:
The provided configurations allow you to run $\Gamma_\text{GNN}$ on `NCI1` with the best hyperparameters.

To run for other datasets and/or models specify the parameters
in `base_config.yaml`.

Run
```shell
$ python lightning_modules.py
```