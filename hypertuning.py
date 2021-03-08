import os
import subprocess
from dataclasses import replace, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any

import dacite
import pandas as pd
import yaml
from ray import tune
from ray.tune.logger import JsonLoggerCallback

from lib.config import Config, HyperConfig, ModelName, HyperConfigGraphletCounting, HyperConfigAnyGNN
from lightning_modules import train_and_test


def main(config: Config, hconfig: HyperConfig) -> None:
    assert (config.num_splits == len(hconfig.split)), f"{config.num_splits} different from {len(hconfig.split)}"

    def experiment(updates: Dict[str, Any]):
        if config.model is ModelName.GraphletCounting:
            updates["num_layers"] = updates["gc_num_layers"]
            updates.pop("gc_num_layers")

        updates.pop("dataset")
        updates.pop("model")

        updated_config = replace(config, **updates)

        return train_and_test(config=updated_config)

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True
        )
        git_hash = "_" + completed.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        git_hash = ""

    tune_conf = {k: tune.grid_search(v) for k, v in asdict(hconfig).items() if isinstance(v, list)}
    tune_conf["model"] = str(config.model)
    tune_conf["dataset"] = config.data_path / str(config.graphlet_size) / config.dataset_name

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    analysis = tune.run(
        experiment,
        config=tune_conf,
        local_dir=f"./experiments/results_{config.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + git_hash,
        resources_per_trial={"gpu": hconfig.gpu_perc, "cpu": 1},
        log_to_file=True,
        callbacks=[JsonLoggerCallback()],
    )

    df: pd.DataFrame = analysis.results_df
    df.to_csv(analysis._checkpoints[0]["local_dir"] + "/run.csv", index=False)


def config_constructor(name: ModelName):
    if name is ModelName.GraphletCounting:
        return HyperConfigGraphletCounting
    else:
        return HyperConfigAnyGNN


def _main() -> None:
    dacite_conf = dacite.Config(cast=[Enum])
    with open("base_config.yaml") as f, open("hyper_config.yaml") as hf:
        conf: Dict = yaml.load(f, Loader=yaml.FullLoader)
        hconf: Dict = yaml.load(hf, Loader=yaml.FullLoader)

    conf: Config = dacite.from_dict(data_class=Config, data=conf,
                                    config=dacite_conf)
    hconf: HyperConfig = dacite.from_dict(data_class=config_constructor(conf.model),
                                          data=hconf, config=dacite_conf)

    main(conf, hconf)


if __name__ == '__main__':
    _main()
