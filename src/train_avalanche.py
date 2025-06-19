from typing import Any, Dict, List, Optional, Tuple
import os
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.experiments.avalanche_mdt_experiment import AvalancheMDTExperiment
from src.data.avalanche_multiview_datamodule import AvalancheMultiViewDataModule

log = RankedLogger(__name__, rank_zero_only=True)
import warnings
warnings.filterwarnings('ignore')

@task_wrapper
def train_avalanche(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model using Avalanche framework for continual learning.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: AvalancheMultiViewDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Initialize and run Avalanche experiment
    log.info(f"Starting Avalanche continual learning with {cfg.avalanche.strategy} strategy")
    log.info(f"Number of experiences: {cfg.avalanche.n_experiences}")
    
    if cfg.avalanche.strategy == "replay":
        log.info(f"Replay buffer size: {cfg.avalanche.replay_buffer_size}")
    
    experiment = AvalancheMDTExperiment(cfg, model, datamodule)
    experiment.prepare_benchmark()
    experiment.setup_strategy()
    
    # Train using Avalanche
    results = experiment.train()
    
    # Convert results to metrics format expected by Lightning
    metric_dict = {}
    if results:
        for key, value in results.items():
            if isinstance(value, (int, float)):
                metric_dict[f"avalanche/{key}"] = value

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for Avalanche continual learning training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Ensure we're using avalanche experiment config
    if not hasattr(cfg, 'avalanche'):
        log.error("Missing 'avalanche' configuration section. Please use experiment=avalanche_mdt")
        raise ValueError("Missing 'avalanche' configuration section")
    
    # Validate required avalanche configurations
    required_avalanche_keys = ['n_experiences', 'strategy']
    for key in required_avalanche_keys:
        if not hasattr(cfg.avalanche, key):
            log.error(f"Missing required avalanche configuration: {key}")
            raise ValueError(f"Missing required avalanche configuration: {key}")
    
    # Validate strategy-specific configurations
    if cfg.avalanche.strategy == "replay" and not hasattr(cfg.avalanche, 'replay_buffer_size'):
        log.warning("replay_buffer_size not specified, using default value of 1000")
        cfg.avalanche.replay_buffer_size = 1000
    
    # apply extra utilities
    extras(cfg)

    # train the model
    metric_dict, _ = train_avalanche(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main() 