from typing import Any, Dict, List, Optional, Tuple
import os
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import avalanche as avl
from avalanche.training import Naive, Replay, EWC, LwF
from avalanche.evaluation import metrics as metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import NCScenario, ni_benchmark
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.training.storage_policy import ReservoirSamplingBuffer

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

log = RankedLogger(__name__, rank_zero_only=True)

class AvalancheExperiment:
    def __init__(
        self,
        cfg: DictConfig,
        model: LightningModule,
        datamodule: LightningDataModule,
    ):
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule
        self.strategy = None
        self.benchmark = None
        
    def prepare_benchmark(self):
        """Prepare the Avalanche benchmark from the datamodule."""
        # Get training data
        train_data = self.datamodule.data_train
        val_data = self.datamodule.data_val
        test_data = self.datamodule.data_test
        
        # Create experience streams
        train_stream = self._create_experience_stream(train_data)
        val_stream = self._create_experience_stream(val_data)
        test_stream = self._create_experience_stream(test_data)
        
        # Create benchmark
        self.benchmark = ni_benchmark(
            train_stream=train_stream,
            test_stream=test_stream,
            n_experiences=self.cfg.avalanche.n_experiences,
            task_labels=False,
            shuffle=True,
            seed=self.cfg.seed if hasattr(self.cfg, 'seed') else 42
        )
        
    def _create_experience_stream(self, dataset):
        """Convert dataset to Avalanche experience stream."""
        # This is a placeholder - you'll need to implement the actual conversion
        # based on your dataset structure
        return avl.training.utils.data_loader.DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True
        )
    
    def setup_strategy(self):
        """Setup the Avalanche training strategy."""
        # Setup evaluation plugin
        eval_plugin = EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            metrics.loss_metrics(epoch=True, experience=True, stream=True),
            loggers=[
                InteractiveLogger(),
                TensorboardLogger()
            ]
        )
        
        # Choose strategy based on config
        strategy_name = self.cfg.avalanche.strategy.lower()
        
        if strategy_name == "naive":
            self.strategy = Naive(
                self.model,
                torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.optimizer.lr),
                self.cfg.model.criterion,
                train_mb_size=self.cfg.data.batch_size,
                train_epochs=self.cfg.trainer.max_epochs,
                eval_mb_size=self.cfg.data.batch_size,
                evaluator=eval_plugin
            )
        elif strategy_name == "replay":
            self.strategy = Replay(
                self.model,
                torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.optimizer.lr),
                self.cfg.model.criterion,
                train_mb_size=self.cfg.data.batch_size,
                train_epochs=self.cfg.trainer.max_epochs,
                eval_mb_size=self.cfg.data.batch_size,
                evaluator=eval_plugin,
                storage_policy=ReservoirSamplingBuffer(max_size=self.cfg.avalanche.replay_buffer_size)
            )
        elif strategy_name == "ewc":
            self.strategy = EWC(
                self.model,
                torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.optimizer.lr),
                self.cfg.model.criterion,
                train_mb_size=self.cfg.data.batch_size,
                train_epochs=self.cfg.trainer.max_epochs,
                eval_mb_size=self.cfg.data.batch_size,
                evaluator=eval_plugin,
                ewc_lambda=self.cfg.avalanche.ewc_lambda
            )
        elif strategy_name == "lwf":
            self.strategy = LwF(
                self.model,
                torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.optimizer.lr),
                self.cfg.model.criterion,
                train_mb_size=self.cfg.data.batch_size,
                train_epochs=self.cfg.trainer.max_epochs,
                eval_mb_size=self.cfg.data.batch_size,
                evaluator=eval_plugin,
                alpha=self.cfg.avalanche.lwf_alpha,
                temperature=self.cfg.avalanche.lwf_temperature
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def train(self):
        """Run the Avalanche training loop."""
        if self.strategy is None or self.benchmark is None:
            raise ValueError("Strategy and benchmark must be initialized before training")
        
        # Training loop
        for experience in self.benchmark.train_stream:
            self.strategy.train(experience)
            self.strategy.eval(self.benchmark.test_stream)
            
        # Final evaluation
        results = self.strategy.eval(self.benchmark.test_stream)
        return results

@task_wrapper
def train_avalanche(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model using Avalanche framework.
    
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

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
    experiment = AvalancheExperiment(cfg, model, datamodule)
    experiment.prepare_benchmark()
    experiment.setup_strategy()
    results = experiment.train()

    return results, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
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