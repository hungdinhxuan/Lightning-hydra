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
from avalanche.evaluation import metrics as metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import NCScenario, ni_benchmark
from avalanche.core import SupervisedPlugin

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
from src.models.base.mdt_module import MDTLitModule
from src.experiments.avalanche_strategies import StrategyFactory, get_available_strategies

log = RankedLogger(__name__, rank_zero_only=True)

class MDTAvalanchePlugin(SupervisedPlugin):
    """Plugin to handle MDT-specific training logic in Avalanche."""
    
    def __init__(self, model: MDTLitModule):
        super().__init__()
        self.model = model
        
    def before_training_exp(self, strategy, **kwargs):
        """Called before each training experience."""
        # Reset metrics for the new experience
        self.model.train_loss.reset()
        self.model.train_acc.reset()
        for k, v in self.model.train_view_acc.items():
            v.reset()
        for k, v in self.model.train_loss_detail.items():
            v.reset()
            
    def after_training_exp(self, strategy, **kwargs):
        """Called after each training experience."""
        # Log experience-specific metrics
        for k, v in self.model.train_view_acc.items():
            strategy.log(f"train/view_{k}_acc", v.compute())
        for k, v in self.model.train_loss_detail.items():
            strategy.log(f"train/view_{k}_loss", v.compute())

class AvalancheMDTExperiment:
    def __init__(
        self,
        cfg: DictConfig,
        model: MDTLitModule,
        datamodule: LightningDataModule,
    ):
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule
        self.strategy = None
        self.benchmark = None
        
    def prepare_benchmark(self):
        """Prepare the Avalanche benchmark from the datamodule."""
        # Setup the datamodule to load experience datasets
        self.datamodule.setup()
        
        # Get experience streams directly from the datamodule
        train_stream = []
        test_stream = []
        
        for exp_idx in range(self.cfg.avalanche.n_experiences):
            # Get datasets for this experience
            train_dataset = self.datamodule.experience_datasets['train'][exp_idx]
            test_dataset = self.datamodule.experience_datasets['eval'][exp_idx]
            
            # Create dataloaders for this experience
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.data.batch_size,
                shuffle=True,
                collate_fn=self.datamodule.collate_fn,
                num_workers=self.cfg.data.num_workers,
                pin_memory=self.cfg.data.pin_memory
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                collate_fn=self.datamodule.collate_fn,
                num_workers=self.cfg.data.num_workers,
                pin_memory=self.cfg.data.pin_memory
            )
            
            train_stream.append(train_loader)
            test_stream.append(test_loader)
        
        # Create benchmark using experience streams
        from avalanche.benchmarks.utils import make_classification_dataset
        from avalanche.benchmarks import GenericCLScenario
        
        # Create experiences list
        train_experiences = []
        test_experiences = []
        
        for exp_idx in range(self.cfg.avalanche.n_experiences):
            train_exp = avl.benchmarks.utils.AvalancheDataset(
                self.datamodule.experience_datasets['train'][exp_idx],
                task_labels=exp_idx,
                targets_task_labels=exp_idx
            )
            test_exp = avl.benchmarks.utils.AvalancheDataset(
                self.datamodule.experience_datasets['eval'][exp_idx],
                task_labels=exp_idx,
                targets_task_labels=exp_idx
            )
            train_experiences.append(train_exp)
            test_experiences.append(test_exp)
        
        self.benchmark = GenericCLScenario(
            train_stream=train_experiences,
            test_stream=test_experiences,
            n_experiences=self.cfg.avalanche.n_experiences,
            task_labels=True,
            shuffle=False
        )
    
    def setup_strategy(self):
        """Setup the Avalanche training strategy with MDT support."""
        # Setup evaluation plugin with MDT-specific metrics
        eval_plugin = EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            metrics.loss_metrics(epoch=True, experience=True, stream=True),
            loggers=[
                InteractiveLogger(),
                TensorboardLogger()
            ]
        )
        
        # Create MDT-specific plugin
        mdt_plugin = MDTAvalanchePlugin(self.model)
        
        # Get strategy name from config
        strategy_name = self.cfg.avalanche.strategy.lower()
        
        # Log available strategies if requested
        if self.cfg.avalanche.get("list_strategies", False):
            strategies = get_available_strategies()
            log.info("Available strategies:")
            for name, desc in strategies.items():
                log.info(f"  - {name}: {desc}")
        
        # Create strategy using factory
        self.strategy = StrategyFactory.create_strategy(
            strategy_name=strategy_name,
            model=self.model,
            cfg=self.cfg,
            eval_plugin=eval_plugin,
            plugins=[mdt_plugin]
        )
    
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
def train_avalanche_mdt(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model using Avalanche framework with MDT support.
    
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: MDTLitModule = hydra.utils.instantiate(cfg.model)

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
    experiment = AvalancheMDTExperiment(cfg, model, datamodule)
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
    metric_dict, _ = train_avalanche_mdt(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value

if __name__ == "__main__":
    main() 