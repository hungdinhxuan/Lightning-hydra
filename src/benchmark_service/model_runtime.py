"""Resident model runtime for benchmark jobs."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import hydra
import lightning as L
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from src.benchmark_service.schemas import RuntimeConfig
from src.utils import extras, instantiate_callbacks, instantiate_loggers

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_py.execution import BenchmarkConfig


class LoadedBenchmarkModel:
    """Load model once, run many benchmark configs with fresh datamodules."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        project_root: Optional[Union[str, Path]] = None,
        warmup: bool = False,
    ) -> None:
        self.runtime_config = runtime_config
        self.project_root = Path(project_root or rootutils.find_root(indicator=".project-root"))
        self.configs_dir = self.project_root / "configs"
        self.model: Optional[LightningModule] = None
        self.base_cfg: Optional[DictConfig] = None
        self.loaded = False
        self.load_count = 0
        self.warmup = warmup

    def load(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.runtime_config.gpu_id)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        torch.set_float32_matmul_precision("high")

        cfg = self._compose_cfg(self.runtime_config.extra_overrides)
        extras(cfg)
        self.model = hydra.utils.instantiate(cfg.model)
        self.model.eval()
        self.base_cfg = cfg
        self.loaded = True
        self.load_count += 1

        if self.warmup:
            self._warmup_noop()

    def reload(self, runtime_config: RuntimeConfig) -> None:
        self.release()
        self.runtime_config = runtime_config
        self.load()

    def release(self) -> None:
        self.model = None
        self.base_cfg = None
        self.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    def matches(self, runtime_config: RuntimeConfig) -> bool:
        return self.runtime_config.signature() == runtime_config.signature()

    def execute_benchmark(self, config: BenchmarkConfig) -> bool:
        self.ensure_loaded()
        expected = RuntimeConfig(
            gpu_id=str(config.gpu_number),
            config_path=config.yaml_config,
            model_path=config.base_model_path,
            adapter_path=config.adapter_paths,
            is_ln=config.is_base_model_path_ln,
            precision=self.runtime_config.precision,
            extra_overrides=list(self.runtime_config.extra_overrides),
        )
        if not self.matches(expected):
            raise ValueError(
                "resident runtime does not match benchmark config; "
                "reload worker or submit job to matching worker"
            )
        return self._run_test(config)

    def _compose_cfg(self, extra_overrides: List[str]) -> DictConfig:
        overrides = [
            f"experiment={self.runtime_config.config_path}",
            "++train=False",
            "++test=True",
            "++model.spec_eval=True",
            f"++model.base_model_path={self.runtime_config.model_path}",
            f"++model.is_base_model_path_ln={str(self.runtime_config.is_ln).lower()}",
        ]
        if self.runtime_config.adapter_path:
            overrides.append(f"++model.adapter_paths={self.runtime_config.adapter_path}")
        overrides.extend(extra_overrides)

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(self.configs_dir)):
            return compose(config_name="train.yaml", return_hydra_config=False, overrides=overrides)

    def _cfg_for_job(self, config: BenchmarkConfig) -> DictConfig:
        if self.base_cfg is None:
            raise RuntimeError("runtime not loaded")
        cfg = copy.deepcopy(self.base_cfg)
        with open_dict(cfg):
            cfg.train = False
            cfg.test = True
            cfg.model.score_save_path = str(config.score_save_path.absolute())
            cfg.model.spec_eval = True
            cfg.data.data_dir = str(config.data_dir.absolute())
            cfg.data.batch_size = config.batch_size
            cfg.data.args.protocol_path = str(config.protocol_path.absolute())
            cfg.data.args.random_start = config.is_random_start
            cfg.data.args.trim_length = config.trim_length
            cfg.paths.output_dir = str(config.score_save_path.parent.absolute() / "hydra_runs" / config.score_save_path.stem)
        return cfg

    def _run_test(self, config: BenchmarkConfig) -> bool:
        if self.model is None:
            raise RuntimeError("runtime model not loaded")
        cfg = self._cfg_for_job(config)
        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        callbacks = instantiate_callbacks(cfg.get("callbacks"))
        logger = instantiate_loggers(cfg.get("logger"))
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        self._prepare_model_for_job(self.model, cfg)
        trainer.test(model=self.model, datamodule=datamodule, ckpt_path=None)
        self._cleanup_after_job(datamodule, trainer)
        return True

    @staticmethod
    def _prepare_model_for_job(model: LightningModule, cfg: DictConfig) -> None:
        score_save_path = str(cfg.model.score_save_path)
        if hasattr(model, "score_save_path"):
            model.score_save_path = score_save_path
        if hasattr(model, "spec_eval"):
            model.spec_eval = bool(cfg.model.get("spec_eval", True))
        if hasattr(model, "kwargs") and isinstance(model.kwargs, dict):
            model.kwargs["score_save_path"] = score_save_path
            model.kwargs["spec_eval"] = bool(cfg.model.get("spec_eval", True))
        model.eval()

    @staticmethod
    def _cleanup_after_job(datamodule: LightningDataModule, trainer: Trainer) -> None:
        if hasattr(datamodule, "teardown"):
            datamodule.teardown("test")
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _warmup_noop(self) -> None:
        # No synthetic batch here: legacy models have varied batch formats.
        if self.model is not None:
            self.model.eval()
