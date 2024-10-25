import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils
from pytorch_lightning import LightningModule
import os
import glob

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def average_checkpoints(checkpoint_dir: str, model: LightningModule, top_k: int = 5) -> str:
    """Average the last top_k checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Lightning model
        top_k: Number of checkpoints to average
        
    Returns:
        Path to the averaged checkpoint
    """
    # Get all checkpoint files that match the pattern
    averaged_ckpt_path = os.path.join(checkpoint_dir, f"averaged_top{top_k}.ckpt")
    if os.path.exists(averaged_ckpt_path):
        log.warning(f"Averaged checkpoint already exists: {averaged_ckpt_path}")
        return averaged_ckpt_path

    checkpoint_pattern = os.path.join(checkpoint_dir, "epoch_*.ckpt")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    
    if len(checkpoint_files) == 0:
        log.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
        
    # Take the last top_k checkpoints
    checkpoint_files = checkpoint_files[-top_k:]
    log.info(f"Averaging {len(checkpoint_files)} checkpoints: {checkpoint_files}")
    
    # Determine device
    device = next(model.parameters()).device
    
    # Load first checkpoint completely to get metadata
    first_checkpoint = torch.load(checkpoint_files[0], map_location=device)
    state_dict = first_checkpoint['state_dict']
    averaged_state = {key: state_dict[key].clone() for key in state_dict}
    
    # Add other checkpoints
    for ckpt_path in checkpoint_files[1:]:
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        for key in averaged_state:
            averaged_state[key] += state_dict[key]
            
    # Average the sum
    for key in averaged_state:
        averaged_state[key] = averaged_state[key] / len(checkpoint_files)
    
    # Create checkpoint with all necessary metadata using dictionary unpacking
    checkpoint = {
        **first_checkpoint,  # Keep all original metadata
        "state_dict": averaged_state,  # Override with averaged weights
        "epoch": first_checkpoint.get("epoch", 0),  # Ensure these keys exist
        "global_step": first_checkpoint.get("global_step", 0),
    }
    
    
    # Save averaged checkpoint
    
    torch.save(checkpoint, averaged_ckpt_path)
    
    return averaged_ckpt_path