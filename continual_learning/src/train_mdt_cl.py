import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from avalanche.benchmarks import CLStream, CLScenario
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer

from src.models.xlsr_conformertcm_mdt_cl import XLSRConformertcmMDTCL
from src.training.mdt_cl_trainer import MDTCLTrainer
from src.data.avalanche_datamodule import AvalancheDataModule
from src.configs.mdt_cl_config import MDTCLConfig

@hydra.main(version_base=None, config_path="configs", config_name="mdt_cl_config")
def main(cfg: DictConfig) -> None:
    """Train the MDT Continual Learning model.
    
    Args:
        cfg: Configuration object
    """
    # Convert config to dataclass
    config = OmegaConf.to_object(cfg)
    
    # Create data module
    data_module = AvalancheDataModule(
        data_dir=config.data.data_dir,
        protocol_dir=config.data.protocol_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        cache_dir=config.data.cache_dir,
        enable_cache=config.data.enable_cache,
        wav_samp_rate=config.data.wav_samp_rate,
        trim_length=config.data.trim_length,
        padding_type=config.data.padding_type,
        random_start=config.data.random_start,
        views=config.data.views,
        view_padding_configs=config.data.view_padding_configs
    )
    
    # Setup data
    data_module.setup()
    
    # Create model
    model = XLSRConformertcmMDTCL(
        ssl_pretrained_path=config.model.ssl_pretrained_path,
        conformer_config=config.model.conformer_config,
        replay_buffer_size=config.model.replay_buffer_size,
        weighted_views=config.model.weighted_views,
        adaptive_weights=config.model.adaptive_weights
    )
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    # Create trainer
    trainer = MDTCLTrainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
        max_epochs=config.training.max_epochs,
        patience=config.training.patience,
        replay_buffer_size=config.model.replay_buffer_size,
        batch_size=config.training.batch_size,
        eval_every=config.training.eval_every,
        checkpoint_dir=config.logging.checkpoint_dir,
        log_dir=config.logging.log_dir
    )
    
    # Train on each task
    for task_id, (train_stream, val_stream) in enumerate(zip(
        data_module.train_stream, data_module.val_stream
    )):
        print(f"\nTraining on Task {task_id + 1}")
        
        # Train on current task
        trainer.train(train_stream, val_stream)
        
        # Save checkpoint after each task
        trainer.save_checkpoint(f"task_{task_id + 1}")
        
        # Evaluate on all previous tasks
        print("\nEvaluating on all previous tasks:")
        for prev_task_id in range(task_id + 1):
            eval_stream = data_module.val_stream[prev_task_id]
            metrics = trainer.evaluate(eval_stream)
            print(f"Task {prev_task_id + 1} metrics:", metrics)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 