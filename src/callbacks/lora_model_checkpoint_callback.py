import os
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import PeftModel

class LoRAModelCheckpoint(ModelCheckpoint):
    def __init__(self, adapter_name="default", **kwargs):
        super().__init__(**kwargs)
        self.adapter_name = adapter_name
        self.last_saved_path = None

    def _save_checkpoint(self, trainer, filepath):
        model = trainer.lightning_module
        peft_model = model.net if hasattr(model, 'net') else model
        
        peft_model.save_pretrained(filepath)
        self.last_saved_path = filepath
        return filepath

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Inject adapter path into main checkpoint"""
        checkpoint['lora_adapter_path'] = self.last_saved_path
        return checkpoint

   