from lightning.pytorch.callbacks import BaseFinetuning

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, module_names=["feature_extractor"]):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._module_names = module_names

    def freeze_before_training(self, pl_module):

        # Freeze all modules in module_names 
        for module_name in self._module_names:
            self.freeze(getattr(pl_module, module_name))

    def finetune_function(self, pl_module, current_epoch, optimizer):
    # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:

            # Unfreeze all modules in module_names
            for module_name in self._module_names:
                self.unfreeze_and_add_param_group(
                    modules=getattr(pl_module, module_name),
                    optimizer=optimizer,
                    train_bn=True,
                )