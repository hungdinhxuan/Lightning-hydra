from lightning.pytorch.callbacks import Callback


class BalanceMiniBatchCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        print("Epoch is starting")
        print(batch)
        import sys
        sys.exit()