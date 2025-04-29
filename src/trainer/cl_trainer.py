
from lightning import Trainer

class ContinualTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task = 0

    def fit(self, model, datamodule, *args, **kwargs):
        # Train on each task sequentially
        # skip the first task if it's a replay-based method
        
        for task_idx in range(len(datamodule.train_datasets)):
            self.current_task = task_idx
            
            super().fit(model, datamodule, *args, **kwargs)
            datamodule.next_task()  # Update buffer and move to next task
            