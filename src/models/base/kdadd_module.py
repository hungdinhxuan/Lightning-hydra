from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from typing import Union
from torchmetrics.metric import Metric
import torch
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM
from src.models.base.base_module import BaseLitModule
from src.models.base.teacher_student_module import TeacherStudentLitModule

class KDAddLitModule(TeacherStudentLitModule):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:

        super().__init__(optimizer, scheduler, args, **kwargs)
    
    def init_losses(self, **kwargs) -> Dict[str, Metric]:
        """
            Initialize the losses with the given arguments. This method is used to initialize the losses
            with the given arguments. The losses are initialized with the given arguments and the losses are returned.
        """
        return {
            "kd_loss": MeanMetric(),
            "ce_loss": MeanMetric(),
            "embed_loss": MeanMetric(),
        }

    