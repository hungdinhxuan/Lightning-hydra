from __future__ import annotations

from typing import Any, Dict, Union

import torch

from src.models.base.mdt_module import MDTLitModule
from src.models.components.xlsr_sls import Model as XLSRSLS


class XLSRSLSMDTLitModule(MDTLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()
        self.compile_model = kwargs.get("compile_model", kwargs.get("compile", False))
        self._is_compiled = False

    def forward(self, x: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        return self.net(x)

    def init_model(self, **kwargs: Any) -> torch.nn.Module:
        injected_net = kwargs.get("net", None)
        if injected_net is not None:
            return injected_net

        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRSLSMDTLitModule")

        args = self.args or {}
        sls_args = args.get("sls", args)
        return XLSRSLS(sls_args, ssl_pretrained_path)

    def on_fit_start(self) -> None:
        if self.compile_model and (not self._is_compiled):
            self.net = torch.compile(self.net)
            self._is_compiled = True
