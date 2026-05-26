from __future__ import annotations

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from src.models.base.mdt_module import MDTLitModule
from src.models.components.xlsr_sls import Model, get_sls_layer_features
from src.models.v2.xlsr_sls_mdt_module import XLSRSLSMDTLitModule


class FakeXLSRFrontend(torch.nn.Module):
    def __init__(self, num_layers: int = 4, frames: int = 9, dim: int = 12) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.frames = frames
        self.out_dim = dim
        self.proj = torch.nn.Linear(1, dim)

    def extract_feat(self, x: torch.Tensor):
        batch = x.shape[0]
        base = x[:, : self.frames].mean(dim=1, keepdim=True).reshape(batch, 1, 1)
        layer_results = []
        for idx in range(self.num_layers):
            layer = self.proj(base + float(idx)).repeat(1, self.frames, 1)
            layer_results.append((layer.transpose(0, 1), None))
        return layer_results[-1][0].transpose(0, 1), layer_results


def test_get_sls_layer_features_from_fairseq_layout() -> None:
    layers = [(torch.randn(7, 2, 5), None), (torch.randn(7, 2, 5), None)]
    pooled, full = get_sls_layer_features(layers, batch_size=2)
    assert pooled.shape == (2, 2, 5)
    assert full.shape == (2, 2, 7, 5)


def test_xlsr_sls_component_forward() -> None:
    model = Model(
        {
            "ssl_dim": 12,
            "hidden_dim": 16,
            "output_dim": 2,
            "pool_time": 3,
            "pool_feature": 4,
        },
        front_end=FakeXLSRFrontend(),
    )
    logits = model(torch.randn(3, 1600))
    assert logits.shape == (3, 2)


def test_xlsr_sls_mdt_module_inherits_mdt_and_runs_multiview_step() -> None:
    net = Model(
        {
            "ssl_dim": 12,
            "hidden_dim": 16,
            "output_dim": 2,
            "pool_time": 3,
            "pool_feature": 4,
        },
        front_end=FakeXLSRFrontend(),
    )
    module = XLSRSLSMDTLitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=None,
        args={},
        weighted_views={"1": 1.0, "2": 1.0},
        cross_entropy_weight=[0.5, 0.5],
    )
    assert isinstance(module, MDTLitModule)

    multiview_batch = {
        "1": (torch.randn(2, 1600), torch.tensor([0, 1])),
        "2": (torch.randn(2, 1600), torch.tensor([1, 0])),
    }
    loss, preds, targets, loss_detail, view_acc = module.model_step(multiview_batch)
    assert loss.ndim == 0
    assert preds.shape == (4,)
    assert targets.shape == (4,)
    assert set(loss_detail) == {"1", "2"}
    assert set(view_acc) == {"1", "2"}


def test_xlsr_sls_hydra_config_instantiates_with_injected_net() -> None:
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model=xlsr_sls"],
            return_hydra_config=True,
        )

    net = Model(
        {
            "ssl_dim": 12,
            "hidden_dim": 16,
            "output_dim": 2,
            "pool_time": 3,
            "pool_feature": 4,
        },
        front_end=FakeXLSRFrontend(),
    )
    with open_dict(cfg):
        cfg.model.args.sls.ssl_dim = 12
        cfg.model.args.sls.hidden_dim = 16
        cfg.model.args.sls.pool_time = 3
        cfg.model.args.sls.pool_feature = 4

    module = hydra.utils.instantiate(cfg.model, net=net, ssl_pretrained_path=None)
    assert isinstance(module, MDTLitModule)
    logits = module(torch.randn(2, 1600))
    assert logits.shape == (2, 2)
