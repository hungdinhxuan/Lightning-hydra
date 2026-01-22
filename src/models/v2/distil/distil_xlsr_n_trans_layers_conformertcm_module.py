import torch
from src.models.components.distil.distil_xlsr_n_trans_layers_conformertcm import Model as Distil_XLSR_N_Trans_Layer_ConformerTCM
from src.models.base.normal_module import NormalLitModule
from typing import Any, Dict, Tuple, Union
from torch import nn

class Distil_XLSR_N_Trans_Layer_ConformerTCMNormalLitModule(NormalLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs  # 👈 Store kwargs for later use
        self.args = args
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()

    def init_model(self, **kwargs) -> nn.Module:
        _args = self.args.get("conformer", {})
        _kwargs = self.args.get("kwargs", {})
        print(_args)
        print(_kwargs)
        return Distil_XLSR_N_Trans_Layer_ConformerTCM(_args, **_kwargs)
    
    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)
    
    # @torch.enable_grad() 
    # @torch.inference_mode(False)
    # def on_test_start(self) -> None:
    #     """Lightning hook that is called when a test starts."""
    #     is_pruning = self.kwargs.get("is_pruning", False)
    #     print("This hook is called")
    #     class W2V2_TA_Wrapper(nn.Module):
    #         def __init__(self, model: nn.Module):
    #             super().__init__()
    #             self.model = model

    #         def extract_feat(self, x):
    #             feat, _ = self.model(x.squeeze(-1))
    #             return feat

    #         def forward(self, x):
    #             return self.extract_feat(x)
    #     from torchaudio.models.wav2vec2.utils import import_fairseq_model
    #     self.net.front_end = W2V2_TA_Wrapper(import_fairseq_model(self.net.front_end.model)).to(self.device)
        
    #     # Change frontend model
    #     #self.net = self.net.requires_grad_()
        
    #     if is_pruning:
    #         print("is_pruning")
    #         import torch_pruning as tp
    #         #with torch.enable_grad():
    #         example_inputs = torch.randn(1, 32000).to(self.device)
    #         print(example_inputs.shape)
    #         imp = tp.importance.GroupMagnitudeImportance(p=2) 
    #         ignored_layers = []
            
    #         #model = self.net
    #         # Enable gradient
    #         self.net = self.net.requires_grad_()
            
    #         for name, m in self.net.named_modules():
    #             # Ignore Wav2Vec2 transformer and embeddings
    #             # if "front_end.model.encoder" in name:  # tsransformer stack
    #             #     ignored_layers.append(m)
    #             if "q_proj" in name or "k_proj" in name or "v_proj" in name:
    #                 ignored_layers.append(m)
    #             if "positional_emb" in name or "class_token" in name or "pos_embed" in name:
    #                 ignored_layers.append(m)
    #             # Ignore norms (LayerNorm/BatchNorm) if unsure
    #             if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
    #                 ignored_layers.append(m)
    #             # Ignore final classifier
    #             if name.endswith("fc5") or name.endswith("LL"):
    #                 ignored_layers.append(m)
                
    #             if "qkv" in name:
    #                 ignored_layers.append(m)
            
    #         pruner = tp.pruner.BasePruner(
    #                 self.net,
    #                 example_inputs,
    #                 importance=imp,
    #                 pruning_ratio=0.2,                 # disable global ratio
    #                 #pruning_ratio_dict=pruning_ratio_dict,
    #                 ignored_layers=ignored_layers,
    #                 round_to=8,
    #                 isomorphic=True, # enable isomorphic pruning to improve global ranking
    #                 global_pruning=True, # global pruning

    #             )

    #     # 3. Prune the model
    #         base_macs, base_nparams = tp.utils.count_ops_and_params(self.net, example_inputs)
    #         tp.utils.print_tool.before_pruning(self.net) # or print(model)
    #         pruner.step()
    #         tp.utils.print_tool.after_pruning(self.net) # or print(model), this util will show the difference before and after pruning
    #         macs, nparams = tp.utils.count_ops_and_params(self.net, example_inputs)
    #         print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")

        # import sys
        # sys.exit(0)