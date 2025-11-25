from pathlib import Path
import torch
from models.modeling_novomolgen_backup import NovoMolGen, NovoMolGenConfig

model_path = "outputs/11_23_25_SAFEGen_lr_KLnormclaped0.1_unfreezeall_noloar_xattn67891011/checkpoint-33400/full_model"
config = NovoMolGenConfig.from_pretrained(model_path)
model = NovoMolGen.from_pretrained(model_path, config=config)

for name, p in model.named_parameters():
    if "transformer.gates" in name:
        print(name, "requires_grad =", p.requires_grad)