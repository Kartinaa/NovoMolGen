#!/usr/bin/env python3
"""
Sample molecules from NovoMolGen model for sanity check.
"""

import os
import sys
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from src.models.modeling_utils import generate_valid_smiles
from src.data_loader.molecule_tokenizer import MoleculeTokenizer
from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig

# Download model from Hugging Face
# local_dir = snapshot_download(
#     repo_id="bisectgroup/NovoMolGen_32M_SAFE_BPE",
#     revision="main",
#     local_dir="./models/novomolgen_32M_safe_bpe/",
#     local_dir_use_symlinks=False
# )


tokenizer_path = "data/tokenizers/tokenizer_bpe_None_SAFE_hf.json"
model_path = "models/novomolgen_32M_safe_bpe/"

try:
    # Load tokenizer
    mol_tokenizer = MoleculeTokenizer.load(tokenizer_path)
    tokenizer = mol_tokenizer.get_pretrained()
    
    # Load model
    cfg = NovoMolGenConfig.from_pretrained(
        model_path,
        use_flash_attn=False,
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        )
    model = NovoMolGen.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=torch.float32,
        device_map='auto',
    ).eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"✅ Loaded custom model from {model_path}")
except Exception as e:
    print(f"❌ Failed to load custom model: {e}")
    


