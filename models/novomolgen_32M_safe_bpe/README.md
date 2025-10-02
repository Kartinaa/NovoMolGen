---
license: mit
datasets:
  - ZINC-22
language:
  - en
tags:
  - molecular-generation
  - drug-discovery
  - llama
  - flash-attention
pipeline_tag: text-generation
---

# NovoMolGen

NovoMolGen is a family of molecular foundation models trained on 
1.5 billion ZINC-22 molecules with Llama architectures and FlashAttention. 
It achieves state-of-the-art performance on both unconstrained and 
goal-directed molecule generation tasks.


## Transformers-native HF checkpoint (`revision="NovoMolGen_checkpoint"`)

We also publish a Transformers-native checkpoint on the `NovoMolGen_checkpoint` revision. This version loads directly with `AutoModelForCausalLM` and works out-of-the-box with `.generate(...)`.

```python 
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("bisectgroup/NovoMolGen_32M_SAFE_BPE", revision='main', device_map='auto')
>>> tokenizer = AutoTokenizer.from_pretrained("bisectgroup/NovoMolGen_32M_SAFE_BPE", revision='main')

>>> input_ids = torch.tensor([[tokenizer.bos_token_id]]).expand(4, -1).contiguous().to(model.device)
>>> outs = model.generate(input_ids=input_ids, temperature=1.0, max_length=128, do_sample=True, pad_token_id=tokenizer.eos_token_id)

>>> molecules = [t.replace(" ", "") for t in tokenizer.batch_decode(outs, skip_special_tokens=True)]
```

## Citation

```bibtex
@misc{chitsaz2025novomolgenrethinkingmolecularlanguage,
      title={NovoMolGen: Rethinking Molecular Language Model Pretraining}, 
      author={Kamran Chitsaz and Roshan Balaji and Quentin Fournier and Nirav Pravinbhai Bhatt and Sarath Chandar},
      year={2025},
      eprint={2508.13408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.13408}, 
}
```
