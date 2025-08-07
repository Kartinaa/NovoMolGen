# Molecule Data Loader

This directory contains all components related to molecular data loading, tokenization, and scaffold filtering, used in training and evaluating generative models for de novo molecule generation.

The pipeline supports multiple molecular representations (`SMILES`, `SELFIES`, `DeepSMILES`, `SAFE`) and allows efficient dataset loading, preprocessing, and tokenization. It integrates well with Hugging Face’s `datasets` and `transformers` libraries.

### Main Components

| File | Description |
|------|-------------|
| `molecule_data_module.py` | Core class `MolDataModule` that handles loading, streaming, tokenization, filtering, and caching of molecular datasets. |
| `molecule_tokenizer.py`  | Implementation of `MoleculeTokenizer`, a unified tokenizer interface built on top of [SAFETokenizer](https://github.com/datamol-io/safe/blob/main/safe/tokenizer.py) with support for BPE, WordPiece, Unigram, and WordLevel models. |
| `ScaffoldFilter.py`      | Implements scaffold-based penalization strategies for controlling exploration and diversity in generative models. |
| `ScaffoldMemory.py`      | Stores previously seen scaffolds and fingerprints to support filtering and memory-based scoring. |
| `utils.py`               | Utilities for converting between SMILES, SELFIES, DeepSMILES, and SAFE. |

---

## Tokenizer Training

To train a tokenizer on a molecular dataset:

```bash
python src/data_loader/molecule_tokenizer.py \
    --dataset "ZINC_270M-raw" \
    --mol_type "SMILES" \
    --tokenizer_type "bpe" \
    --splitter "atomwise" \
    --vocab_size 500
```


This will create a HuggingFace-compatible `tokenizer.json` file saved under `./data/tokenizers/`. The `atomwise` splitter (from Schwaller et al.) is used to tokenize SMILES at the atomic level, improving interpretability and robustness.