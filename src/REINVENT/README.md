
# REINVENT
## Molecular De Novo design using Recurrent Neural Networks and Reinforcement Learning

Searching chemical space as described in [Molecular De Novo Design through Deep Reinforcement Learning](https://arxiv.org/abs/1704.07555)

This project is a fork of the original REINVENT repository available [here](https://github.com/MarcusOlivecrona/REINVENT).

### Key Modifications
- **Codebase Upgrade**: Transitioned from TensorFlow/Python 2.7 as used in the original repository to PyTorch/Python 3.10+, enhancing the codebase for compatibility with PyTorch 2.1.2.
- **Logging System**: Integrated Weights & Biases for experiment tracking, replacing the original Vizard visualization system.
- **Scoring and Optimization**: Updated the scoring functions based on our [metrics](../eval/molecule_evaluation.py)


## Usage

To train a Prior starting with a SMILES file called mols.smi:

* First filter the SMILES and construct a vocabulary from the remaining sequences. `python ./src/REINVENT/data_structs.py mols.smi`   - Will generate data/mols_filtered.smi and data/Voc. A filtered file containing around 1.1 million SMILES and the corresponding Voc is contained in "data".

* Then use `python ./src/REINVENT/train_prior.py` to train the Prior. A pretrained Prior is included.

To train an Agent using our Prior, use the main.py script. For example:

* `python src/REINVENT_main.py --scoring-function QED --num-steps 1000`

You can adjust the parameters as needed to tailor the training process.

