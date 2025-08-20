# Molecule Evaluation Metrics

In this codebase, we provide a suite of evaluation metrics for small molecule generation. It includes molecular property assessments, docking evaluations, and fragment-based metrics for evaluating generative models.

## Provided Metrics

### 1. Molecular Properties
These metrics assess the fundamental chemical and physical properties of generated molecules:
- **logP**: Octanol-water partition coefficient (Lipophilicity).
- **SA (Synthetic Accessibility Score)**: A heuristic measure of the ease of synthesis.
- **QED (Quantitative Estimate of Drug-likeness)**: A metric that balances multiple drug-like properties.
- **TPSA (Topological Polar Surface Area)**: Determines a molecule's ability to permeate cell membranes.
- **Molecular Weight**: The sum of atomic weights in a molecule.
- **Bertz Complexity**: A measure of structural complexity.
- **Number of Aromatic/Aliphatic Rings**: Counts different ring types.
- **Number of Rotatable Bonds**: A measure of molecular flexibility.

### 2. Fragment-Based Metrics
- **FCD (Fréchet ChemNet Distance)**: Measures distributional similarity between generated and real molecules.
- **SNN (Average Maximum Similarity)**: Measures similarity between generated and known molecules.
- **Scaffold Similarity**: Compares core molecular structures between generated and real compounds.
- **Fragment Distribution**: Assesses molecular fragmentation consistency.

### 3. Docking Scores
- **Docking Simulations**: Predicts binding affinity scores for molecular docking simulations.
- **Vina-based Docking**: Uses [AutoDock Vina](http://vina.scripps.edu/) for flexible molecular docking.

### 4. Drug-likeness & Toxicity Prediction
- **DBPP Score**: A predictor of drug-likeness using machine learning.
- **ADMET Models**: Includes models for absorption, metabolism, and toxicity predictions.
- **RAScore**: Measures synthetic feasibility using a trained model.

### 5. Guacamol Benchmark Tasks
- Includes similarity-based and scaffold-hopping tasks such as:
  - Rediscovery tasks (Celecoxib, Troglitazone, Thiothixene)
  - Molecular property optimization (DRD2, QED, logP)

### 6. Tartarus-Based Evaluations
- **PCE Prediction**: Predicts power conversion efficiency for organic photovoltaics.
- **Reactivity Estimation**: Estimates reaction barriers for synthesis feasibility.
- **TADF Evaluation**: Computes triplet-singlet energy gaps for OLED applications.

## Source of Metrics
These metrics are derived from various cheminformatics tools and prior research:
- **RDKit**: Open-source toolkit for cheminformatics.
- **AutoDock Vina**: Docking simulations.
- **FCD**: Torch implementation from `fcd_torch`.
- **Moses Benchmark**: Drug-like molecule generation evaluation.
- **Guacamol**: Generative molecule benchmark suite.
- **Tartarus**: Organic electronics property prediction.

## Usage

### Running the Evaluator
The evaluation framework can be used via the `MoleculeEvaluator` class:

```python
from molecule_evaluation import MoleculeEvaluator

evaluator = MoleculeEvaluator(task_names=["logP", "QED", "SA", "Docking_fa7"])
results = evaluator(gen_smiles=["CCO", "CCC"])
print(results)
```

### Dependencies
To use this evaluation suite, install the following dependencies:
```bash
pip install rdkit numpy pandas torch rootutils loguru tdc fcd_torch
```
For docking evaluations, ensure you have:
```bash
sudo apt install autodock-vina
```