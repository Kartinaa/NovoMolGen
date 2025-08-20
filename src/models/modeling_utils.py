
import torch
import rootutils
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_loader import safe_to_smiles, selfies_to_smiles, deepsmiles_to_smiles
from src.data_loader.molecule_tokenizer import MoleculeTokenizer
from src.models.modeling_novomolgen import NovoMolGen
from src.eval.utils import mapper
from src.eval.components.moses import canonic_smiles


@torch.inference_mode
def generate_valid_smiles(
        model: NovoMolGen,
        tokenizer: MoleculeTokenizer,
        batch_size: int = 4,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        device: torch.device = torch.device("cuda"),
        return_canonical_unique: bool = False,
        ):
    
    outputs = model.sample(
        tokenizer=tokenizer, 
        batch_size=batch_size, 
        max_length=max_length, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        device=device
        )
    decoded_strings = outputs[model.mol_type]
    sequences = outputs['sequences']

    converted_sequences = []  # always SMILES or kept as-is if mol_type == SMILES
    main_mol_sequences = []  # the raw molecule strings in original format
    valid_indices = []

    for i, raw_str in enumerate(decoded_strings):
        if model.mol_type == "SMILES":
            # Keep all SMILES unchanged
            converted_sequences.append(raw_str)
            # For SMILES, "main_mol_sequences" is effectively the same
            main_mol_sequences.append(raw_str)
            valid_indices.append(i)

        elif model.mol_type == "SELFIES":
            smiles = selfies_to_smiles(raw_str)
            if smiles:
                converted_sequences.append(smiles)
                main_mol_sequences.append(raw_str)
                valid_indices.append(i)

        elif model.mol_type == "SAFE":
            smiles = safe_to_smiles(raw_str)
            if smiles:
                converted_sequences.append(smiles)
                main_mol_sequences.append(raw_str)
                valid_indices.append(i)

        elif model.mol_type == "Deep SMILES":
            smiles = deepsmiles_to_smiles(raw_str)
            if smiles:
                converted_sequences.append(smiles)
                main_mol_sequences.append(raw_str)
                valid_indices.append(i)

        else:
            raise NotImplementedError(
                f"Molecule type '{model.mol_type}' is not supported."
            )

    if len(valid_indices) > 0:
        valid_idx_tensor = torch.tensor(valid_indices, device=sequences.device)
        filtered_token_ids = sequences[valid_idx_tensor]
    else:
        # No valid sequences
        filtered_token_ids = torch.empty((0, sequences.shape[1]),
                                            dtype=sequences.dtype,
                                            device=sequences.device)

    # Always return SMILES + filtered tokens
    result = {
        "SMILES": converted_sequences,
        "sequences": filtered_token_ids,
    }

    # If mol_type is not SMILES, add the raw main moltype strings
    if model.mol_type != "SMILES":
        result[model.mol_type] = main_mol_sequences

    if return_canonical_unique:

        generated_smiles = result['SMILES']
        canonic_generated_smiles = mapper(1)(canonic_smiles, generated_smiles)
        unique_generated_smiles = []
        for s in canonic_generated_smiles:
            if s is not None and s not in unique_generated_smiles:
                unique_generated_smiles.append(s)
        if len(unique_generated_smiles) == 0:
            return {'SMILES': [], 'sequences': torch.empty(0, dtype=torch.long, device=device)}

        seen = set()
        boolean_mask = []
        unique_idx = []
        for i,s in enumerate(canonic_generated_smiles):
            if s in unique_generated_smiles and s not in seen:
                boolean_mask.append(True)
                unique_idx.append(i)
                seen.add(s)
            else:
                boolean_mask.append(False)

        unique_generated_smiles = np.array(generated_smiles)[unique_idx].tolist()
        sequences = sequences[torch.tensor(unique_idx)]
        
        result = {
            "SMILES": unique_generated_smiles,
            "sequences": sequences,
        }
        if model.mol_type != "SMILES":
            result[model.mol_type] = np.array(main_mol_sequences)[unique_idx].tolist()

    assert len(result['SMILES']) == len(result['sequences'])
    
    return result