
import torch
import rootutils
import numpy as np
import types

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


@torch.inference_mode()
def _sample_molecules_hfmodel(
    self,
    tokenizer,
    batch_size: int = 16,
    prompt: str = "",
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    remove_spaces: bool = True,
    device: torch.device = torch.device("cuda"),
    **gen_kwargs,
):
    device = next(self.parameters()).device
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    assert pad is not None, "pad_token_id is not set"

    if prompt == "":
        base = torch.tensor([[bos if bos is not None else (eos if eos is not None else pad)]], device=device)
    else:
        base = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

    input_ids = base.expand(batch_size, -1).contiguous()

    gen_args = dict(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_length,
        eos_token_id=eos,
        pad_token_id=pad,
        return_dict_in_generate=True,
    )
    gen_args.update(gen_kwargs)

    out = self.generate(input_ids=input_ids, **gen_args)
    seqs = out.sequences

    texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)

    result = {
            f'{self.mol_type}': [t.replace(" ", "") if remove_spaces else t for t in texts],
            "sequences": seqs.detach(),
        }
    return result

def prepare_hf_model(model, mol_type: str = "SMILES"):
    model.sample = types.MethodType(_sample_molecules_hfmodel, model)
    model.mol_type = mol_type
    return model