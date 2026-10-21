from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import torch
import esm
from Bio.PDB import PDBParser, Polypeptide
from Bio.Data import IUPACData

# Basic 3->1 dictionary (unified to uppercase keys)
AA3_TO1 = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}

# Common variants and aliases supplement
AA3_TO1.update({
    "MSE": "M",   # Selenomethionine -> treat as Met
    "SEC": "U",   # Selenocysteine
    "PYL": "O",   # Pyrrolysine
    # Histidine tautomer/PK forms seen in some force fields/modeling software
    "HSD": "H", "HSE": "H", "HSP": "H", "HIP": "H", "HID": "H", "HIE": "H",
    # Cys variants
    "CYM": "C", "CYX": "C",
})

def aa3_to1_safe(resname: str) -> str | None:
    """Convert three-letter to one-letter; return None instead of raising error if unrecognized."""
    name = resname.strip().upper()
    # First try dictionary
    if name in AA3_TO1:
        return AA3_TO1[name]
    # Then try biopython's built-in converter
    try:
        return Polypeptide.three_to_one(name)
    except Exception:
        return None

def _is_std_aa(residue) -> bool:
    """Check if it's a standard amino acid: filter out water/ligands/heteroatoms, only accept residues that can map to 1-letter."""
    het, resseq, icode = residue.id
    if het.strip() != "":           # Non-empty means HETATM (including water HOH, ions, ligands, etc.)
        return False
    aa1 = aa3_to1_safe(residue.get_resname())
    return aa1 is not None

def _res_uid(residue):
    """Uniquely identify a residue using (resseq, icode), compatible with insertion codes."""
    het, resseq, icode = residue.id
    return (int(resseq), (icode or " ").strip() or " ")

class ESM2PocketEmbedder:
    """
    - Extract single chain sequence from PDB → ESM-2 residue embeddings (without BOS/EOS)
    - Return corresponding embedding subset based on residue set in pocket PDB
    """
    def __init__(self,
                 model_name: str = "esm2_t33_650M_UR50D",
                 repr_layer: int = 33,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.repr_layer = repr_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Official API: load model and alphabet in one go
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def _get_chain(self, structure, chain_id: Optional[str]):
        model0 = next(structure.get_models())
        if chain_id is None:
            return next(model0.get_chains())  # Default to first chain
        return model0[chain_id]

    def extract_residue_embeddings(self,
                                   pdb_path: str,
                                   chain_id: Optional[str] = None
                                   ) -> Tuple[List[Tuple[int, str, str]], torch.Tensor]:
        """
        Returns:
          - res_list: [(resseq, icode, aa1), ...] in sequence order
          - emb: [L, H]  one vector per residue (without BOS/EOS)
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        chain = self._get_chain(structure, chain_id)

        seq_chars: List[str] = []
        res_list: List[Tuple[int, str, str]] = []  # (resseq, icode, aa1)
        for res in chain.get_residues():
            if not _is_std_aa(res):
                continue
            key = _res_uid(res)
            aa3 = res.get_resname().upper()
            aa1 = aa3_to1_safe(aa3)
            if aa1 is None:
                print(f'not standard aa {aa3}')
                break
            seq_chars.append(aa1)
            res_list.append((key[0], key[1], aa1))

        if len(seq_chars) == 0:
            raise ValueError("No standard amino acids found in the specified chain.")
        seq = "".join(seq_chars)

        data = [("protein", seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]  # [1, L+2, H]
        L = len(seq)
        per_res = reps[0, 1:1+L, :].detach().cpu()      # [L, H]
        return res_list, per_res

    def pocket_embeddings(self,
                          full_pdb_path: str,
                          pocket_pdb_path: str,
                          chain_id: Optional[str] = None
                          ) -> Tuple[List[Tuple[int, str, str]], torch.Tensor]:
        """
        Based on full PDB embeddings, return embeddings for residues in pocket PDB (in pocket file residue order).
        Returns:
          - pocket_res_list: [(resseq, icode, aa1), ...]
          - pocket_emb: [K, H]
        Note: Aligned via (resseq, icode), single chain assumption; non-standard residues in pocket will be skipped.
        """
        # 1) Full protein residue embeddings
        full_res_list, full_emb = self.extract_residue_embeddings(full_pdb_path, chain_id=chain_id)
        index_by_uid: Dict[Tuple[int, str], int] = {
            (r[0], r[1]): i for i, r in enumerate(full_res_list)
        }

        # 2) Read pocket residue UIDs (in pocket PDB order)
        parser = PDBParser(QUIET=True)
        pocket_struct = parser.get_structure("pocket", pocket_pdb_path)
        pocket_chain = self._get_chain(pocket_struct, chain_id)

        pocket_keys: List[Tuple[int, str]] = []
        pocket_res_list: List[Tuple[int, str, str]] = []
        for res in pocket_chain.get_residues():
            if not _is_std_aa(res):
                continue
            uid = _res_uid(res)
            aa3 = res.get_resname().upper()
            aa1 = aa3_to1_safe(aa3)
            if aa1 is None:
                print(f'not standard aa {aa3}')
                break
            pocket_keys.append(uid)
            pocket_res_list.append((uid[0], uid[1], aa1))

        if len(pocket_keys) == 0:
            raise ValueError("No standard amino acids found in pocket PDB.")

        # 3) Select subset embeddings
        idxs: List[int] = []
        missing: List[Tuple[int, str]] = []
        for uid in pocket_keys:
            if uid in index_by_uid:
                idxs.append(index_by_uid[uid])
            else:
                missing.append(uid)

        if missing:
            # Here we choose: ignore missing and only return existing ones; you can also change to strict alignment and raise error
            # raise KeyError(f"Pocket residues not found in full protein embedding: {missing}")
            pass

        if len(idxs) == 0:
            raise RuntimeError("No overlapping pocket residues found in full protein embeddings.")

        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        pocket_emb = full_emb.index_select(0, idx_tensor)  # [K, H]
        return pocket_res_list[:len(idxs)], pocket_emb



### How to use the class:

# esm_2_embedder = ESM2PocketEmbedder(
#     model_name="esm2_t33_650M_UR50D",
#     repr_layer=33,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )


# full_res_list, full_emb = esm_2_embedder.extract_residue_embeddings('/home/yang2531/Documents/Project/GNN_VAE_draft/data/small_frag/high/target_CHEMBL202/CHEMBL22/protein.pdb', chain_id=None)
# print(len(full_res_list), full_emb.shape)  # e.g. (L, [L, 1280])

# pocket_res_list, pocket_emb = esm_2_embedder.pocket_embeddings('/home/yang2531/Documents/Project/GNN_VAE_draft/data/small_frag/high/target_CHEMBL202/CHEMBL22/protein.pdb', 
#                                                                "/home/yang2531/Documents/Project/GNN_VAE_draft/data/small_frag/high/target_CHEMBL202/CHEMBL22/protein_6A.pdb", 
#                                                                chain_id=None)
# print(len(pocket_res_list), pocket_emb.shape)  # K, [K, 1280]

# pocket_vec = pocket_emb.mean(dim=0)  # [H]
# print(pocket_vec)
# pocket_vec.unsqueeze(0).shape

if __name__ == "__main__":
    print('test')