from safe import SAFEConverter
import safe as sf
from rdkit import Chem
encoder = SAFEConverter(ignore_stereo=True)

# Purine scaffold with attachment point
scaffold_smiles = '*C1NC2NCC(*)C(N3CCN(*)CC3)C2N1'


# 先检查 SMILES 是否有效
mol = Chem.MolFromSmiles(scaffold_smiles)
if mol is None:
    print(f"Invalid SMILES: {scaffold_smiles}")
    print("Trying to sanitize...")
    
    # 尝试不同的选项
    mol = Chem.MolFromSmiles(scaffold_smiles, sanitize=False)
    print(f"If without sanitize, the mol is None: {mol is None}")
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            scaffold_smiles = Chem.MolToSmiles(mol)
            print(f"Sanitized SMILES: {scaffold_smiles}")
        except Exception as e:
            print(f"Sanitization failed: {e}")
else:
    print(f"Valid SMILES: {scaffold_smiles}")

# Encode with fragmentation disabled (as done in scaffold_decoration)
with sf.utils.attr_as(encoder, 'slicer', None):
    encoded = encoder.encoder(scaffold_smiles, allow_empty=True)

print(f'Original:  {scaffold_smiles}')  # O=c1[nH]cnc2nc([*])ccc12
print(f'Encoded:   {encoded}')           # O=c1[nH]cnc2nc3ccc12  ← [*] became 3!
print(f'Decoded:   {encoder.decoder(encoded, remove_dummies=False)}')  # O=c1[nH]cnc2nc([*:3])ccc12 

from safe.utils import standardize_attach
print(f'Decoded:   {standardize_attach(encoder.decoder(encoded, remove_dummies=False))}')