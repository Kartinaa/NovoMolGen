import safe as sf
import datamol as dm
designer = sf.SAFEDesign.load_default(verbose=True)

scaffold_smiles = '*C1CCC2C(*)CC3NNC(=O)N3C2C1'

generated_smiles = designer.scaffold_decoration(
    scaffold=scaffold_smiles, # using the scaffold as input directly
    n_samples_per_trial=5,
    n_trials=1,
    sanitize=True,
    do_not_fragment_further=True,
)

print(generated_smiles)
# generated_mols = [dm.to_mol(x) for x in generated_smiles]
# dm.to_image(generated_mols)