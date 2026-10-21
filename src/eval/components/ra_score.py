import pickle

import numpy as np
import rootutils
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Set up the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class RAScorerXGB:
    """Prediction of machine learned retrosynthetic accessibility score The RAScore is calculated
    based on the predictions made on 200,000 compounds sampled from ChEMBL. The compounds were
    subjected to retrosynthetic analysis using a CASP tool (AiZynthfinder) and output used as
    labels to train a binary classifier.

    If the compounds are ChEMBL like use the RAscore models. Else if the compounds are more exotic,
    are small fragments, or closely resemble GDB. GBDscore may give a better result.

    This class facilitates predictions from the resulting model.
    """

    def __init__(self, dataset: str = "gdbchembl"):
        """Initializes the RAScorerXGB object.

        :param dataset: The dataset to used for training the model. Options are 'chembl',
            'gdbchembl' or 'gdbmedchem'
        """
        self.xgb_model = pickle.load(
            open(f"./data/rascore/XGB_{dataset}_ecfp_counts/model.pkl", "rb")
        )

    def ecfp(self, mol: Mol) -> np.ndarray:
        """Generates the ECFP fingerprint of a given compound.

        :param mol: RDKit molecule object
        :return: ECFP fingerprint of the compound
        """
        fingerprint = GetMorganFingerprintAsBitVect(mol, 3).ToBitString()
        fps = np.asarray(list(fingerprint)).astype("int")
        return fps

    def predict(self, mol: Mol) -> float:
        """Predicts the retrosynthetic accessibility score of a given compound.

        :param mol: RDKit molecule object
        :return: RAScore of the compound
        """
        arr = self.ecfp(mol)
        proba = self.xgb_model.predict_proba(arr.reshape(1, -1))
        return proba[0][1]
