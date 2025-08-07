import os
import pickle
import warnings

import numpy as np
import rootutils
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set up the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class DrugPredictor:
    """Given a SMILES string, predict the drug-likeness score of the molecule."""

    def __init__(self):
        """Initializes the DrugPredictor object."""
        # Load the dbpp model
        with open("./data/dbpp/dbpp_predictor.model", "rb") as model_file:
            self.dbpp_model = pickle.load(model_file)

        # Load the ADMET models
        self.model_names = [
            "HIA_svm_Morgan.model",
            "HOB_svm_Morgan.model",
            "P-pgi_svm_Morgan.model",
            "P-pgs_svm_Morgan.model",
            "Caco2_svm_Morgan.model",
            "BCRPi_svm_Morgan.model",
            "BSEPi_svm_Morgan.model",
            "OCT2i_svm_Morgan.model",
            "OATP1B1i_svm_Morgan.model",
            "OATP1B3i_svm_Morgan.model",
            "CL_svm_Morgan.model",
            "MMP_svm_Morgan.model",
            "hERG_svm_Morgan.model",
            "Ames_svm_Morgan.model",
            "Repro_svm_Morgan.model",
            "Carc_svm_Morgan.model",
            "Gene_svm_Morgan.model",
            "DILI_svm_Morgan.model",
            "Kidney_svm_Morgan.model",
            "ROA_svm_Morgan.model",
        ]

        self.admet_models = {}
        for model_name in self.model_names:
            with open(os.path.join("./data/dbpp/admet_models/", model_name), "rb") as model_file:
                self.admet_models[model_name] = pickle.load(model_file)

    def endpoint(self, model, fps: np.ndarray) -> float:
        """Predicts the endpoint score of a given compound.

        :param model: Model to use for prediction
        :param fps: Fingerprint of the compound
        :return: Endpoint score of the compound
        """
        score = model.predict_proba(fps)[:, 1]
        return score

    def ecfp(self, mol: Mol) -> np.ndarray:
        """Generates the ECFP fingerprint of a given compound.

        :param mol: RDKit molecule object
        :return: ECFP fingerprint of the compound
        """
        fingerprint = GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
        fps = np.asarray(list(fingerprint)).astype("int").reshape(1, -1)
        return fps

    def predict(self, smi: str, mol: Mol) -> float:
        """Predicts the drug-likeness score of a given compound.

        :param smi: SMILES of the compound
        :param mol: RDKit molecule object
        :return: Drug-likeness score of the compound
        """
        generator = MakeGenerator(("rdkit2dnormalized",))
        feature = generator.process(smi)  # type: ignore
        features = feature[1:]  # type: ignore
        property_scores = np.asarray(features)[[46, 48, 57, 58, 61, 103]].reshape(1, -1)
        fps = self.ecfp(mol)
        admet_scores = [
            self.endpoint(self.admet_models[model_name], fps) for model_name in self.model_names
        ]
        admet_scores = np.asarray(admet_scores).reshape(1, -1)
        property_profile = np.concatenate((property_scores, admet_scores), axis=1)
        score = self.dbpp_model.predict_proba(property_profile)[:, 1][0]
        return score
