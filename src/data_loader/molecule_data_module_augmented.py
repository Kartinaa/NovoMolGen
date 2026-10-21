"""
Augmented Molecule Data Module for SAFE data augmentation.

This module extends MolDataModule to support pre-computed SAFE representations.
When the dataset contains a 'SAFE' column (from augmented datasets), it uses
the pre-computed value instead of computing SAFE encoding on-the-fly.

This is backward-compatible: if no 'SAFE' column exists, it falls back to
the original behavior of computing SAFE from SMILES.
"""

import warnings
import safe

from src.data_loader.molecule_data_module import MolDataModule


class MolDataModuleAugmented(MolDataModule):
    """
    Molecule Data Module with support for pre-computed SAFE representations.

    This class extends MolDataModule to use pre-computed SAFE strings when available
    in the dataset (e.g., from augmented datasets created by 10_augment_safe_representations.py).

    The only change from MolDataModule is in transfer_mol_type(), which checks for
    a pre-computed 'SAFE' column before computing SAFE encoding.
    """

    @staticmethod
    def transfer_mol_type(
        element: dict,
        target_mol_type: str,
    ) -> dict:
        """Convert molecule representation to target type.

        This version supports pre-computed SAFE representations from augmented datasets.

        Args:
            element: Dictionary with the data (must contain 'SMILES', may contain 'SAFE')
            target_mol_type: Target molecule type ('SAFE', 'SELFIES', 'Deep SMILES', 'SMILES')

        Returns:
            Dictionary with the converted molecule string
        """
        if target_mol_type == "SELFIES":
            import selfies as sf
            try:
                converted = sf.encoder(element["SMILES"])
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get selfies {e}", UserWarning)
                converted = ""
        elif target_mol_type == "SAFE":
            # Use pre-computed SAFE if available (from augmented dataset)
            if "SAFE" in element and element["SAFE"]:
                converted = element["SAFE"]
            else:
                # Fall back to computing canonical SAFE
                try:
                    converted = safe.encode(element["SMILES"], ignore_stereo=True)
                except Exception as e:
                    warnings.simplefilter("ignore")
                    warnings.warn(f"Cannot get safe {e}", UserWarning)
                    converted = ""
        elif target_mol_type == "Deep SMILES":
            from deepsmiles import Converter
            try:
                converter = Converter(rings=True, branches=True)
                converted = converter.encode(element["SMILES"])
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get deep smiles {e}", UserWarning)
                converted = ""
        elif target_mol_type == "SMILES":
            from molvs import standardize_smiles
            try:
                converted = standardize_smiles(element["SMILES"])
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get smiles {e}", UserWarning)
                converted = ""
        else:
            raise ValueError(f"mol_type {target_mol_type} not supported")
        return {f"{target_mol_type}": converted}
