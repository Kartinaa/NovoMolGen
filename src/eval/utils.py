import subprocess
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
import math
from subprocess import DEVNULL

from multiprocess.pool import Pool
from rdkit import Chem, rdBase


@contextmanager
def suppress_output(verbose):
    """Suppress output when."""
    if verbose:
        pass
    else:
        with open(devnull, "w") as fnull:
            with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                yield (err, out)


def run_command(command, verbose):
    if verbose:
        subprocess.run(command, shell=True)
    else:
        subprocess.run(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)


def mapper(n_jobs):
    """Returns function for map call.

    If n_jobs == 1, will use standard map If n_jobs > 1, will use multiprocessing pool If n_jobs is
    a pool object, will return its map function
    """
    if n_jobs == 1:

        def _mapper(*args, **kwargs):  # type: ignore
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def disable_rdkit_log():
    rdBase.DisableLog("rdApp.*")


def enable_rdkit_log():
    rdBase.EnableLog("rdApp.*")


def get_mol(smiles_or_mol):
    """Loads SMILES/molecule into RDKit's object."""
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    elif isinstance(smiles_or_mol, float) and math.isnan(smiles_or_mol):
        return None
    return smiles_or_mol
