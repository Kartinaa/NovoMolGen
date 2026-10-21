import os
import random
import numpy as np
import subprocess
from dataclasses import dataclass, field
from multiprocessing import Manager, Process, Queue, set_start_method
from shutil import rmtree
from typing import Any, Dict, List, Tuple
from rdkit import Chem
from tqdm import tqdm
from loguru import logger
from rdkit.Chem import AllChem

import rootutils
from openbabel import pybel

# Set up the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.components.filter_ import process_molecule  # noqa

# It's often a good practice to set the start method explicitly,
# especially to 'spawn' on Unix-like systems where 'fork' is the default.
# This can help avoid some issues related to process forking.
try:
    set_start_method("spawn")
except RuntimeError:
    # Ignore the error if the start method has already been set.
    pass


class TqdmToLoguru:
    def __init__(self, logger, level="INFO"):
        self.logger = logger
        self.level = level

    def write(self, msg):
        msg = msg.strip()  # Remove any trailing whitespace
        if msg:
            self.logger.log(self.level, msg)

    def flush(self):
        pass  # Required to have a `flush()` for compatibility with `tqdm`



@dataclass
class DockingConfig:
    target_name: str
    vina_program: str = "qvina"
    temp_dir: str = field(default_factory=lambda: DockingConfig.make_docking_dir())
    exhaustiveness: int = 1
    num_sub_proc: int = 8
    num_cpu_dock: int = 1
    num_modes: int = 10
    maxAttempts: int = 1000  # default for rdkit is 200, we increase it for better embeddings
    timeout_gen3d: int = 30
    timeout_dock: int = 50
    seed: int = 42
    receptor_file: str = field(init=False)
    box_parameter: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(
        init=False
    )

    def __post_init__(self):
        self.receptor_file = f"./data/docking/receptors/{self.target_name}/receptor.pdbqt"
        self.vina_program = f"./data/docking/{self.vina_program}"
        self.box_parameter = self.get_box_parameters()

    def get_box_parameters(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        box_parameters = {
            "fa7": {"center": (10.131, 41.879, 32.097), "size": (20.673, 20.198, 21.362)},
            "parp1": {"center": (26.413, 11.282, 27.238), "size": (18.521, 17.479, 19.995)},
            "5ht1b": {"center": (-26.602, 5.277, 17.898), "size": (22.5, 22.5, 22.5)},
            "jak2": {"center": (114.758, 65.496, 11.345), "size": (19.033, 17.929, 20.283)},
            "braf": {"center": (84.194, 6.949, -7.081), "size": (22.032, 19.211, 14.106)},
            "3pbl": {"center": (9, 22.5, 26), "size": (15, 15, 15)},
            "1iep": {"center": (15.613, 53.380, 15.454), "size": (15, 15, 15)},
            "2rgp": {
                "center": (16.292, 34.870, 92.035),
                "size": (15, 15, 15),
            },
            "3eml": {
                "center": (-9.063, -7.144, 55.862),
                "size": (15, 15, 15),
            },
            "3ny8": {"center": (2.248, 4.684, 51.398), "size": (15, 15, 15)},
            "4rlu": {
                "center": (-0.735, 22.755, -31.236),
                "size": (15, 15, 15),
            },
            "4unn": {
                "center": (5.684, 18.191, -7.371),
                "size": (15, 15, 15),
            },
            "5mo4": {
                "center": (-44.901, 20.490, 8.483),
                "size": (15, 15, 15),
            },
            "7l11": {
                "center": (-21.814, -4.216, -27.983),
                "size": (15, 15, 15),
            },
            "1err": {"center": (67.225, 33.534, 72.22), "size": (20, 20, 20)},
            "2iik": {"center": (-0.001, 3.656, -16.404), "size": (20, 20, 20)},
        }
        return box_parameters[self.target_name]

    @staticmethod
    def make_docking_dir():
        for i in range(100):
            tmp_dir = f"tmp/tmp{i}"
            if not os.path.exists(tmp_dir):
                print(f"Docking tmp dir: {tmp_dir}")
                os.makedirs(tmp_dir)
                return tmp_dir
        raise ValueError("tmp/tmp0~99 are full. Please delete tmp dirs.")


class Docking:
    """Python class for docking using Vina."""

    def __init__(self, docking_params: Any) -> None:
        """Initialize the DockingVina class.

        :param docking_params: Dictionary containing the docking parameters
        :raises NotImplementedError: If the target name is not implemented
        """

        super().__init__()

        self.temp_dir = docking_params.temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.vina_program = docking_params.vina_program
        self.receptor_file = docking_params.receptor_file
        box_parameter = docking_params.box_parameter
        (self.box_center, self.box_size) = box_parameter["center"], box_parameter["size"]
        self.exhaustiveness = docking_params.exhaustiveness
        self.num_sub_proc = docking_params.num_sub_proc
        self.num_cpu_dock = docking_params.num_cpu_dock
        self.num_modes = docking_params.num_modes
        self.timeout_gen3d = docking_params.timeout_gen3d
        self.timeout_dock = docking_params.timeout_dock
        self.seed = docking_params.seed
        self.maxAttempts = docking_params.maxAttempts

    def gen_3d(self, smi: str, ligand_mol_file: str) -> None:
        """Generate 3D structure of a molecule using OpenBabel.

        :param smi: Input SMILES string
        :param ligand_mol_file: Output molecule file
        """
        run_line = f"obabel -:{smi} --gen3D -O {ligand_mol_file} --fast -c --ff MMFF94 --steps 50"
        result = subprocess.check_output(  # noqa
            run_line.split(),
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            text=True,
        )

    def gen_3d_with_rdkit(self, smi: str, ligand_mol_file: str, seed: int) -> None:
        """Generate 3D structure of a molecule using RDKit."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Set up ETKDG parameters for better embedding
        params = AllChem.ETKDG()
        params.randomSeed = seed
        params.maxAttempts = self.maxAttempts  # Increase max attempts to improve success rate
        params.numThreads = 1  # Ensure reproducibility by using a single thread

        # Embed the molecule
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            raise ValueError(f"Failed to generate 3D coordinates for molecule: {smi}")

        # Optimize the molecule with MMFF
        AllChem.MMFFOptimizeMolecule(mol)

        # Write to the desired file format
        with Chem.SDWriter(ligand_mol_file) as writer:
            writer.write(mol)

    def docking(
            self,
            receptor_file: str,
            ligand_pdbqt_file: str,
            docking_pdbqt_file: str,
            seed: int,
    ) -> List[float]:
        """Docking using a given docking program.

        :param receptor_file: Receptor file
        :param ligand_pdbqt_file: Ligand PDBQT file
        :param docking_pdbqt_file: Docking PDBQT file
        :param seed: seed
        :return: List of affinity scores
        """
        run_line = "{} --receptor {} --ligand {} --out {}".format(
            self.vina_program,
            receptor_file,
            ligand_pdbqt_file,
            docking_pdbqt_file,
        )
        run_line += " --center_x %s --center_y %s --center_z %s" % (self.box_center)
        run_line += " --size_x %s --size_y %s --size_z %s" % (self.box_size)
        run_line += " --cpu %d" % (self.num_cpu_dock)
        run_line += " --num_modes %d" % (self.num_modes)
        run_line += " --exhaustiveness %d " % (self.exhaustiveness)
        run_line += " --seed %d " % (seed)
        result = subprocess.check_output(
            run_line.split(),
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            text=True,
        )
        result_lines = result.split("\n")

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith("-----+"):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith("Writing output"):
                break
            if result_line.startswith("Refine time"):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            #            mode = int(lis[0])
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q: Queue, data: List[Tuple[int, str]], num_sub_proc: int):
        """Create the subprocesses for docking.

        :param q: Queue for subprocesses
        :param data: List of data
        :param num_sub_proc: Number of subprocesses
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put("DONE")

    def docking_subprocess(self, q: Queue, return_dict: Dict[int, float], sub_id: int = 0):
        """Subprocess for docking.

        :param q: Queue for subprocesses
        :param return_dict: Dictionary for storing the results
        :param sub_id: Subprocess ID, defaults to 0
        """
        seed = self.seed  # + sub_id
        random.seed(seed)
        np.random.seed(seed)

        while True:
            qqq = q.get()
            if qqq == "DONE":
                break
            (idx, smi) = qqq
            pass_filt = process_molecule(smi)
            if pass_filt[1] == "Fail":
                return_dict[idx] = 10000
                continue

            receptor_file = self.receptor_file
            ligand_mol_file = f"{self.temp_dir}/ligand_{sub_id}.mol"
            ligand_pdbqt_file = f"{self.temp_dir}/ligand_{sub_id}.pdbqt"
            docking_pdbqt_file = f"{self.temp_dir}/dock_{sub_id}.pdbqt"

            # Generate 3D structure
            try:
                # self.gen_3d(smi, ligand_mol_file)
                self.gen_3d_with_rdkit(smi, ligand_mol_file, seed=seed)
            except Exception as e:
                logger.error(f"gen_3d unexpected error:{e}")
                return_dict[idx] = 10000
                continue

            ms = list(pybel.readfile("mol", ligand_mol_file))
            m = ms[0]
            m.write("pdbqt", ligand_pdbqt_file, overwrite=True)

            # Check the quality of the generated structure
            try:
                ob_cmd = ["obenergy", ligand_pdbqt_file]
                command_obabel_check = subprocess.run(ob_cmd, capture_output=True)
                command_obabel_check = command_obabel_check.stdout.decode("utf-8").split("\n")[-2]
                _ = float(command_obabel_check.split(" ")[-2])
            except Exception as e:
                logger.error(f"Energy calculation unexpected error:{e}")
                return_dict[idx] = 10000
                continue

            # Docking
            try:
                affinity_list = self.docking(
                    receptor_file,
                    ligand_pdbqt_file,
                    docking_pdbqt_file,
                    seed=seed,
                )
            except Exception as e:
                logger.error(f"Docking unexpected error:{e}")
                return_dict[idx] = 10000
                continue
            if len(affinity_list) == 0:
                affinity_list.append(10000)

            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def __call__(self, smiles_list: List[str]) -> List[float]:
        """Call the DockingVina class.

        :param smiles_list: List of SMILES strings
        :return: List of affinity scores
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator, args=(q1, data, self.num_sub_proc))
        proc_master.start()

        # Create slave process
        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess, args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()

        # Set up a progress bar integrated with the logger
        tqdm_log_stream = TqdmToLoguru(logger, level="INFO")
        with tqdm(total=len(data), desc="Docking Progress", unit="molecule", file=tqdm_log_stream) as pbar:
            # Continuously check the progress of `return_dict` until all processes are complete
            while len(return_dict) < len(data):
                pbar.update(len(return_dict) - pbar.n)

        for proc in procs:
            proc.join()

        # Handling potential issues with process not terminating
        for proc in procs + [proc_master]:
            if proc.is_alive():
                logger.warning(f"Process {proc.pid} did not terminate. Terminating now.")
                proc.terminate()

        # Clean up the temp directory
        rmtree(self.temp_dir)

        keys = sorted(return_dict.keys())
        affinity_list = [return_dict[key] for key in keys]

        return affinity_list


if __name__ == "__main__":
    docking_cfg = DockingConfig(target_name="fa7", num_sub_proc=8)
    target = Docking(docking_cfg)
    affinity_list = target(["CCO", "CCN"])
    print(affinity_list)
