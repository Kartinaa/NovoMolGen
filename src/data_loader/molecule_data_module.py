import os
import warnings
import json
from pathlib import Path
from typing import Union, Optional, List
import torch

import rootutils
import safe
import selfies as sf
from datasets import load_dataset, load_from_disk
from datasets.config import HF_CACHE_HOME
from datasets.naming import camelcase_to_snakecase
from deepsmiles import Converter
from molvs import standardize_smiles
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_loader.molecule_tokenizer import MoleculeTokenizer  # noqa


class MolDataModule:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        mol_type: Optional[str] = "SMILES",
        max_seq_length: int = 64,
        num_proc: int = 4,
        streaming: bool = False,
        validation_set_names: Optional[Union[str, List[str]]] = None,
        filter_validation_set: bool = False,
        # Extra conditioning options for collator
        include_condition_features: bool = False,  # Include pocket_vec, evo_vec, ifp, ligand_vec
        latent_dim: int = 128,
        device: Optional[Union[str, "torch.device"]] = None,
    ):
        """
        Args:
            dataset_name (str): Dataset name.
            tokenizer_path (str): Tokenizer path for local tokenizer.
            tokenizer_name (str): Tokenizer name.
            mol_type (str): Molecule type. Defaults to "SMILES".
            max_seq_length (int): Maximum sequence length. Defaults to 64.
            num_proc (int): Number of processes for tokenization and transferring mol_type.
            streaming (bool): Streaming. Defaults to False.
            validation_set_names (str or list): Validation set names.
            filter_validation_set (bool): To filter out the validation set in training set.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        self.mol_type = mol_type
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self.streaming = streaming
        self.num_invalid = 0
        self.validation_set_names = validation_set_names
        self.filter_validation_set = filter_validation_set
        self.include_condition_features = include_condition_features
        self.latent_dim = latent_dim
        # Collator returns CPU tensors by default; allow override
        try:
            import torch as _torch
            self.device = _torch.device(device) if device is not None else _torch.device("cpu")
        except Exception:
            self.device = None

        _tok_name = (
            Path(tokenizer_name).name
            if tokenizer_name is not None
            else Path(tokenizer_path).name # it will return the basename of the tokenizer path.
        )
        _dat_name = Path(dataset_name).name # return basename again
        _tok_name = _tok_name.replace(f"_{_dat_name}", "").replace(".json", "") # remove .json
        _dat_path = self.get_cache_dir(dataset_name)
        self.save_directory = os.path.join(
            HF_CACHE_HOME, "datasets", _dat_path, f"tokenized_{_tok_name}"
        )

        # Load tokenizer
        tokenizer = MoleculeTokenizer.load(tokenizer_path)
        self.tokenizer = tokenizer.get_pretrained()
        # To get rid of fast tokenizer warning, from: https://github.com/huggingface/transformers/issues/22638#issuecomment-1560406455
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        ### Prepare batches of tokenized data for language model training, `mlm=False` means 
        ### no masked language modeling (causal language modeling).
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Wrap with custom collate_fn to optionally add extra fields
        self.collate_fn = self._build_collate_fn()

        if validation_set_names is None:
            self.eval_dataset = None
            self.valid_set = set([])
        ### Don't understand this part, why validation set has split "train"?
        elif isinstance(validation_set_names, str):
            self.eval_dataset = load_dataset(
                validation_set_names, split="train", num_proc=num_proc
            )
            if self.filter_validation_set:
                # this set will use moe than 1GB of RAM
                self.valid_set = set(self.eval_dataset[self.mol_type])
        else:
            self.eval_dataset = {}
            for name in validation_set_names:
                ### `load_dataset()` is very flexible, which can load dataset from local directory 
                ### or from Hugging Face Hub.
                self.eval_dataset[Path(name).name] = load_dataset(
                    name, split="train", num_proc=num_proc
                )
            if self.filter_validation_set:
                # this set will use moe than 1GB of RAM
                self.valid_set = set()
                for v in self.eval_dataset.values():
                    self.valid_set = self.valid_set.union(set(v[self.mol_type]))

        self.prepare_eval_dataset()
        self.train_dataset = None

    def _build_collate_fn(self):
        """Wrap Hugging Face LM collator to optionally attach protein inputs and posterior params.

        Returns a function that takes a list[dict] and returns a dict of tensors on the configured device.
        """
        import torch

        base_collator = self.data_collator
        include_cond_features = self.include_condition_features
        target_device = self.device if self.device is not None else torch.device("cpu")

        def _collate(features: List[dict]):
            batch = base_collator(features)
            # Ensure expected dtypes
            if "input_ids" in batch:
                batch["input_ids"] = batch["input_ids"].to(target_device)
            if "attention_mask" in batch:
                # keep original int64 mask from collator; model may cast as needed
                batch["attention_mask"] = batch["attention_mask"].to(target_device)
            if "labels" in batch:
                batch["labels"] = batch["labels"].to(target_device)

            B = batch["input_ids"].size(0) if "input_ids" in batch else len(features)

            if include_cond_features:
                # Check if condition features are already in the dataset
                if all(key in features[0] for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]):
                    # Load real condition features from dataset
                    batch["pocket_vec"] = torch.stack([torch.tensor(f["pocket_vec"], dtype=torch.float32) for f in features]).to(target_device)
                    batch["evo_vec"] = torch.stack([torch.tensor(f["evo_vec"], dtype=torch.float32) for f in features]).to(target_device)
                    batch["ifp"] = torch.stack([torch.tensor(f["ifp"], dtype=torch.float32) for f in features]).to(target_device)
                    batch["ligand_vec"] = torch.stack([torch.tensor(f["ligand_vec"], dtype=torch.float32) for f in features]).to(target_device)
                else:
                    # Fallback: use random initialization if features not in dataset
                    print("Warning: Condition features not found in dataset, using random initialization")
                    batch["pocket_vec"] = torch.randn(B, 512, dtype=torch.float32, device=target_device)  # Uni-Mol pocket embedding
                    batch["evo_vec"] = torch.randn(B, 1280, dtype=torch.float32, device=target_device)    # ESM-2 evolutionary embedding
                    batch["ifp"] = torch.randn(B, 16384, dtype=torch.float32, device=target_device)      # Interaction fingerprint
                    batch["ligand_vec"] = torch.randn(B, 1536, dtype=torch.float32, device=target_device) # Ligand molecular representation

            return batch

        return _collate

    @staticmethod
    def make_dummy_batch(
        batch_size: int = 2,
        seq_len: int = 8,
        vocab_size: int = 100,
        include_condition_features: bool = False,
        device: Optional[Union[str, "torch.device"]] = "cuda",
        condition_data: Optional[dict] = None,
    ) -> dict:
        """Create a dummy batch for unit tests with optional extra fields.

        Args:
            batch_size: Number of samples in batch
            seq_len: Sequence length
            vocab_size: Vocabulary size
            include_condition_features: Whether to include condition features
            device: Target device
            condition_data: Optional dict with real condition data keys:
                - "pocket_vec": List of pocket vectors [batch_size, 512]
                - "evo_vec": List of evolutionary vectors [batch_size, 1280] 
                - "ifp": List of interaction fingerprints [batch_size, 16384]
                - "ligand_vec": List of ligand vectors [batch_size, 1536]

        Returns a dict with keys compatible with the model forward.
        """
        import torch

        dev = torch.device(device) if device is not None else torch.device("cpu")
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=dev)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=dev)
        labels = input_ids.clone()

        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        if include_condition_features:
            if condition_data is not None and all(key in condition_data for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]):
                # Use real condition data if provided
                batch["pocket_vec"] = torch.tensor(condition_data["pocket_vec"], dtype=torch.float32, device=dev)
                batch["evo_vec"] = torch.tensor(condition_data["evo_vec"], dtype=torch.float32, device=dev)
                batch["ifp"] = torch.tensor(condition_data["ifp"], dtype=torch.float32, device=dev)
                batch["ligand_vec"] = torch.tensor(condition_data["ligand_vec"], dtype=torch.float32, device=dev)
            else:
                # Fallback: Add random condition feature vectors for testing
                batch["pocket_vec"] = torch.randn(batch_size, 512, dtype=torch.float32, device=dev)
                batch["evo_vec"] = torch.randn(batch_size, 1280, dtype=torch.float32, device=dev)
                batch["ifp"] = torch.randn(batch_size, 16384, dtype=torch.float32, device=dev)
                batch["ligand_vec"] = torch.randn(batch_size, 1536, dtype=torch.float32, device=dev)

        return batch

    @staticmethod
    def get_cache_dir(dataset_name):
        """
        Get cache directory.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            str: Cache directory.
        """
        namespace_and_dataset_name = dataset_name.split("/")
        namespace_and_dataset_name[-1] = camelcase_to_snakecase(
            namespace_and_dataset_name[-1]
        )
        cached_relative_path = "___".join(namespace_and_dataset_name)
        return cached_relative_path

    def filter_smiles(self, example: dict):
        # Returns False if the SMILES is in the validation set, True otherwise
        if example[self.mol_type] != "":
            if self.filter_validation_set:
                res = example[self.mol_type] not in self.valid_set
            else:
                res = True
        else:
            self.num_invalid += 1
            res = False
        return res

    @staticmethod
    def tokenize_function(
        element: dict,
        max_length: int,
        mol_type: str,
        tokenizer: PreTrainedTokenizerFast,
    ) -> dict:
        """Tokenize a single element of the dataset.

        Args:
            element (dict): Dictionary with the data to be tokenized.
            max_length (int): Maximum length of the tokenized sequence.
            mol_type (str): mol_type of the dataset to be tokenized.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to be used.

        Returns:
            dict: Dictionary with the tokenized data.
        """
        outputs = tokenizer(
            element[mol_type],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
        )
        return {"input_ids": outputs["input_ids"]}

    @staticmethod
    def transfer_mol_type(
        element: dict,
        target_mol_type: str,
    ) -> dict:
        """Tokenize a single element of the dataset.

        Args:
            element (dict): Dictionary with the data.
            target_mol_type (str): mol_type of the dataset to be transfer to.

        Returns:
            dict: Converted molecule string.
        """
        if target_mol_type == "SELFIES":
            try:
                converted = sf.encoder(element["SMILES"])
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get selfies {e}", UserWarning)
                converted = ""
        elif target_mol_type == "SAFE":
            try:
                # TODO: check if ignore_stereo is needed
                converted = safe.encode(element["SMILES"], ignore_stereo=True)
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get safe {e}", UserWarning)
                converted = ""
        elif target_mol_type == "Deep SMILES":
            try:
                converter = Converter(rings=True, branches=True)
                converted = converter.encode(element["SMILES"])
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get deep smiles {e}", UserWarning)
                converted = ""
        elif target_mol_type == "SMILES":
            try:
                converted = standardize_smiles(element["SMILES"])
            except Exception as e:
                warnings.simplefilter("ignore")
                warnings.warn(f"Cannot get smiles {e}", UserWarning)
                converted = ""
        else:
            raise ValueError(f"mol_type {target_mol_type} not supported")
        return {f"{target_mol_type}": converted}

    def create_tokenized_datasets(self):
        """
        Create tokenized datasets and save them to disk.
        """
        assert self.streaming is False

        # Load dataset
        dataset = load_dataset(self.dataset_name, num_proc=self.num_proc, split="train")
        column_names = list(dataset.features)
        ### Check if Molecular type conversion needed.
        if self.mol_type not in column_names:
            column_names += [self.mol_type]
            # change mol_type to self.mol_type in dataset
            print("change mol_type from to", self.mol_type)
            ### `map()` is used to apply the first function (`transfer_mol_type()` in this case) 
            ### to each element of the dataset.
            dataset = dataset.map(
                self.transfer_mol_type,
                batched=False,
                num_proc=self.num_proc,
                fn_kwargs={
                    "target_mol_type": self.mol_type,
                },
            )

            dataset = dataset.filter(
                self.filter_smiles, batched=False, num_proc=self.num_proc
            )

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=self.num_proc,
            fn_kwargs={
                "max_length": self.max_seq_length,
                "mol_type": self.mol_type,
                "tokenizer": self.tokenizer,
            },
        )

        tokenized_dataset.save_to_disk(self.save_directory)
        print(f"tokenized dataset saved at: {self.save_directory}")

    def prepare_tokenized_streaming_dataset(self):
        """
        Prepare tokenized streaming dataset. If the dataset is already tokenized, it will be loaded from disk.
        Otherwise, it will be loaded from stream.

        Returns:
            Dataset: Tokenized streaming dataset.
        """
        if os.path.exists(self.save_directory):

            print(f'prepare tokenized streaming dataset from {self.save_directory}')
            dataset_stat_path = os.path.join(self.save_directory, 'state.json')
            with open(dataset_stat_path) as f:
                d = json.load(f)

            data_files = []
            for i, k in enumerate(d['_data_files']):
                data_files.append(os.path.join(self.save_directory, k['filename']))

            tokenized_dataset = load_dataset("arrow", data_files=data_files, streaming=True, split="train")
            print("prepare streaming dataset")
            return tokenized_dataset

        else:
            warnings.warn(
                "tokenized dataset didn't found locally.\nloading from stream."
            )
            dataset = load_dataset(self.dataset_name, split="train", streaming=True)
            column_names = list(dataset.features)
            if self.mol_type not in column_names:
                # change mol_type to self.mol_type in dataset
                print("change mol_type from to", self.mol_type)
                dataset = dataset.map(
                    self.transfer_mol_type,
                    fn_kwargs={
                        "target_mol_type": self.mol_type,
                    },
                )
                column_names += [self.mol_type]

            dataset = dataset.filter(self.filter_smiles)

            tokenized_dataset = dataset.map(
                self.tokenize_function,
                remove_columns=column_names,
                fn_kwargs={
                    "max_length": self.max_seq_length,
                    "mol_type": self.mol_type,
                    "tokenizer": self.tokenizer,
                },
            )
            print("prepare streaming dataset")
            return tokenized_dataset

    def load_tokenized_dataset(self):
        """
        Load tokenized dataset from disk if exists, otherwise create streaming dataset.

        Returns:
            Dataset: Tokenized dataset.
        """
        if self.streaming:
            self.train_dataset = self.prepare_tokenized_streaming_dataset()

        else:
            if not os.path.exists(self.save_directory):
                warnings.warn(
                    "tokenized dataset didn't found locally.\ncreating tokenized dataset may takes time."
                )
                self.create_tokenized_datasets()
            print(f'prepare tokenized dataset from {self.save_directory}')
            self.train_dataset = load_from_disk(self.save_directory)

    def _prepare_valid_set(self, _val_dataset):
        """
        Prepare validation set.

        Args:
            _val_dataset (Dataset): Validation dataset.

        Returns:
            Dataset: Prepared tokenized validation dataset.
        """
        column_names = list(_val_dataset.features)
        if self.mol_type not in column_names:
            print("change mol_type from to", self.mol_type)
            _val_dataset = _val_dataset.map(
                self.transfer_mol_type,
                batched=True,
                num_proc=self.num_proc,
                fn_kwargs={
                    "target_mol_type": self.mol_type,
                },
            )
            column_names += [self.mol_type]

        _val_dataset = _val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=self.num_proc,
            fn_kwargs={
                "max_length": self.max_seq_length,
                "mol_type": self.mol_type,
                "tokenizer": self.tokenizer,
            },
        )
        return _val_dataset

    def prepare_eval_dataset(self):
        """
        Prepare evaluation dataset.

        Returns:
            Dataset: Prepared tokenized evaluation dataset.
        """
        if type(self.eval_dataset) is dict:
            for key, val_dataset in self.eval_dataset.items():
                self.eval_dataset[key] = self._prepare_valid_set(val_dataset)
        else:
            self.eval_dataset = self._prepare_valid_set(self.eval_dataset)
