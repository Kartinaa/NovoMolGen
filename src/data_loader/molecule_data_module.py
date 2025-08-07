import os
import warnings
import json
from pathlib import Path
from typing import Union, Optional, List

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

        _tok_name = (
            Path(tokenizer_name).name
            if tokenizer_name is not None
            else Path(tokenizer_path).name
        )
        _dat_name = Path(dataset_name).name
        _tok_name = _tok_name.replace(f"_{_dat_name}", "").replace(".json", "")
        _dat_path = self.get_cache_dir(dataset_name)
        self.save_directory = os.path.join(
            HF_CACHE_HOME, "datasets", _dat_path, f"tokenized_{_tok_name}"
        )

        # Load tokenizer
        tokenizer = MoleculeTokenizer.load(tokenizer_path)
        self.tokenizer = tokenizer.get_pretrained()
        # To get rid of fast tokenizer warning, from: https://github.com/huggingface/transformers/issues/22638#issuecomment-1560406455
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        if validation_set_names is None:
            self.eval_dataset = None
            self.valid_set = set([])
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
        if self.mol_type not in column_names:
            column_names += [self.mol_type]
            # change mol_type to self.mol_type in dataset
            print("change mol_type from to", self.mol_type)
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
