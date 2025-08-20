"""Code for molecule tokenization.

Adapted from https://github.com/datamol-io/safe/blob/main/safe/tokenizer.py

SMILES regex pattern for tokenization. Designed by Schwaller et. al.

References
----------
.. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
1572-1583 DOI: 10.1021/acscentsci.9b00576
"""

from argparse import ArgumentParser
from typing import Dict, Optional

import rootutils
from datasets import load_dataset
from safe import SAFETokenizer
from tokenizers import Regex, Tokenizer, decoders
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import (
    BpeTrainer,
    UnigramTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
)
from transformers import PreTrainedTokenizerFast

# Set up the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

UNK_TOKEN = "<unk>"
PADDING_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [UNK_TOKEN, PADDING_TOKEN, BOS_TOKEN, EOS_TOKEN]


class MoleculeTokenizer(SAFETokenizer):
    """Class to initialize and train a tokenizer for molecule string.

    Once trained, you can use the converted version of the tokenizer to an HuggingFace
    PreTrainedTokenizerFast.
    """

    vocab_files_names = "tokenizer.json"

    def __init__(
        self,
        tokenizer_type: str = "bpe",
        splitter: Optional[str] = "atomwise",
        trainer_args=None,
        decoder_args=None,
        token_model_args=None,
    ):
        super().__init__()
        self.tokenizer_type = tokenizer_type
        self.trainer_args = trainer_args or {}
        self.decoder_args = decoder_args or {}
        self.token_model_args = token_model_args or {}

        if tokenizer_type == "bpe":
            self.model = BPE(unk_token=UNK_TOKEN, **self.token_model_args)
            self.trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS, **self.trainer_args)  # type: ignore

        elif tokenizer_type == "wordpiece":
            self.model = WordPiece(unk_token=UNK_TOKEN, **self.token_model_args)
            self.trainer = WordPieceTrainer(special_tokens=SPECIAL_TOKENS, **self.trainer_args)  # type: ignore

        elif tokenizer_type == "unigram":
            self.model = Unigram(**self.token_model_args)
            self.trainer = UnigramTrainer(
                special_tokens=SPECIAL_TOKENS, unk_token=UNK_TOKEN, **self.trainer_args
            )

        elif tokenizer_type == "wordlevel":
            self.model = WordLevel(unk_token=UNK_TOKEN, **self.token_model_args)
            self.trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS, **self.trainer_args)  # type: ignore

        else:
            raise ValueError(f"Invalid tokenizer type {tokenizer_type}")

        self.tokenizer = Tokenizer(self.model)
        self.tokenizer = self.set_special_tokens(self.tokenizer)
        if splitter == "atomwise":
            self.tokenizer.pre_tokenizer = Split(
                Regex(
                    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
                ),
                behavior="isolated",
            )  # type: ignore
        self.tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", self.bos_token_id),
                ("<eos>", self.eos_token_id),
            ],
        )  # type: ignore
        if tokenizer_type == "wordpiece":
            self.tokenizer.decoder = decoders.WordPiece(**self.decoder_args)  # type: ignore
        else:
            self.tokenizer.decoder = decoders.BPEDecoder(**self.decoder_args)  # type: ignore

    @classmethod
    def set_special_tokens(
        cls,
        tokenizer: Tokenizer,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
    ) -> Tokenizer:
        """Set special tokens in the tokenizer.

        :param tokenizer: Tokenizer to set special tokens
        :param bos_token: BOS token, defaults to BOS_TOKEN
        :param eos_token: EOS token, defaults to EOS_TOKEN
        :return: Tokenizer with special tokens set
        """
        tokenizer.bos_token = bos_token  # type: ignore
        tokenizer.eos_token = eos_token  # type: ignore
        tokenizer.pad_token = PADDING_TOKEN  # type: ignore
        tokenizer.unk_token = UNK_TOKEN  # type: ignore

        if isinstance(tokenizer, Tokenizer):
            tokenizer.add_special_tokens(
                [
                    PADDING_TOKEN,
                    UNK_TOKEN,
                    eos_token,
                    bos_token,
                ]
            )
        return tokenizer

    @property
    def bos_token_id(self) -> int:
        """Get the bos token ID.

        :return: BOS token ID
        """
        return self.tokenizer.token_to_id(self.tokenizer.bos_token)  # type: ignore

    @property
    def pad_token_id(self) -> int:
        """Get the pad token ID.

        :return: PAD token ID
        """
        return self.tokenizer.token_to_id(self.tokenizer.pad_token)  # type: ignore

    @property
    def eos_token_id(self) -> int:
        """Get the eos token ID.

        :return: eos token ID
        """
        return self.tokenizer.token_to_id(self.tokenizer.eos_token)  # type: ignore

    def __len__(self) -> int:
        """Get the length of the tokenizer.

        :return: Length of the tokenizer
        """
        return len(self.tokenizer.get_vocab().keys())

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary of the tokenizer.

        :return: Vocabulary of the tokenizer
        """
        return self.tokenizer.get_vocab()

    def save(self, file_name=None):
        """Save the tokenizer to a file.

        :param file_name: File name, defaults to None
        """
        self.tokenizer.save(file_name)

    @classmethod
    def load(cls, file_name: str):
        """Load a tokenizer from a file.

        :param file_name: File name to load the tokenizer from
        :return: Tokenizer
        """
        tokenizer = Tokenizer.from_file(file_name)
        mol_tokenizer = cls("bpe")
        mol_tokenizer.tokenizer = mol_tokenizer.set_special_tokens(tokenizer)
        mol_tokenizer.tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", mol_tokenizer.bos_token_id),
                ("<eos>", mol_tokenizer.eos_token_id),
            ],
        )  # type: ignore
        return mol_tokenizer

    def get_pretrained(self, **kwargs) -> PreTrainedTokenizerFast:
        r"""Get a pretrained tokenizer from this tokenizer.

        Returns:
            Returns pre-trained fast tokenizer for hugging face models.
        """
        tk = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        # now we need to add special_tokens
        tk.add_special_tokens(
            {
                "bos_token": self.tokenizer.bos_token,  # type: ignore
                "eos_token": self.tokenizer.eos_token,  # type: ignore
                "pad_token": self.tokenizer.pad_token,  # type: ignore
                "unk_token": self.tokenizer.unk_token,  # type: ignore
            }
        )
        if (
            tk.model_max_length is None
            or tk.model_max_length > 1e8
            and hasattr(self.tokenizer, "model_max_length")
        ):
            tk.model_max_length = self.tokenizer.model_max_length  # type: ignore
            setattr(
                tk,
                "model_max_length",
                getattr(self.tokenizer, "model_max_length"),
            )
        return tk


def train_tokenizer(
    dataset: str = "ZINC_250K-raw",
    mol_type: str = "SMILES",
    tokenizer_type: str = "bpe",
    splitter: Optional[str] = "atomwise",
    vocab_size: int = 30000,
    min_frequency: int = 2000,
    dropout: float = 0,
    batch_size: int = 50000,
) -> None:
    trainer_args = {
        "min_frequency": min_frequency,
        "vocab_size": vocab_size,
        "show_progress": True,
    }

    token_model_args = {}
    if tokenizer_type == "bpe":
        token_model_args["dropout"] = dropout

    hf_dataset = load_dataset(f"MolGen/{dataset}")

    def batch_iterator(batch_size=batch_size):
        batch = []
        for example in hf_dataset["train"]:  # type: ignore
            batch.append(example[mol_type])  # type: ignore
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  # yield last batch
            yield batch

    tokenizer = MoleculeTokenizer(tokenizer_type=tokenizer_type, splitter=splitter, 
                                  trainer_args=trainer_args, token_model_args=token_model_args)

    tokenizer.train_from_iterator(batch_iterator())

    tokenizer.save(
        f"./data/tokenizers/tokenizer_{tokenizer_type}_{splitter}_{mol_type}_{vocab_size}_{min_frequency}_{dropout}.json"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a tokenizer")
    parser.add_argument("--dataset", type=str, default="ZINC_270M-raw")
    parser.add_argument("--mol_type", type=str, default="SMILES")
    parser.add_argument("--tokenizer_type", type=str, default="bpe")
    parser.add_argument("--splitter", type=str, default="atomwise")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=50000)
    args = parser.parse_args()
    train_tokenizer(
        args.dataset,
        args.mol_type,
        args.tokenizer_type,
        args.splitter,
        args.vocab_size,
        args.min_frequency,
        args.dropout,
        args.batch_size,
    )
