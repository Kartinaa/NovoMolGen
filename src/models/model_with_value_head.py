import rootutils
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig


class NovoMolGenRewardModel(nn.Module):
    def __init__(
            self, 
            pretrained_model: Optional[NovoMolGen], 
            pad_token_id: int = None, 
            problem_type: str = "regression",
            num_labels: int = 1,
            **kwargs):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.pretrained_model.base_config.pad_token_id = pad_token_id
        self.problem_type = problem_type
        self.num_labels = num_labels

        self.score = nn.Sequential(
            nn.Linear(self.pretrained_model.base_config.hidden_size, self.pretrained_model.base_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.pretrained_model.base_config.hidden_size, self.num_labels)
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Labels for computing the left padding mask. Indices should be in ``[0, ..., sequence_length - 1]``.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        hidden_states = base_model_output.hidden_states
        logits = self.score(hidden_states)

        batch_size = input_ids.shape[0]

        if self.pretrained_model.base_config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.pretrained_model.base_config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.pretrained_model.base_config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                raise NotImplementedError

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            hidden_states=base_model_output.hidden_states,
        )
        

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        pretrained_model_state_dict = {f"pretrained_model.{k}":v for k,v in pretrained_model_state_dict.items()}

        score_state_dict = self.score.state_dict(*args, **kwargs)
        for k, v in score_state_dict.items():
            pretrained_model_state_dict[f"score.{k}"] = v
        return pretrained_model_state_dict
