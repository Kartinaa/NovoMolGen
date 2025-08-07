import copy
import json
import os.path
import re
import shutil
import inspect
from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.modeling_utils import unwrap_model, logger
from functools import partial
from safetensors.torch import load_file as safe_load_file

try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    GPTLMHeadModel = None

try:
    from flash_attn.models.llama import llama_config_to_gpt2_config, inv_remap_state_dict_hf_llama
except ImportError:
    llama_config_to_gpt2_config = None
    inv_remap_state_dict_hf_llama = None


def state_dict_from_pretrained(model_name, checkpoint_path: str = "", device=None, dtype=None, **kwargs):
    """
    code modified from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/pretrained.py
    """

    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    is_sharded = False
    load_safe = False

    # Try loading from HF hub instead of from local files
    resolved_archive_file = cached_file(model_name, os.path.join(checkpoint_path, WEIGHTS_NAME),
                                        _raise_exceptions_for_missing_entries=False, **kwargs)
    if resolved_archive_file is None:
        resolved_archive_file = cached_file(model_name, os.path.join(checkpoint_path, WEIGHTS_INDEX_NAME),
                                            _raise_exceptions_for_missing_entries=False, **kwargs)
        if resolved_archive_file is not None:
            is_sharded = True

    if resolved_archive_file is None:
        raise EnvironmentError(f"Model name {model_name} was not found.")

    if load_safe:
        loader = partial(safe_load_file, device=mapped_device)
    else:
        loader = partial(torch.load, map_location=mapped_device)

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different
        # checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            model_name, resolved_archive_file
        )
        state_dict = {}
        for sharded_file in resolved_archive_file:
            state_dict.update(loader(sharded_file))
    else:
        state_dict = loader(resolved_archive_file)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}

    return state_dict


class NovoMolGenConfig(LlamaConfig):

    def __init__(self,
                 use_flash_attn: bool = True,
                 fused_bias_fc: bool = True,
                 fused_mlp: bool = False,
                 fused_dropout_add_ln: bool = True,
                 residual_in_fp32: bool = True,
                 loss_type: str = 'ForCausalLM',
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.use_flash_attn = use_flash_attn
        self.fused_bias_fc = fused_bias_fc
        self.fused_mlp = fused_mlp
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        self.loss_type = loss_type
        self.auto_map = {"AutoModelForCausalLM": "modeling_novomolgen.NovoMolGen"}

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            checkpoint_path: str = "",
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            **kwargs,
    ):

        resolved_archive_config_file = cached_file(pretrained_model_name_or_path,
                                                   os.path.join(checkpoint_path, "config.json"),
                                                   _raise_exceptions_for_missing_entries=False, force_download=force_download)

        if resolved_archive_config_file is not None:
            with open(resolved_archive_config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_dict = json.loads(text)

        else:
            raise EnvironmentError(f"config for {pretrained_model_name_or_path} was not found.")

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class NovoMolGen(GPTLMHeadModel):
    def __init__(
            self,
            config: NovoMolGenConfig,
            mol_type: str = "SMILES",
    ):
        self.base_config = config
        self.mol_type = mol_type
        config = llama_config_to_gpt2_config(config)
        config.use_flash_attn = self.base_config.use_flash_attn
        config.fused_bias_fc = self.base_config.fused_bias_fc
        config.fused_mlp = self.base_config.fused_mlp
        config.fused_dropout_add_ln = self.base_config.fused_dropout_add_ln
        config.residual_in_fp32 = self.base_config.residual_in_fp32
        GPTLMHeadModel.__init__(self, config)

    # TODO: here we ignore attention_mask to make it compatible with HF trainer. The MHA in flash-attention should
    #  be reimplement and integrate attention_mask like here:
    #  https://github.com/huggingface/transformers/blob/0864dd3beb238b7bec3528a3d1d6c17a28f51a51/src/transformers/models/llama/modeling_llama.py#L536
    def forward(self, input_ids, attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None,
                position_ids=None, inference_params=None, num_last_tokens=0, **loss_kwargs):
        """
                input_ids: (batch, seqlen) int tensor
                inference_params: for generation. Adapted from Megatron-LM (and Apex)
                https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
                num_last_tokens: if > 0, only return the logits for the last n tokens
                """
        assert (
                input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape
        hidden_states = self.transformer(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        if self.output_scale != 1.0:
            hidden_states = hidden_states * self.output_scale
        if not self.norm_head:
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_head_weight = F.normalize(self.lm_head.weight)
            lm_logits = F.linear(hidden_states, lm_head_weight, bias=self.lm_head.bias)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=lm_logits, labels=labels, vocab_size=self.base_config.vocab_size,
                                      **loss_kwargs)

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states
        )

    @property
    def loss_function(self):
        if getattr(self.base_config, "loss_type", None) is not None:
            loss_type = self.base_config.loss_type
        else:
            loss_type = self.__class__.__name__
            if loss_type not in LOSS_MAPPING:
                loss_groups = f"({'|'.join(LOSS_MAPPING)})"
                loss_type = re.findall(loss_groups, self.__class__.__name__)
                if len(loss_type) > 0:
                    loss_type = loss_type[0]
                else:
                    loss_type = None
        if loss_type is None or loss_type not in LOSS_MAPPING and getattr(self.base_config, "loss_type",
                                                                          None) is not None:
            print(
                f"`loss_type={loss_type}` was set in the base_config but it is unrecognised."
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            safe_serialization: bool = False,
            **kwargs,
    ):

        if safe_serialization:
            raise ImportError("`safe_serialization` is not implemented yet`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)
        # Save the config
        if is_main_process:
            self.base_config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self)
            state_dict = model_to_save.state_dict()

        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        torch.save(state_dict, os.path.join(save_directory, weights_name))

        # find the file where NovoMolGen is defined
        src = inspect.getsourcefile(type(self))
        if src:
            dst = os.path.join(save_directory, os.path.basename(src))
            shutil.copy(src, dst)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        checkpoint_path: str = "",
        config: Optional[Union[NovoMolGenConfig, str, os.PathLike]] = None,
        **kwargs,
        ):
        if config is None:
            config = NovoMolGenConfig.from_pretrained(pretrained_model_name_or_path, checkpoint_path=checkpoint_path, **kwargs)
        model = cls(config)

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(os.path.join(pretrained_model_name_or_path, checkpoint_path, WEIGHTS_NAME))
        else:
            state_dict = state_dict_from_pretrained(pretrained_model_name_or_path, checkpoint_path=checkpoint_path, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def sample(
            self,
            tokenizer,
            batch_size: int = 4,
            max_length: int = 64,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.95,
            device: torch.device = torch.device("cuda"),
    ):
        """
        Generate a batch of sequences from the model.

        Returns a dictionary with up to three keys:
        {
            "<mol_type>": <list of raw sequences in that moltype>,
            "sequences": <torch.LongTensor of valid token IDs>
        }
        """
        input_ids = tokenizer.encode("", return_tensors="pt").to(device)
        # Repeat the prompt for the desired batch size
        input_ids = input_ids.repeat_interleave(batch_size, dim=0)
        # If the tokenizer includes an EOS token for an empty prompt, we remove it.
        if input_ids.shape[1] > 1:
            input_ids = input_ids[:, :-1]

        generation_output = self.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        sequences = self._filter_tokens_after_eos(
            generation_output.sequences, eos_id=tokenizer.eos_token_id
        )

        decoded_strings = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        decoded_strings = [s.replace(" ", "") for s in decoded_strings]

        result = {
            self.mol_type: decoded_strings,
            "sequences": sequences,
        }
        return result

    @staticmethod
    def _filter_tokens_after_eos(sequences, eos_id):
        output = copy.deepcopy(sequences)
        for i in range(sequences.size(0)):
            row = sequences[i]
            eos_position = (row == eos_id).nonzero()
            if eos_position.numel() > 0:
                eos_position = eos_position[0, 0].item()  # Get the index of the first occurrence
                output[i, eos_position + 1:] = eos_id
        return output

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # HF’s GenerationMixin would normally do more, but for a basic LM this usually suffices:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
