import dataclasses
import json
import os
import rootutils
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from transformers import SchedulerType
import numpy as np


import torch
from transformers.utils import is_torch_bf16_gpu_available
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling

from collections.abc import Mapping

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import NovoMolGen
from src.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class PolicyTrainerConfig:
    """Configuration class for OffPolicyTrainer."""
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    type: str = "policy_trainer"
    """type of trainer"""
    seed: int = 0
    """Seed value for random generations"""
    task_name: Optional[str] = "Albuterol_Similarity"
    """Name of task to use - used only for tracking purposes"""
    num_epochs: int = 1
    """Number of training epoch"""
    learning_rate: float = 1e-4
    """Adam learning rate"""
    per_device_train_batch_size: int = 1024
    """Number of samples per optimisation step"""
    per_device_eval_batch_size: int = 2048
    """Number of samples per evalution step"""
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation step"""
    weight_decay: float = 0.0
    """Weight decay"""
    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping"""
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="constant_with_warmup",
        metadata={"help": "The scheduler type to use."},
    )
    """Learning rate scheduler type"""
    lr_scheduler_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    fp16: bool = False
    """Use fp16"""
    bf16: bool = True
    """Use bf16"""
    save_total_limit: int = 2
    """Save total limit"""
    save_steps: int = 100
    """Save checkpoint steps"""
    eval_step: int = 1
    """Logging steps"""
    higher_is_better: bool = True
    """Higher is better"""
    sequence_logp_reduction: str = ''
    """Normlize sequence log probability with sequence length"""

    dataset_name: str = "MolGen/ZINC_250K_prop"
    use_random_sampling: bool = True

    # Generation config
    max_length: int = 64
    """Maximum length of generated sequence"""
    temperature: float = 1.0
    """Sampling temperature"""
    top_k: int = 0.
    """Sampling top k"""
    top_p: float = 0.
    """Sampling top p"""
    num_evaluation_samples: int = 3000
    """Number of evaluation samples"""
    compute_metric_eval: bool = True
    """Compute metrics during evalution"""
    stop_criteria_rules: Dict = None
    """Stop criteria rules"""
    normalize_docking: float = None
    """Normolizing factor of reward for dokcing"""

    # will not use directly
    checkpoint: Union[int, str] = None
    save_path: str = None
    additional_evaluation_metrics: List[str] = None
    wandb_tag: str = None

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

        if self.bf16:
            if torch.cuda.is_available() and not is_torch_bf16_gpu_available():
                # gpu
                raise ValueError(
                    "Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU with cuda>=11.0"
                )

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.fp16:
            self.mixed_precision_dtype = "fp16"
        elif self.bf16:
            self.mixed_precision_dtype = "bf16"
        os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

        if self.task_name.startswith("docking"):
            logger.info("change higher_is_better to False")
            self.higher_is_better = False
            if self.normalize_docking is None:
                self.normalize_docking = -20.0
                logger.info("set normalize_docking to -20.0")

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

def flatten_dict(nested: Dict, sep: str = "/") -> Dict:
    """Flatten dictionary and concatenate nested keys with separator."""

    def recurse(nest: Dict, prefix: str, into: Dict) -> None:
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                recurse(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    recurse(nested, "", flat)
    return flat

def create_reference_model(model: NovoMolGen):
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    """
    ref_model = deepcopy(model)

    # Freeze all parameters in the model
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model.eval()


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@dataclass
class TrainerState:

    global_step: int = 0
    log_history: List[Dict[str, float]] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StoppingCriteria:
    def __init__(self, rules):
        """
        Initialize with a set of rules.
        Each rule is a dictionary with:
        - 'task_name': Name of the metric to check (e.g., 'validity')
        - 'threshold': Threshold value for the metric
        - 'higher': Boolean indicating if the metric should be higher or lower than the threshold
        """
        self.rules = rules

    def check(self, metrics):
        """
        Evaluate if all criteria are met based on the given metrics.
        Args:
            metrics (dict): Dictionary of metric values, e.g., {'validity': 0.95, 'unique@1k': 0.92}

        Returns:
            bool: True if all criteria are satisfied, False otherwise.
        """
        if self.rules is None:
            return False
        for rule_name, rule in self.rules.items():
            task_name = rule['task_name']
            threshold = rule['threshold']
            higher = rule['higher']

            # Ensure the metric exists
            if task_name not in metrics:
                print(f"Warning: Metric '{task_name}' not found in computed metrics.")
                return True

            # Check the condition
            if higher:
                if metrics[task_name] < threshold:
                    print(f"Criteria '{rule_name}' not met: {task_name} < {threshold}")
                    return True
            else:
                if metrics[task_name] > threshold:
                    print(f"Criteria '{rule_name}' not met: {task_name} > {threshold}")
                    return True

        # All criteria are satisfied
        return False
