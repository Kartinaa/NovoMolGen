### Entrypoints
- file: `src/main.py`
  - main classes/functions: `EntryPoint.train`, `EntryPoint.finetune`, `EntryPoint.tokenize_dataset`, `run_train`, `run_finetune`, `run_tokenize`
  - calls:
    - train: loads Hydra config → builds `MolDataModule` → loads tokenized datasets → creates model via `init_model` → wraps in `HFTrainer` → `HFTrainer.train`
    - finetune: loads Hydra config → `_prepare_base_model` (tokenizer+model) → builds `MoleculeEvaluator`-backed `reward_fn` → selects trainer from `_FINETUNE_REGISTRY` → `TrainerCls.train`
- file: `src/eval/molecule_evaluation.py` (standalone runner for metrics)
  - main classes/functions: `MoleculeEvaluator` (also has `if __name__ == "__main__":` for quick tests)

### Config flow
- file: `src/utils.py`
  - main classes/functions: `_load_cfg` (Hydra `initialize`/`compose`), `creat_unique_experiment_name`, `creat_unique_experiment_name_for_finetune`, `_FINETUNE_REGISTRY`, `_prepare_base_model`, `_build_callbacks`
  - calls:
    - `EntryPoint.*` → `_load_cfg` (loads from `configs/…`)  
    - `run_train` uses `cfg.trainer` → `HFTrainingArguments`, `cfg.dataset` → `MolDataModule`, `cfg.model` → `init_model`, `cfg.eval`/`wandb_logs` → `_build_callbacks`
    - `run_finetune` uses `cfg.finetune` to select trainer via `_FINETUNE_REGISTRY` and to construct trainer config dataclass

### Dataset/Collator
- file: `src/data_loader/molecule_data_module.py`
  - main classes/functions: `MolDataModule.__init__`, `create_tokenized_datasets`, `load_tokenized_dataset`, fields: `tokenizer` (`PreTrainedTokenizerFast`), `data_collator` (`DataCollatorForLanguageModeling(mlm=False)`), `train_dataset`, `eval_dataset`, `max_seq_length`
  - calls:
    - constructed in `run_train` with `cfg.dataset`  
    - `run_tokenize` constructs and invokes `create_tokenized_datasets` for Arrow cache
- file: `src/data_loader/molecule_tokenizer.py`
  - main classes/functions: `MoleculeTokenizer` (+ CLI under `__main__`), supports bpe/wordpiece/unigram/wordlevel, `.get_pretrained()`
  - calls: used by `MolDataModule` and `_prepare_base_model`
- file: `src/data_loader/__init__.py`
  - re-exports: `MolDataModule`, `MoleculeTokenizer`, helpers

### Model class
- file: `src/models/modeling_novomolgen.py`
  - main classes/functions: `NovoMolGenConfig` (extends `LlamaConfig`), `NovoMolGen` (extends `GPTLMHeadModel`), `forward(...)` with CausalLM head, `auto_map` for HF Auto classes
  - calls: constructed via `utils.init_model` in train; loaded via `NovoMolGen.from_pretrained` in finetune
- file: `src/models/model_with_value_head.py`
  - main classes/functions: `NovoMolGenRewardModel` (value head; used in RL-style finetune)

### Trainer
- file: `src/trainer/hf_trainer.py`
  - main classes/functions: `HFTrainer` (fork/wrapper over `transformers.Trainer`), training loop `_maybe_log_save_evaluate`
  - calls: used in `run_train` with datasets/collator/tokenizer/model
- file: `src/trainer/policy_trainer.py`
  - main classes/functions: `PolicyTrainer` base, optimizer/scheduler prep (`_prepare_optimizer_and_scheduler`), checkpoint load/save; abstract `train`
  - calls: base for finetune trainers
- file: `src/trainer/sft_trainer.py`
  - main classes/functions: `SFTConfig` (dataclass), `SFTTrainer.train`, per-step `train_step` using `NovoMolGen`
  - calls: selected via `_FINETUNE_REGISTRY` when `cfg.finetune.type == "SFT"`
- file: `src/trainer/reinvent_trainer.py`
  - main classes/functions: `REINVENTConfig`, `REINVENTTrainer.train`, `train_step` with policy/reference model and reward shaping
  - calls: selected via `_FINETUNE_REGISTRY` for `"REINVENT"`
- file: `src/trainer/augment_hc_trainer.py`
  - main classes/functions: `AugmentedHCTrainer` (extends `REINVENTTrainer`), `AugmentedHCConfig` (dataclass; inferred)
  - calls: selected via `_FINETUNE_REGISTRY` for `"AugmentedHC"`

### Evaluation
- file: `src/eval/molecule_evaluation.py`
  - main classes/functions: `MoleculeEvaluator.__call__` computes metrics: validity/uniqueness/intdiv, MOSES props (logP, QED, SA, TPSA, rings), fragment metrics (FCD/SNN/Frag/Scaf), docking via `Docking(DockingConfig)`, TDC oracles; optional Wasserstein vs provided stats
  - calls: used in `run_finetune` to build `reward_fn`; also usable standalone
- file: `src/callbacks/evaluator.py` and `src/callbacks/wandb.py`
  - main classes/functions: `Evaluator` callback for periodic eval; `WandbCallback` for logging/artifacts
  - calls: assembled in `_build_callbacks` and passed to `HFTrainer`

### How components call each other

- finetune flow
  - `EntryPoint.finetune` → `_load_cfg` (Hydra)  
  - `_prepare_base_model` → `MoleculeTokenizer.load(...).get_pretrained()` and `NovoMolGen` or `from_pretrained`  
  - build `MoleculeEvaluator` → define `reward_fn(smiles)`  
  - select `(CfgCls, TrainerCls)` from `_FINETUNE_REGISTRY` by `cfg.finetune.type`  
  - instantiate `TrainerCls(config=CfgCls(...), model, reward_fn, tokenizer)`  
  - `TrainerCls.train(...)` → internal dataloading/generation/optimization (SFT/REINVENT/AugmentedHC)

- inference (generation) flow
  - typical: load trained `NovoMolGen` via `NovoMolGen.from_pretrained(path)` and tokenizer via `MoleculeTokenizer.load(...).get_pretrained()`  
  - call `model.generate(...)` with tokenized prompts; optionally postprocess via `MoleculeEvaluator` for metrics

### Simple call graphs (text)

- training (supervised, HF)
  - main.py:EntryPoint.train
    → utils._load_cfg
    → MolDataModule(cfg.dataset) → load_tokenized_dataset
    → utils.init_model(cfg.model) → NovoMolGen
    → utils._build_callbacks(cfg) → [Evaluator, WandbCallback]
    → HFTrainer(model, args, tokenizer, data_collator, train_dataset, eval_dataset, callbacks)
    → HFTrainer.train

- finetune (policy/RL-style)
  - main.py:EntryPoint.finetune
    → utils._load_cfg
    → utils._prepare_base_model(cfg) → [MoleculeTokenizer, NovoMolGen/from_pretrained]
    → MoleculeEvaluator(task_names=[cfg.finetune.task_name])
    → reward_fn(smiles) using MoleculeEvaluator
    → utils._FINETUNE_REGISTRY[cfg.finetune.type] → (CfgCls, TrainerCls)
    → TrainerCls(config, model, reward_fn, tokenizer)
    → TrainerCls.train

- evaluation (standalone or callback)
  - MoleculeEvaluator(gen_smiles[, stats])
    → metric dispatch (MOSES/fragment/docking/TDC)
    → returns score dict; used in reward_fn or periodic eval

- tokenizer/dataset
  - EntryPoint.tokenize_dataset
    → utils._load_cfg
    → MolDataModule(**cfg or cfg.dataset)
    → MolDataModule.create_tokenized_datasets
    → Arrow cache + `DataCollatorForLanguageModeling`

- STATUS: Completed concise architecture map and call graphs.