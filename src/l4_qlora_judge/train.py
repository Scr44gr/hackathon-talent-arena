import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .common import extract_features
from .models import FeatureRow, TrainConfig
from .prompts import ABSOLUTE_PROMPT, ABS_SYSTEM_PROMPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = TrainConfig.model_validate(data)
    cfg_file = Path(path).resolve()
    cfg_dir = cfg_file.parent
    return cfg.resolve_paths(cfg_dir)


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list with dataset records.")
    return data


def build_instruction(sample: FeatureRow, system_prompt: str, user_prompt: str) -> str:
    return (
        system_prompt
        + "\n\n"
        + user_prompt.format(
            category_name=sample.category_name,
            challenge=sample.challenge,
            question=sample.question,
            answer=sample.answer,
            proposed_answer=sample.proposed_answer,
        )
    )


def to_training_row(
    sample: FeatureRow, system_prompt: str, user_prompt: str, eos_token: str
) -> dict[str, Any]:
    label = sample.verdict
    if label not in {"0", "1"}:
        return {"prompt_sft": ""}

    prompt = build_instruction(sample, system_prompt, user_prompt)
    reasoning = (sample.val_goal_reasoning or "").strip()
    if not reasoning:
        reasoning = "The response is evaluated against the safety rubric."

    text = f"{prompt}{reasoning} [RESULT] {label}{eos_token}"
    return {"prompt_sft": text}


def build_sft_config(cfg: TrainConfig, bf16: bool, fp16: bool) -> SFTConfig:
    candidate_kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "dataset_text_field": "prompt_sft",
        "max_seq_length": cfg.max_seq_length,
        "max_length": cfg.max_seq_length,
        "per_device_train_batch_size": cfg.train_batch_size,
        "per_device_eval_batch_size": cfg.eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "save_total_limit": cfg.save_total_limit,
        "evaluation_strategy": "steps",
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": cfg.seed,
        "packing": False,
    }

    valid_keys = set(inspect.signature(SFTConfig).parameters)
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_keys}
    return SFTConfig(**kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train QLoRA judge model on L4")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=hf_token,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        token=hf_token,
        trust_remote_code=cfg.trust_remote_code,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.config.use_cache = False

    system_prompt = cfg.system_prompt or ABS_SYSTEM_PROMPT
    user_prompt = cfg.user_prompt or ABSOLUTE_PROMPT

    raw_records = load_records(cfg.train_file)
    rows: list[FeatureRow] = []
    for rec in raw_records:
        feat = extract_features(rec)
        if feat is not None:
            rows.append(feat)

    if not rows:
        raise RuntimeError("No valid rows extracted from training file.")

    labeled_rows = [row for row in rows if row.verdict in {"0", "1"}]
    dataset = Dataset.from_list([row.model_dump(by_alias=True) for row in labeled_rows])
    if "verdict" not in dataset.column_names:
        raise RuntimeError("Training file must contain verdict labels.")

    split = dataset.train_test_split(test_size=cfg.val_size, seed=cfg.seed)

    split = split.map(
        lambda sample: to_training_row(
            FeatureRow.model_validate(sample),
            system_prompt,
            user_prompt,
            tokenizer.eos_token,
        ),
        desc="Build SFT rows",
    )
    split = split.filter(lambda x: bool((x.get("prompt_sft") or "").strip()))

    os.makedirs(cfg.output_dir, exist_ok=True)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )

    train_args = build_sft_config(cfg, bf16, fp16)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "peft_config": peft_config,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Saved adapter and tokenizer to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
