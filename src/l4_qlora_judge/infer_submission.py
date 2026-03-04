import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .common import extract_features, split_reason_and_result
from .models import FeatureRow, InferConfig, MetricsRow, SubmissionRow
from .promptnoises import CustomConfig, process_prompts
from .prompts import ABSOLUTE_PROMPT, ABS_SYSTEM_PROMPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> InferConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = InferConfig.model_validate(data)

    cfg_file = Path(path).resolve()
    cfg_dir = cfg_file.parent
    return cfg.resolve_paths(cfg_dir)


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list with dataset records.")
    return data


def build_instruction(sample: FeatureRow, question_text: str) -> str:
    return (
        ABS_SYSTEM_PROMPT
        + "\n\n"
        + ABSOLUTE_PROMPT.format(
            category_name=sample.category_name,
            challenge=sample.challenge,
            question=question_text,
            answer=sample.answer,
            proposed_answer=sample.proposed_answer,
        )
    )


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    outputs: list[str] = []
    device = model.device

    for prompt_batch in chunked(prompts, batch_size):
        messages = [[{"role": "user", "content": p}] for p in prompt_batch]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(device)

        do_sample = temperature > 0
        with torch.no_grad():
            gen_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        in_len = model_inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen_ids[:, in_len:], skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def quick_metrics(items: list[MetricsRow]) -> dict[str, float]:
    labeled = [x for x in items if x.verdict in {"0", "1"}]
    if not labeled:
        return {}

    y_true = np.array([int(x.verdict or "0") for x in labeled])
    po = np.array([int(x.po_m_pred) for x in labeled])
    pt = np.array([int(x.pt_m_pred) for x in labeled])
    pg = np.array([int(x.pg_m_pred) for x in labeled])

    acc = float((y_true == po).mean())
    var = float((~((po == pt) & (po == pg))).mean())
    robust = 1.0 - var
    return {"accuracy_po": acc, "robustness": robust, "variability": var}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for hackathon submission")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    hf_token = os.getenv("HF_TOKEN")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.adapter_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        token=hf_token,
        trust_remote_code=cfg.trust_remote_code,
        device_map="auto",
        quantization_config=bnb,
    )
    model = PeftModel.from_pretrained(base_model, cfg.adapter_path)
    model.eval()

    records = load_records(cfg.input_file)
    features: list[FeatureRow] = []
    for rec in records:
        feat = extract_features(rec)
        if feat is not None:
            features.append(feat)

    custom_cfg = CustomConfig(
        n_typos=1,
        n_grammar_changes=4,
        remove_open_questions=True,
        strip_accents=True,
        remove_commas=True,
        lowercase=True,
    )

    po_prompts: list[str] = []
    pt_prompts: list[str] = []
    pg_prompts: list[str] = []

    for feat in features:
        variants = process_prompts([feat.question], custom_cfg=custom_cfg)[0]
        po_prompts.append(build_instruction(feat, variants["prompt_original"]))
        pt_prompts.append(build_instruction(feat, variants["prompt_typos"]))
        pg_prompts.append(
            build_instruction(feat, variants["prompt_grammatical_errors"])
        )

    po_out = batched_generate(
        model,
        tokenizer,
        po_prompts,
        cfg.batch_size,
        cfg.max_new_tokens,
        cfg.temperature,
    )
    pt_out = batched_generate(
        model,
        tokenizer,
        pt_prompts,
        cfg.batch_size,
        cfg.max_new_tokens,
        cfg.temperature,
    )
    pg_out = batched_generate(
        model,
        tokenizer,
        pg_prompts,
        cfg.batch_size,
        cfg.max_new_tokens,
        cfg.temperature,
    )

    submission: list[SubmissionRow] = []
    internal_for_metrics: list[MetricsRow] = []

    for feat, out_po, out_pt, out_pg in zip(features, po_out, pt_out, pg_out):
        po_reason, po_pred = split_reason_and_result(
            out_po, default_pred=cfg.default_pred_if_missing
        )
        pt_reason, pt_pred = split_reason_and_result(
            out_pt, default_pred=cfg.default_pred_if_missing
        )
        pg_reason, pg_pred = split_reason_and_result(
            out_pg, default_pred=cfg.default_pred_if_missing
        )

        row = SubmissionRow(
            message_id=feat.message_id,
            po_m_pred=po_pred,
            po_m_reason=po_reason,
            pt_m_pred=pt_pred,
            pt_m_reason=pt_reason,
            pg_m_pred=pg_pred,
            pg_m_reason=pg_reason,
        )
        submission.append(row)

        metrics_row = MetricsRow(
            verdict=feat.verdict,
            po_m_pred=po_pred,
            pt_m_pred=pt_pred,
            pg_m_pred=pg_pred,
        )
        internal_for_metrics.append(metrics_row)

    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [row.model_dump(by_alias=True) for row in submission],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved submission to: {output_path}")

    metrics = quick_metrics(internal_for_metrics)
    if metrics:
        print(
            "Quick metrics -> "
            f"accuracy_po={metrics['accuracy_po']:.4f}, "
            f"robustness={metrics['robustness']:.4f}, "
            f"variability={metrics['variability']:.4f}"
        )


if __name__ == "__main__":
    main()
