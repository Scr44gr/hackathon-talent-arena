import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .common import extract_features
from .models import FeatureRow, PreprocessConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> PreprocessConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = PreprocessConfig.model_validate(data)
    return cfg.resolve_paths(Path(path).resolve().parent)


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list with dataset records.")
    return data


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def fingerprint(row: FeatureRow) -> str:
    base = "||".join(
        [
            normalize_text(row.category_name).lower(),
            normalize_text(row.challenge).lower(),
            normalize_text(row.question).lower(),
            normalize_text(row.answer).lower(),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def clean_row(row: FeatureRow) -> FeatureRow:
    return row.model_copy(
        update={
            "category_name": normalize_text(row.category_name),
            "challenge": normalize_text(row.challenge),
            "question": normalize_text(row.question),
            "answer": normalize_text(row.answer),
            "proposed_answer": normalize_text(row.proposed_answer),
            "val_goal_reasoning": normalize_text(row.val_goal_reasoning),
            "conversation": normalize_text(row.conversation),
        }
    )


def filter_rows(
    rows: list[FeatureRow], cfg: PreprocessConfig
) -> tuple[list[FeatureRow], dict[str, int]]:
    stats = Counter()
    out: list[FeatureRow] = []
    for row in rows:
        if len(row.question) < cfg.min_question_chars:
            stats["dropped_short_question"] += 1
            continue
        if len(row.answer) < cfg.min_answer_chars:
            stats["dropped_short_answer"] += 1
            continue
        if cfg.drop_unlabeled and row.verdict not in {"0", "1"}:
            stats["dropped_unlabeled"] += 1
            continue
        out.append(row)
    return out, dict(stats)


def dedupe_rows(
    rows: list[FeatureRow], cfg: PreprocessConfig
) -> tuple[list[FeatureRow], dict[str, int]]:
    stats = Counter()
    deduped = rows

    if cfg.dedupe_by_message_id:
        seen_ids: set[str] = set()
        tmp: list[FeatureRow] = []
        for row in deduped:
            mid = str(row.message_id) if row.message_id is not None else ""
            if mid and mid in seen_ids:
                stats["dropped_duplicate_message_id"] += 1
                continue
            if mid:
                seen_ids.add(mid)
            tmp.append(row)
        deduped = tmp

    if cfg.dedupe_by_content:
        seen_fp: set[str] = set()
        tmp = []
        for row in deduped:
            fp = fingerprint(row)
            if fp in seen_fp:
                stats["dropped_duplicate_content"] += 1
                continue
            seen_fp.add(fp)
            tmp.append(row)
        deduped = tmp

    return deduped, dict(stats)


def to_dicts(rows: list[FeatureRow]) -> list[dict[str, Any]]:
    return [row.model_dump(by_alias=True) for row in rows]


def stratify_labels(rows: list[FeatureRow], by_category: bool) -> list[str] | None:
    labels = []
    for row in rows:
        if row.verdict not in {"0", "1"}:
            return None
        if by_category:
            labels.append(f"{row.verdict}__{row.category_name}")
        else:
            labels.append(row.verdict)

    counts = Counter(labels)
    if not counts:
        return None
    if min(counts.values()) < 2:
        return None
    return labels


def split_rows(
    rows: list[FeatureRow], cfg: PreprocessConfig
) -> tuple[list[FeatureRow], list[FeatureRow], list[FeatureRow]]:
    if cfg.val_size == 0 and cfg.test_size == 0:
        return rows, [], []

    stratify = stratify_labels(rows, cfg.stratify_by_category)

    remaining_size = cfg.val_size + cfg.test_size
    try:
        train_rows, rem_rows = train_test_split(
            rows,
            test_size=remaining_size,
            random_state=cfg.seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_rows, rem_rows = train_test_split(
            rows,
            test_size=remaining_size,
            random_state=cfg.seed,
            shuffle=True,
            stratify=None,
        )

    if cfg.test_size == 0:
        return train_rows, rem_rows, []

    val_ratio_in_rem = cfg.val_size / remaining_size
    rem_stratify = stratify_labels(rem_rows, cfg.stratify_by_category)
    try:
        val_rows, test_rows = train_test_split(
            rem_rows,
            test_size=(1 - val_ratio_in_rem),
            random_state=cfg.seed,
            shuffle=True,
            stratify=rem_stratify,
        )
    except ValueError:
        val_rows, test_rows = train_test_split(
            rem_rows,
            test_size=(1 - val_ratio_in_rem),
            random_state=cfg.seed,
            shuffle=True,
            stratify=None,
        )
    return train_rows, val_rows, test_rows


def class_distribution(rows: list[FeatureRow]) -> dict[str, int]:
    return dict(Counter([row.verdict or "unlabeled" for row in rows]))


def category_distribution(rows: list[FeatureRow]) -> dict[str, int]:
    return dict(Counter([row.category_name for row in rows]))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess judge dataset")
    parser.add_argument(
        "--config", required=True, help="Path to preprocess YAML config"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    raw = load_records(cfg.input_file)
    extracted: list[FeatureRow] = []
    dropped_invalid = 0
    for rec in raw:
        feat = extract_features(rec)
        if feat is None:
            dropped_invalid += 1
            continue
        extracted.append(clean_row(feat))

    filtered, filter_stats = filter_rows(extracted, cfg)
    deduped, dedupe_stats = dedupe_rows(filtered, cfg)

    if not deduped:
        raise RuntimeError("No rows left after preprocessing.")

    train_rows, val_rows, test_rows = split_rows(deduped, cfg)

    out_dir = Path(cfg.output_dir)
    write_json(out_dir / "dataset_clean.json", to_dicts(deduped))
    write_json(out_dir / "train.json", to_dicts(train_rows))
    if val_rows:
        write_json(out_dir / "val.json", to_dicts(val_rows))
    if test_rows:
        write_json(out_dir / "test.json", to_dicts(test_rows))

    report = {
        "input_file": cfg.input_file,
        "counts": {
            "raw": len(raw),
            "extracted": len(extracted),
            "dropped_invalid": dropped_invalid,
            "after_filter": len(filtered),
            "after_dedupe": len(deduped),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "filter_stats": filter_stats,
        "dedupe_stats": dedupe_stats,
        "class_distribution": {
            "all": class_distribution(deduped),
            "train": class_distribution(train_rows),
            "val": class_distribution(val_rows),
            "test": class_distribution(test_rows),
        },
        "category_distribution": {
            "all": category_distribution(deduped),
            "train": category_distribution(train_rows),
            "val": category_distribution(val_rows),
            "test": category_distribution(test_rows),
        },
    }
    write_json(out_dir / "preprocess_report.json", report)

    print(f"Saved processed data to: {out_dir}")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
