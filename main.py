import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    parser = argparse.ArgumentParser(description="L4 QLoRA Judge CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Run QLoRA training")
    train_parser.add_argument(
        "--config", required=True, help="Path to train YAML config"
    )

    infer_parser = sub.add_parser("infer", help="Run submission inference")
    infer_parser.add_argument(
        "--config", required=True, help="Path to inference YAML config"
    )

    eval_parser = sub.add_parser("eval", help="Run LLM-judge evaluation")
    eval_parser.add_argument(
        "--config", required=True, help="Path to evaluation YAML config"
    )

    prep_parser = sub.add_parser("preprocess", help="Run dataset preprocessing")
    prep_parser.add_argument(
        "--config", required=True, help="Path to preprocessing YAML config"
    )

    args = parser.parse_args()

    if args.command == "train":
        from l4_qlora_judge.train import main as train_main

        train_main(["--config", args.config])
        return

    if args.command == "infer":
        from l4_qlora_judge.infer_submission import main as infer_main

        infer_main(["--config", args.config])
        return

    if args.command == "eval":
        from l4_qlora_judge.eval_pydantic_evals import main as eval_main

        eval_main(["--config", args.config])
        return

    if args.command == "preprocess":
        from l4_qlora_judge.preprocess import main as preprocess_main

        preprocess_main(["--config", args.config])
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
