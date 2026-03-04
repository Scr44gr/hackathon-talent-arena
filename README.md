# L4 QLoRA Judge Pipeline

## Quick start with uv

1) Create and activate environment:

```bash
uv venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```bash
uv sync
```

If you need a specific CUDA build for PyTorch, install torch first with your preferred index/wheel, then run `uv sync`.

Optional: copy `.env-example` to `.env` and set tokens.

3) Train:

```bash
uv run judge-train --config configs/train_l4.yaml
```

4) Run inference and generate submission:

```bash
uv run judge-infer --config configs/infer_l4.yaml

# Optional: run LLM-as-a-Judge evaluation with pydantic-evals
uv run judge-eval --config configs/eval_llm_judge.yaml
```

## Notes

- The scripts perturb only the `question` field for robustness (`po`, `pt`, `pg`).
- Output format follows the hackathon schema:
  - `message-id`, `po_m_pred`, `po_m_reason`, `pt_m_pred`, `pt_m_reason`, `pg_m_pred`, `pg_m_reason`
- If your input file includes `verdict`, inference also prints quick offline metrics (accuracy + robustness score).

## Colab workflow (Fine-Tuning + LoRA)

Use a GPU runtime (L4 recommended).

1) Clone and install:

```bash
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>/l4_qlora_judge
!pip install -U pip
!pip install -e .
```

2) Set API/auth tokens (as needed):

```python
import os
os.environ["HF_TOKEN"] = "<your_hf_token>"
# for pydantic-evals judge model on OpenAI
os.environ["OPENAI_API_KEY"] = "<your_openai_key>"
```

3) Train QLoRA:

```bash
!judge-train --config configs/train_l4.yaml
```

4) Generate submission-style predictions:

```bash
!judge-infer --config configs/infer_l4.yaml
```

5) Evaluate with pydantic-evals LLMJudge:

```bash
!judge-eval --config configs/eval_llm_judge.yaml
```

Artifacts:
- adapter/tokenizer: `output/prometheus_l4_qlora`
- submission: `output/submission_l4_qlora.json`
- judge eval report: `output/eval_llm_judge_report.json`
