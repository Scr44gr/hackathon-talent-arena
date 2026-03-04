from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JudgeBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Message(JudgeBaseModel):
    role: str = ""
    content: str = ""


class Category(JudgeBaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None


class ValidationGoal(JudgeBaseModel):
    val_goal_reasoning: Optional[str] = None


class Validation(JudgeBaseModel):
    goal: Optional[ValidationGoal] = None


class RawRecord(JudgeBaseModel):
    messages: list[Message] = Field(default_factory=list)
    challenge: Optional[str] = None
    category: Optional[Category] = None


class InputRecord(JudgeBaseModel):
    message_id: Optional[int | str] = Field(default=None, alias="message-id")
    raw: Optional[RawRecord] = None
    category: Optional[Category] = None
    challenge: Optional[str] = None
    proposed_answer: Optional[str] = None
    verdict: Optional[str | int] = None
    validation: Optional[Validation] = None


class Turn(JudgeBaseModel):
    question: str
    answer: str
    history: list[Message]
    conversation: str


class FeatureRow(JudgeBaseModel):
    message_id: Optional[int | str] = Field(default=None, alias="message-id")
    challenge: str = ""
    category_name: str = "Unknown"
    question: str
    answer: str
    history: list[Message] = Field(default_factory=list)
    conversation: str = ""
    proposed_answer: str
    verdict: Optional[Literal["0", "1"]] = None
    val_goal_reasoning: str = "The response is evaluated against the safety rubric."


class SubmissionRow(JudgeBaseModel):
    message_id: Optional[int | str] = Field(default=None, alias="message-id")
    po_m_pred: Literal["0", "1"]
    po_m_reason: str
    pt_m_pred: Literal["0", "1"]
    pt_m_reason: str
    pg_m_pred: Literal["0", "1"]
    pg_m_reason: str


class MetricsRow(JudgeBaseModel):
    verdict: Optional[Literal["0", "1"]] = None
    po_m_pred: Literal["0", "1"]
    pt_m_pred: Literal["0", "1"]
    pg_m_pred: Literal["0", "1"]


class TrainConfig(JudgeBaseModel):
    seed: int = 42
    model_name: str = "prometheus-eval/prometheus-7b-v2.0"
    trust_remote_code: bool = False
    train_file: str = "../data/dataset_sample.json"
    output_dir: str = "../output/prometheus_l4_qlora"
    max_seq_length: int = 1536
    val_size: float = 0.15
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    @field_validator("val_size")
    @classmethod
    def validate_val_size(cls, value: float) -> float:
        if value <= 0 or value >= 0.5:
            raise ValueError("val_size must be > 0 and < 0.5")
        return value

    def resolve_paths(self, cfg_dir: Path) -> "TrainConfig":
        return self.model_copy(
            update={
                "train_file": str((cfg_dir / self.train_file).resolve()),
                "output_dir": str((cfg_dir / self.output_dir).resolve()),
            }
        )


class PreprocessConfig(JudgeBaseModel):
    seed: int = 42
    input_file: str = "../data/dataset_sample.json"
    output_dir: str = "../data/processed"
    val_size: float = 0.15
    test_size: float = 0.0
    drop_unlabeled: bool = True
    dedupe_by_message_id: bool = True
    dedupe_by_content: bool = True
    min_question_chars: int = 8
    min_answer_chars: int = 8
    stratify_by_category: bool = True

    @field_validator("val_size", "test_size")
    @classmethod
    def validate_split_size(cls, value: float) -> float:
        if value < 0 or value >= 1:
            raise ValueError("split sizes must be >= 0 and < 1")
        return value

    @field_validator("min_question_chars", "min_answer_chars")
    @classmethod
    def validate_min_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("minimum char limits must be >= 0")
        return value

    def resolve_paths(self, cfg_dir: Path) -> "PreprocessConfig":
        if (self.val_size + self.test_size) >= 1:
            raise ValueError("val_size + test_size must be < 1")
        return self.model_copy(
            update={
                "input_file": str((cfg_dir / self.input_file).resolve()),
                "output_dir": str((cfg_dir / self.output_dir).resolve()),
            }
        )


class InferConfig(JudgeBaseModel):
    seed: int = 42
    adapter_path: str = "../output/prometheus_l4_qlora"
    base_model_name: str = "prometheus-eval/prometheus-7b-v2.0"
    trust_remote_code: bool = False
    input_file: str = "../data/validation_dataset_sample.json"
    output_file: str = "../output/submission_l4_qlora.json"
    batch_size: int = 4
    max_new_tokens: int = 220
    temperature: float = 0.0
    default_pred_if_missing: Literal["0", "1"] = "0"

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0:
            raise ValueError("temperature must be >= 0")
        return value

    def resolve_paths(self, cfg_dir: Path) -> "InferConfig":
        return self.model_copy(
            update={
                "adapter_path": str((cfg_dir / self.adapter_path).resolve()),
                "input_file": str((cfg_dir / self.input_file).resolve()),
                "output_file": str((cfg_dir / self.output_file).resolve()),
            }
        )
