from dataclasses import dataclass, field
from typing import Any, Dict, Optional


VALID_UNIT_STATUSES = {
    "pending",
    "success",
    "retryable_error",
    "permanent_error",
}


@dataclass
class LLMWorkUnit:
    unit_id: str
    row_id: str
    field_name: str
    input_text: str
    output_column: str
    model: str
    step_name: str
    prompt_version: str
    step_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "row_id": self.row_id,
            "field_name": self.field_name,
            "input_text": self.input_text,
            "output_column": self.output_column,
            "model": self.model,
            "step_name": self.step_name,
            "prompt_version": self.prompt_version,
        }


@dataclass
class LLMResultRecord:
    unit_id: str
    row_id: str
    output_column: str
    status: str
    parsed_output: Optional[str] = None
    output_value: Any = None
    raw_output: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    review_flag: bool = False
    review_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "row_id": self.row_id,
            "output_column": self.output_column,
            "status": self.status,
            "parsed_output": self.parsed_output,
            "output_value": self.output_value,
            "raw_output": self.raw_output,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "review_flag": self.review_flag,
            "review_reason": self.review_reason,
        }


@dataclass
class LLMProgressRecord:
    unit_id: str
    status: str
    retry_count: int
    last_error_type: Optional[str] = None
    last_error_message: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status,
            "retry_count": self.retry_count,
            "last_error_type": self.last_error_type,
            "last_error_message": self.last_error_message,
            "updated_at": self.updated_at,
        }


@dataclass
class LLMRunOutcome:
    status: str  # complete | retryable_incomplete | permanent_failure
    processed_units: int
    remaining_units: int