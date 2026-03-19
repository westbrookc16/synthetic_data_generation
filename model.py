from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DIYRepairQA(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., ge=1, description="Unique integer identifier for the record")
    category: str = Field(..., min_length=3, description="Dataset category used to generate the record.")
    prompt: str = Field(default="", description="Full prompt text used to generate the record.")
    model: str = Field(description="Which model was used.")
    
    question: str = Field(..., min_length=10, description="A realistic DIY repair question from a homeowner")
    answer: str = Field(..., min_length=20, description="A clear, actionable answer with step-by-step guidance")
    equipment_problem: str = Field(..., min_length=3, description='The specific problem being addressed (e.g. "dripping faucet")')
    tools_required: list[str] = Field(..., min_length=1, description="Tools a typical homeowner would realistically own")
    steps: list[str] = Field(..., min_length=3, description="Ordered, numbered repair steps")
    safety_info: str = Field(..., min_length=10, description="Relevant safety warnings and precautions")
    tips: list[str] = Field(..., min_length=1, description="Practical tips to make the repair easier or more reliable")

    @model_validator(mode="before")
    @classmethod
    def populate_category_from_legacy_prompt(cls, data: object) -> object:
        if isinstance(data, dict) and "category" not in data and "prompt" in data:
            data = dict(data)
            data["category"] = data["prompt"]
        return data

    @field_validator("category", "question", "answer", "equipment_problem", "safety_info")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("prompt")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("tools_required", "steps", "tips")
    @classmethod
    def validate_non_empty_list_items(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("list items must be non-empty strings")
        return cleaned

    @field_validator("steps")
    @classmethod
    def validate_steps_are_numbered(cls, values: list[str]) -> list[str]:
        for i, step in enumerate(values, start=1):
            if not (step.startswith(f"{i}. ") or step.startswith(f"{i}) ")):
                raise ValueError(f"step {i} must start with '{i}. ' or '{i}) '")
        return values
