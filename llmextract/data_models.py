# llmextract/data_models.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CharInterval(BaseModel):
    """
    Represents the character interval of an extraction in the source text.

    Attributes:
        start: The starting character index (inclusive).
        end: The ending character index (exclusive).
    """

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int

    @field_validator("start", "end", mode="before")
    def _coerce_int_and_non_negative(cls, v: Any) -> int:
        if not isinstance(v, int):
            try:
                v = int(v)
            except Exception as exc:
                raise TypeError("start and end must be integers") from exc
        if v < 0:
            raise ValueError("start and end must be non-negative")
        return v

    @model_validator(mode="after")
    def _check_order(self):
        if self.start >= self.end:
            raise ValueError("CharInterval.start must be less than CharInterval.end")
        return self


class Extraction(BaseModel):
    """
    Represents a single piece of structured information extracted from text.

    Attributes:
        extraction_class: The category or type of the extraction (e.g., "medication").
        extraction_text: The exact text snippet extracted from the source document.
        attributes: A dictionary of structured attributes related to the extraction.
        char_interval: The character position of the extraction in the source text.
    """

    model_config = ConfigDict(extra="ignore")

    extraction_class: str
    extraction_text: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    char_interval: Optional[CharInterval] = None
    id: Optional[str] = None

    @field_validator("extraction_class", "extraction_text", mode="before")
    def _coerce_to_str_and_strip(cls, v: Any) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            v = str(v)
        return v.strip()

    @model_validator(mode="after")
    def _validate_required(self):
        if not self.extraction_class:
            raise ValueError("extraction_class must be a non-empty string")
        if not self.extraction_text:
            raise ValueError("extraction_text must be a non-empty string")
        if self.attributes is None:
            self.attributes = {}
        elif not isinstance(self.attributes, dict):
            raise TypeError("attributes must be a dict")
        return self


class ExampleData(BaseModel):
    """
    A few-shot example containing an input text and its correct extractions.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    extractions: List[Extraction]

    @field_validator("text", mode="before")
    def _coerce_text(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v)
        return v


class AnnotatedDocument(BaseModel):
    """
    The final output: original document with its extractions and optional metadata.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    extractions: List[Extraction] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", mode="before")
    def _coerce_text(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v)
        return v

    @model_validator(mode="after")
    def _normalize_metadata(self):
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        return self
