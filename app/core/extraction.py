"""The extraction contract: the fields the model may propose each turn.

This is the boundary between the model and the code. The model fills these
fields; validation here decides what survives. The schema has the same shape
for every domain; only the allowed values change.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model, field_validator

from .domain import Domain
from .records import FOLLOWUP_TOPICS

Emotion = Literal["neutral", "frustrated", "angry", "anxious", "confused", "refusing", "positive"]


class Extraction(BaseModel):
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    id_last4: Optional[str] = None
    account_number: Optional[str] = None

    caller_role: Optional[Literal["holder", "representative"]] = None
    rep_name: Optional[str] = None
    relationship: Optional[str] = None

    record_id: Optional[str] = None
    party: Optional[str] = None
    date: Optional[str] = None
    month: Optional[int] = Field(default=None, ge=1, le=12)
    year: Optional[int] = None
    followup_topic: Optional[str] = None
    question: Optional[str] = None

    scope: Literal["in_scope", "out_of_scope", "smalltalk"] = "in_scope"
    emotion: Emotion = "neutral"
    wants_human: bool = False
    accepts_human: bool = False
    declines_human: bool = False
    refuses_verification: bool = False
    restricted_request: bool = False
    emergency: bool = False
    email_choice: Optional[Literal["send", "skip"]] = None
    done: bool = False

    @field_validator("followup_topic")
    @classmethod
    def _topic(cls, v):
        return v if v in FOLLOWUP_TOPICS else None

    @field_validator("date")
    @classmethod
    def _iso(cls, v):
        try:
            return date.fromisoformat(v).isoformat() if v else None
        except ValueError:
            return None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "unknown", "n/a"}:
            return None
        return v


@lru_cache(maxsize=8)
def model_for(domain_key: str):
    from .domain import get_domain
    d = get_domain(domain_key)
    return create_model(
        f"TurnExtraction_{domain_key}", __base__=Extraction,
        category=(Optional[Literal[tuple(d.categories)]], None),
        record_status=(Optional[Literal[tuple(d.status_groups)]], None),
        intent=(Optional[Literal[tuple([*d.intents, "other"])]], None),
    )


def json_schema(domain: Domain) -> dict:
    s = model_for(domain.key).model_json_schema()
    s.pop("title", None)
    return s


def parse(raw: dict, domain: Domain):
    M = model_for(domain.key)
    clean = {k: v for k, v in (raw or {}).items() if k in M.model_fields}
    try:
        return M(**clean)
    except Exception:
        # Drop fields one at a time until it validates; a bad field never kills the turn.
        ok = {}
        for k, v in clean.items():
            try:
                M(**{**ok, k: v})
                ok[k] = v
            except Exception:
                continue
        return M(**ok)
