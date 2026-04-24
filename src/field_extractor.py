"""Layout-agnostic invoice / LC / B/L / receipt field extractor.

The extracted Markdown from any engine in src.extract goes through one LLM
call here; the LLM must fill the fields it sees and leave the rest null.
The schema deliberately covers commercial invoices, LCs, and B/Ls in a
single flat-ish model — the lab is for inter-engine comparison, not for
strict UCP 600 / SWIFT MT700 validation.

LLM providers (pluggable):
  - "local"  : reuse src.infer.chat (MLX or HF Qwen2.5)
  - "openai" : OpenAI-compatible API (Azure / DashScope / Ollama-OpenAI)
  - "ollama" : local Ollama server (no API key)

Usage:
    from src.field_extractor import extract_fields, get_llm
    llm = get_llm("local")
    doc = extract_fields(markdown, llm=llm)
    print(doc.fields.model_dump_json(indent=2))
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ============================================================
# Schema
# ============================================================
class Party(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    address: str | None = None
    contact: str | None = None
    tax_id: str | None = None
    swift_bic: str | None = None       # for bank parties on LCs


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str | None = None
    quantity: float | None = None
    unit: str | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    hs_code: str | None = None         # tariff / commodity code


class InvoiceLikeDoc(BaseModel):
    """Layout-agnostic schema covering invoice / LC / B/L / receipt."""
    model_config = ConfigDict(extra="ignore")

    document_number: str | None = None
    document_date: date | None = None
    document_type_guess: str | None = Field(
        default=None,
        description="LLM's free-text label, e.g. 'commercial invoice'.",
    )
    issuer: Party | None = None
    recipient: Party | None = None
    currency: str | None = Field(default=None, description="ISO 4217 if detectable.")
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total_amount: Decimal | None = None
    incoterm: str | None = None
    port_of_loading: str | None = None
    port_of_discharge: str | None = None
    shipment_date: date | None = None
    due_date: date | None = None
    payment_terms: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    other_fields: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


@dataclass
class ExtractedDoc:
    engine: str                        # source extraction engine ID
    fields: InvoiceLikeDoc
    raw_markdown: str
    latency_ms: float
    llm_model: str
    raw_response: str = ""             # for debugging / agreement matrix


# ============================================================
# LLM Protocol + adapters
# ============================================================
class ChatLLM(Protocol):
    name: str

    def chat_json(self, system: str, user: str) -> str:
        """Return a raw JSON string. The model must respect json_object mode."""
        ...


class LocalChatLLM:
    """Wrap src.infer.chat (MLX or HF). No native JSON mode — we coerce in
    the prompt and parse defensively."""

    def __init__(self, model_id: str | None = None):
        from . import config as C
        self._model_id = model_id
        # Best human-readable label for the UI
        self.name = model_id or os.environ.get("LOCAL_LLM_MODEL") or (
            C.STAGE2_LLM_MLX if os.environ.get("USE_MLX", "1") == "1"
            else C.STAGE2_LLM_HF
        )

    def chat_json(self, system: str, user: str) -> str:
        from .infer import chat
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user
             + "\n\nRespond ONLY with a single JSON object, no prose, no code fences."},
        ]
        return chat(
            messages,
            model_id=self._model_id,
            max_tokens=2560,
            temperature=0.0,
        )


class OpenAIChatLLM:
    """OpenAI-compatible endpoint (OpenAI / Azure / DashScope / Ollama-OpenAI).

    Configured via env:
      OPENAI_API_KEY         (required)
      OPENAI_BASE_URL        (optional — for Azure / DashScope / Ollama)
      FIELD_LLM_MODEL        (optional — default 'gpt-4o-mini')
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai 未安装。运行: uv sync --extra extract"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置。")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = (
            OpenAI(api_key=api_key, base_url=base_url) if base_url
            else OpenAI(api_key=api_key)
        )
        self.name = model or os.environ.get("FIELD_LLM_MODEL", "gpt-4o-mini")

    def chat_json(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or "{}"


class OllamaChatLLM:
    """Plain Ollama HTTP API with `format: json` for guaranteed JSON.

    Configured via env:
      OLLAMA_BASE_URL    (default http://localhost:11434)
      OLLAMA_MODEL       (default qwen2.5:1.5b-instruct)
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        import httpx
        self._httpx = httpx
        self._base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.name = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")

    def chat_json(self, system: str, user: str) -> str:
        r = self._httpx.post(
            f"{self._base}/api/chat",
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=180.0,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


def get_llm(provider: str) -> ChatLLM:
    """Factory. provider: 'local' | 'openai' | 'ollama'."""
    if provider == "local":
        return LocalChatLLM()
    if provider == "openai":
        return OpenAIChatLLM()
    if provider == "ollama":
        return OllamaChatLLM()
    raise ValueError(f"unknown LLM provider: {provider!r}")


# ============================================================
# Extraction
# ============================================================
_EXAMPLE_JSON = """{
  "document_number": "INV-2024-0001",
  "document_date": "2024-03-15",
  "document_type_guess": "commercial invoice",
  "issuer":    {"name": "ACME Co., Ltd", "address": "123 Main St", "contact": null, "tax_id": null, "swift_bic": null},
  "recipient": {"name": "Buyer Inc",    "address": "456 Park Ave", "contact": null, "tax_id": null, "swift_bic": null},
  "currency": "USD",
  "subtotal": 1000.00,
  "tax": 80.00,
  "total_amount": 1080.00,
  "incoterm": "FOB",
  "port_of_loading": "Shanghai",
  "port_of_discharge": "Los Angeles",
  "shipment_date": "2024-04-01",
  "due_date": "2024-05-15",
  "payment_terms": "Net 60",
  "line_items": [
    {"description": "Widget A", "quantity": 100, "unit": "pcs", "unit_price": 10.00, "amount": 1000.00, "hs_code": "8473.30"}
  ],
  "other_fields": {"PO Number": "PO-9988", "LC Reference": "LC123456"},
  "notes": "Payment on sight."
}"""

SYSTEM_PROMPT = """You are a document-understanding assistant. You receive the
Markdown text of a single business document — commercial invoice, Letter of
Credit (SWIFT MT700-like), Bill of Lading, receipt, or similar. The layout
varies; the document may be in English or Chinese.

Your job: extract what IS PRESENT in the document into the SAME JSON shape as
the example below. Rules:

- Use null when a field is not in the document. Do NOT guess or invent values.
- Currency: ISO 4217 codes (USD, EUR, CNY, HKD, ...).
- Dates: ISO 8601 (YYYY-MM-DD).
- Numbers: digits only, no thousand separators, no currency symbols.
- `issuer` and `recipient` are objects with the keys shown; use null if absent.
- `line_items` is a JSON array of objects with the keys shown; empty array if none.
- `other_fields` is an object whose keys AND values are both strings — use it for
  named fields that don't fit (e.g. "PO Number", "LC Field :47A:", bank refs).
- Output ONE JSON object with EXACTLY the keys in the example, no extras, no
  prose, no markdown fences.

Example output (values are illustrative — yours come from the document):
""" + _EXAMPLE_JSON + "\n"


def _build_prompt(markdown: str) -> tuple[str, str]:
    return SYSTEM_PROMPT, f"Document Markdown:\n\n```markdown\n{markdown}\n```"


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _coerce_json(raw: str) -> dict:
    """Strip code fences, extract first {...} block, parse."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Find the first balanced JSON object
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object found in LLM response: {raw[:200]!r}")
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError(f"unbalanced JSON in LLM response: {raw[:200]!r}")
    return json.loads(text[start:end])


def _sanitize(data: dict) -> dict:
    """Best-effort fixups for tiny/loose LLMs:
       - drop $ref / $defs noise the model sometimes echoes back
       - coerce other_fields values to strings; drop non-mapping entirely
       - coerce line_items to list, dropping malformed items
    """
    if not isinstance(data, dict):
        return {}
    cleaned = {k: v for k, v in data.items() if not k.startswith("$")}
    # other_fields → dict[str, str]
    of = cleaned.get("other_fields")
    if isinstance(of, dict):
        cleaned["other_fields"] = {
            str(k): (
                ", ".join(map(str, v)) if isinstance(v, (list, tuple))
                else "" if v is None
                else str(v)
            )
            for k, v in of.items()
        }
    elif of is not None:
        cleaned.pop("other_fields", None)
    # line_items → list of dicts
    li = cleaned.get("line_items")
    if li is not None and not isinstance(li, list):
        cleaned.pop("line_items", None)
    elif isinstance(li, list):
        cleaned["line_items"] = [x for x in li if isinstance(x, dict)]
    # issuer/recipient → must be dict if present (drop $ref echoes)
    for party_key in ("issuer", "recipient"):
        v = cleaned.get(party_key)
        if isinstance(v, dict) and any(k.startswith("$") for k in v):
            cleaned[party_key] = None
    return cleaned


def extract_fields(markdown: str, llm: ChatLLM, engine: str = "") -> ExtractedDoc:
    """Single LLM call → InvoiceLikeDoc. engine is just a label for the
    returned ExtractedDoc."""
    sys_msg, user_msg = _build_prompt(markdown)
    t0 = time.time()
    raw = llm.chat_json(sys_msg, user_msg)
    latency = (time.time() - t0) * 1000
    try:
        data = _sanitize(_coerce_json(raw))
        fields = InvoiceLikeDoc.model_validate(data)
    except Exception as e:
        logger.warning(f"field parse failed for engine={engine}: {e}; raw={raw[:300]!r}")
        fields = InvoiceLikeDoc()
    return ExtractedDoc(
        engine=engine,
        fields=fields,
        raw_markdown=markdown,
        latency_ms=latency,
        llm_model=llm.name,
        raw_response=raw,
    )
