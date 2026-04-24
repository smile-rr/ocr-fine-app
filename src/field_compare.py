"""Inter-engine agreement matrix for ExtractedDoc results.

Given {engine_id: ExtractedDoc} from N engines, build a DataFrame with one
row per (flattened) field and one column per engine. Cells hold the
normalized value for comparison; the trailing 'agreement' column counts
distinct non-null answers.

UI uses the matrix two ways:
  - Style cells by agreement (green / amber / red)
  - Show raw value (un-normalized) in tooltips / expanders
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from .field_extractor import ExtractedDoc, InvoiceLikeDoc, LineItem, Party


# ============================================================
# Flatten Pydantic InvoiceLikeDoc -> {dotted.path: value}
# ============================================================
_PARTY_FIELDS = list(Party.model_fields)
_LINEITEM_FIELDS = list(LineItem.model_fields)


def flatten(doc: InvoiceLikeDoc) -> dict[str, Any]:
    """Flatten the nested model to a stable, comparable {field_path: value}."""
    out: dict[str, Any] = {}

    for fname in InvoiceLikeDoc.model_fields:
        val = getattr(doc, fname)
        if fname in ("issuer", "recipient"):
            party: Party | None = val
            for sub in _PARTY_FIELDS:
                out[f"{fname}.{sub}"] = getattr(party, sub) if party else None
        elif fname == "line_items":
            items: list[LineItem] = val or []
            out["line_items.count"] = len(items)
            for i, item in enumerate(items[:10]):           # cap row count for matrix
                for sub in _LINEITEM_FIELDS:
                    out[f"line_items[{i}].{sub}"] = getattr(item, sub)
        elif fname == "other_fields":
            d: dict = val or {}
            out["other_fields.count"] = len(d)
            for k, v in list(d.items())[:20]:
                out[f"other_fields.{k}"] = v
        else:
            out[fname] = val

    return out


# ============================================================
# Normalization (so 'USD 1,234.50' == '1234.5' for comparison)
# ============================================================
_CURRENCY_SYM_RE = re.compile(r"[¥$€£₹]|RMB|CNY|USD|EUR|GBP|HKD|JPY", re.IGNORECASE)


def normalize(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        try:
            d = Decimal(str(value)).quantize(Decimal("0.01"))
            # drop trailing zero on integers (1234.00 -> 1234)
            s = format(d.normalize(), "f")
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s
        except Exception:
            return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # currency-looking strings: strip symbols + commas, keep digits/dot/sign
        if _CURRENCY_SYM_RE.search(s) or any(ch.isdigit() for ch in s) and "," in s:
            stripped = _CURRENCY_SYM_RE.sub("", s).replace(",", "").strip()
            try:
                d = Decimal(stripped)
                return normalize(d)
            except Exception:
                pass
        return " ".join(s.lower().split())
    return str(value)


# ============================================================
# Agreement matrix
# ============================================================
def agreement_matrix(results: dict[str, ExtractedDoc]) -> pd.DataFrame:
    """rows = field path, cols = engine IDs (+ 'distinct').

    Cells contain the normalized stringified value; missing/null cells are NaN.
    The 'distinct' column = count of distinct non-null normalized values across
    engines. 1 = perfect agreement (or only one engine answered),
    >1 = disagreement.
    """
    if not results:
        return pd.DataFrame()

    flat_per_engine = {eng: flatten(doc.fields) for eng, doc in results.items()}
    # Stable union of keys
    all_fields: list[str] = []
    seen = set()
    for f in flat_per_engine.values():
        for k in f:
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    rows: list[dict[str, Any]] = []
    for field in all_fields:
        row: dict[str, Any] = {"field": field}
        normalized_vals: list[str] = []
        for eng in results:
            val = flat_per_engine[eng].get(field)
            n = normalize(val)
            row[eng] = n
            if n is not None:
                normalized_vals.append(n)
        row["distinct"] = len(set(normalized_vals))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("field")
    return df


def style_agreement(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Color cells by row-level agreement.

      green  : all engines that answered agree (distinct == 1)
      amber  : some engines didn't answer but the answers given agree
               (distinct == 1 AND at least one NaN cell in row)
      red    : engines disagree (distinct > 1)
      gray   : every engine returned null (distinct == 0)
    """
    engine_cols = [c for c in df.columns if c != "distinct"]

    def color_row(row: pd.Series):
        distinct = int(row["distinct"])
        n_engines = len(engine_cols)
        n_filled = sum(1 for c in engine_cols if pd.notna(row[c]))
        if distinct == 0:
            color = "background-color: rgba(128,128,128,0.10)"
        elif distinct > 1:
            color = "background-color: rgba(220,53,69,0.20)"        # red
        elif n_filled < n_engines:
            color = "background-color: rgba(255,193,7,0.20)"        # amber
        else:
            color = "background-color: rgba(40,167,69,0.18)"        # green
        return [color] * len(row)

    return df.style.apply(color_row, axis=1)
