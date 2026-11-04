"""
Autonomous CFO: Deep 3-Way Matching Engine
Meta Global AI Developer Hackathon - single-file Streamlit app

Run:
    pip install -U streamlit plotly pandas numpy openai pillow
    streamlit run app.py

Works fully offline (deterministic matching engine + local fallbacks).
When an API key is entered in the sidebar, the app makes real calls through the
OpenAI-compatible SDK (Meta Model API / Baseten) for:
  - vision field extraction        (Muse Spark)
  - audit workpapers               (Muse Spark)
  - analytics code generation      (Muse Code)
  - voice transcription            (Muse Voice Transcribe)
Every live call falls back to the local engine on any error.
"""
from __future__ import annotations

import base64
import builtins
import hashlib
import html
import io
import json
import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except Exception:  # openai not installed -> offline mode only
    OpenAI = None

try:
    from PIL import Image
except Exception:  # pillow missing -> synthetic page preview only
    Image = None

st.set_page_config(
    page_title="Autonomous CFO | 3-Way Matching Engine",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants & master data (mock ERP)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_OPTIONS = ["meta-models/muse-spark-open", "muse-spark"]
DEFAULT_BASE_URL = "https://inference.baseten.co/v1"
DEFAULT_VOICE_MODEL = "muse-voice-transcribe"
DEFAULT_QUERY = "Show me vendor price variance trends over time"

EXC_TYPES = ["QUANTITY_SHORTAGE", "PRICE_VARIANCE", "UNAPPROVED_FREIGHT", "BANK_DETAIL_MISMATCH"]
SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
SEV_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
SEV_COLOR = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
PRICE_TOLERANCE_PCT = 2.0

ACCOUNTS = {
    "1400": "Inventory",
    "5000": "Cost of Goods Sold (COGS)",
    "5100": "Freight-In",
    "2000": "Accounts Payable",
}
CATEGORY_ACCOUNT = {"Inventory": "1400", "COGS": "5000"}

STATUS_LABEL = {
    "AUTO_POSTED": "Auto-posted (STP)",
    "PENDING_REVIEW": "Pending review",
    "APPROVED_POSTED": "Approved & posted",
    "ADJUSTED_POSTED": "Adjusted & posted",
    "DISPUTED": "Disputed",
}

VENDORS = {
    "V-100": {"name": "Apex Industrial Supply", "bank": "ACC-4455-7712-0091"},
    "V-200": {"name": "Northwind Components Ltd", "bank": "ACC-9021-3348-7710"},
    "V-300": {"name": "Orion Packaging Co", "bank": "ACC-1180-6672-3305"},
    "V-400": {"name": "Helix Chemicals Inc", "bank": "ACC-7734-0059-2216"},
    "V-500": {"name": "Vertex Logistics Parts", "bank": "ACC-5562-8841-1093"},
}


def _pl(sku, desc, cat, po_qty, po_price, grn_qty):
    return {"sku": sku, "desc": desc, "category": cat, "po_qty": po_qty, "po_price": po_price, "grn_qty": grn_qty}


PURCHASE_ORDERS = {
    # Perfect-match batch
    "PO-7001": {"vendor_id": "V-100", "grn": "GRN-9001", "freight_cap": 0.0, "lines": [
        _pl("SKU-BRG-6205", "Ball Bearing 6205", "Inventory", 500, 12.40, 500),
        _pl("SKU-FST-M8", "Hex Fastener M8", "Inventory", 2000, 0.35, 2000)]},
    "PO-7002": {"vendor_id": "V-200", "grn": "GRN-9002", "freight_cap": 0.0, "lines": [
        _pl("SKU-PCB-114", "Control PCB Rev C", "Inventory", 300, 48.00, 300)]},
    "PO-7003": {"vendor_id": "V-300", "grn": "GRN-9003", "freight_cap": 0.0, "lines": [
        _pl("SKU-BOX-L", "Corrugated Box Large", "COGS", 5000, 1.15, 5000),
        _pl("SKU-TAPE-48", "Packing Tape 48mm", "COGS", 800, 2.10, 800)]},
    "PO-7004": {"vendor_id": "V-400", "grn": "GRN-9004", "freight_cap": 0.0, "lines": [
        _pl("SKU-SOL-20L", "Industrial Solvent 20L", "Inventory", 120, 86.50, 120)]},
    "PO-7005": {"vendor_id": "V-500", "grn": "GRN-9005", "freight_cap": 0.0, "lines": [
        _pl("SKU-GSK-77", "Gasket Set 77", "Inventory", 650, 9.80, 650)]},
    "PO-7006": {"vendor_id": "V-100", "grn": "GRN-9006", "freight_cap": 0.0, "lines": [
        _pl("SKU-BRG-6308", "Ball Bearing 6308", "Inventory", 250, 21.75, 250)]},
    # Fraud & shortage batch
    "PO-8001": {"vendor_id": "V-200", "grn": "GRN-9101", "freight_cap": 0.0, "lines": [
        _pl("SKU-CAP-22", "Capacitor 22uF", "Inventory", 4000, 0.42, 4000)]},
    "PO-8002": {"vendor_id": "V-100", "grn": "GRN-9102", "freight_cap": 0.0, "lines": [
        _pl("SKU-BRG-6205", "Ball Bearing 6205", "Inventory", 1000, 12.40, 820)]},
    "PO-8003": {"vendor_id": "V-300", "grn": "GRN-9103", "freight_cap": 0.0, "lines": [
        _pl("SKU-BOX-L", "Corrugated Box Large", "COGS", 3000, 1.15, 3000)]},
    "PO-8004": {"vendor_id": "V-400", "grn": "GRN-9104", "freight_cap": 0.0, "lines": [
        _pl("SKU-SOL-20L", "Industrial Solvent 20L", "Inventory", 80, 86.50, 80)]},
    "PO-8005": {"vendor_id": "V-500", "grn": "GRN-9105", "freight_cap": 0.0, "lines": [
        _pl("SKU-GSK-77", "Gasket Set 77", "Inventory", 900, 9.80, 900)]},
    "PO-8006": {"vendor_id": "V-200", "grn": "GRN-9106", "freight_cap": 200.0, "lines": [
        _pl("SKU-PCB-114", "Control PCB Rev C", "Inventory", 150, 48.00, 150)]},
    "PO-8007": {"vendor_id": "V-100", "grn": "GRN-9107", "freight_cap": 0.0, "lines": [
        _pl("SKU-BRG-6308", "Ball Bearing 6308", "Inventory", 400, 21.75, 360),
        _pl("SKU-FST-M8", "Hex Fastener M8", "Inventory", 5000, 0.35, 5000)]},
    "PO-8008": {"vendor_id": "V-300", "grn": "GRN-9108", "freight_cap": 0.0, "lines": [
        _pl("SKU-TAPE-48", "Packing Tape 48mm", "COGS", 1200, 2.10, 1200)]},
    # Used by the sample scanned invoice in Tab 1
    "PO-9001": {"vendor_id": "V-400", "grn": "GRN-9201", "freight_cap": 0.0, "lines": [
        _pl("SKU-SOL-20L", "Industrial Solvent 20L", "Inventory", 60, 86.50, 55)]},
}


def _inv(inv_id, po, d, lines, freight=0.0, bank=None):
    vendor = VENDORS[PURCHASE_ORDERS[po]["vendor_id"]]
    return {
        "invoice_id": inv_id, "po": po, "vendor_id": PURCHASE_ORDERS[po]["vendor_id"],
        "vendor": vendor["name"], "date": d, "bank": bank or vendor["bank"], "freight": float(freight),
        "lines": [{"sku": s, "qty": float(q), "price": float(p)} for s, q, p in lines],
        "source": "batch",
    }


def batch_perfect():
    return [
        _inv("INV-101", "PO-7001", "2026-10-05", [("SKU-BRG-6205", 500, 12.40), ("SKU-FST-M8", 2000, 0.35)]),
        _inv("INV-102", "PO-7002", "2026-10-06", [("SKU-PCB-114", 300, 48.00)]),
        _inv("INV-103", "PO-7003", "2026-10-07", [("SKU-BOX-L", 5000, 1.15), ("SKU-TAPE-48", 800, 2.10)]),
        _inv("INV-104", "PO-7004", "2026-10-08", [("SKU-SOL-20L", 120, 86.50)]),
        _inv("INV-105", "PO-7005", "2026-10-09", [("SKU-GSK-77", 650, 9.80)]),
        _inv("INV-106", "PO-7006", "2026-10-10", [("SKU-BRG-6308", 250, 21.75)]),
    ]


def batch_fraud():
    return [
        _inv("INV-401", "PO-8001", "2026-10-12", [("SKU-CAP-22", 4000, 0.42)]),
        _inv("INV-402", "PO-8002", "2026-10-12", [("SKU-BRG-6205", 1000, 12.40)]),
        _inv("INV-403", "PO-8003", "2026-10-13", [("SKU-BOX-L", 3000, 1.31)]),
        _inv("INV-404", "PO-8004", "2026-10-14", [("SKU-SOL-20L", 80, 86.50)], freight=1250.0),
        _inv("INV-405", "PO-8005", "2026-10-14", [("SKU-GSK-77", 900, 9.80)], bank="ACC-0001-9999-4410"),
        _inv("INV-406", "PO-8006", "2026-10-15", [("SKU-PCB-114", 150, 48.00)], freight=180.0),
        _inv("INV-407", "PO-8007", "2026-10-16", [("SKU-BRG-6308", 400, 22.60), ("SKU-FST-M8", 5000, 0.35)], freight=640.0),
        _inv("INV-408", "PO-8008", "2026-10-16", [("SKU-TAPE-48", 1200, 2.10)], freight=95.0),
    ]


BATCHES = {"Perfect Match Batch": batch_perfect, "Fraud & Shortage Batch": batch_fraud}

SAMPLE_EXTRACTION = {
    "document_type": "INVOICE",
    "vendor_name": "Helix Chemicals Inc",
    "invoice_number": "INV-2291",
    "po_number": "PO-9001",
    "invoice_date": "2026-10-22",
    "currency": "USD",
    "bank_account": "ACC-7734-0059-2216",
    "freight": 0.0,
    "line_items": [{"sku": "SKU-SOL-20L", "description": "Industrial Solvent 20L", "qty": 60.0, "unit_price": 86.50}],
    "ocr_confidence": 0.964,
}

BASE_REGIONS = [
    ("Header · Vendor & Logo", (0.05, 0.04, 0.95, 0.17), "#38bdf8"),
    ("Bill-To / Ship-To", (0.05, 0.19, 0.50, 0.30), "#a78bfa"),
    ("Invoice Meta · No / PO / Date", (0.55, 0.19, 0.95, 0.30), "#f472b6"),
    ("Line-Item Table", (0.05, 0.33, 0.95, 0.62), "#22c55e"),
    ("Totals & Freight", (0.60, 0.64, 0.95, 0.76), "#eab308"),
    ("Bank Details", (0.05, 0.80, 0.60, 0.92), "#ef4444"),
]
REGION_LABELS = [r[0] for r in BASE_REGIONS]


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def h(s) -> str:
    """HTML-escape and neutralise '$' so Streamlit's markdown never parses LaTeX."""
    return html.escape(str(s)).replace("$", "&#36;")


def md_safe(s) -> str:
    return str(s).replace("$", r"\$")


def money(x) -> str:
    return f"${float(x):,.2f}"


def r2(x) -> float:
    return round(float(x), 2)


def mask(acct) -> str:
    a = str(acct)
    return "••••" + a[-4:] if len(a) > 4 else a


def rgba(hexcolor: str, alpha: float) -> str:
    c = hexcolor.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def badge(sev: str) -> str:
    return f'<span class="badge badge-{sev.lower()}">{h(sev)}</span>'


def _stretch(fn, *args, **kwargs):
    """Full-width rendering that works on old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def parse_json_loose(text: str) -> dict:
    text = (text or "").strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[s:e + 1])


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
.stApp { background: #0e1117; }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: #0a0f1d; border-right: 1px solid #1c2640; }
.hero-title { font-size: 1.85rem; font-weight: 750; letter-spacing: -0.01em; color: #f1f5ff; margin: 0; }
.hero-sub { color: #8fa0c4; font-size: .95rem; margin: .2rem 0 1rem 0; }
.mode-pill { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .75rem; font-weight: 700; margin-left: 10px; vertical-align: middle; }
.mode-live { background: rgba(34,197,94,.15); color: #4ade80; border: 1px solid #166534; }
.mode-offline { background: rgba(234,179,8,.12); color: #facc15; border: 1px solid #854d0e; }
.kpi-card { background: linear-gradient(160deg, #151c30 0%, #101627 100%); border: 1px solid #26304a;
  border-left: 4px solid var(--accent, #38bdf8); border-radius: 12px; padding: 14px 18px; min-height: 104px; }
.kpi-label { color: #8fa0c4; font-size: .78rem; font-weight: 600; }
.kpi-value { color: #f8fafc; font-size: 1.85rem; font-weight: 750; line-height: 1.25; font-variant-numeric: tabular-nums; }
.kpi-sub { color: #6f7fa3; font-size: .76rem; }
.badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 750; letter-spacing: .03em; }
.badge-critical { background:#ef4444; color:#fff; }
.badge-high { background:#f97316; color:#1a0b00; }
.badge-medium { background:#eab308; color:#1f1700; }
.badge-low { background:#22c55e; color:#04210f; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 8px; font-size:.74rem; font-weight:600;
  background:#1a2340; color:#b9c5e6; border:1px solid #2a3760; }
.q-wrap { overflow-x:auto; border:1px solid #26304a; border-radius:12px; background:#111729; }
.q-table { width:100%; border-collapse:collapse; font-size:.86rem; }
.q-table th { text-align:left; color:#8fa0c4; font-weight:600; padding:10px 12px; border-bottom:1px solid #26304a; white-space:nowrap; }
.q-table td { padding:9px 12px; border-bottom:1px solid #1b2440; color:#e5e9f5; white-space:nowrap; }
.q-table tr:last-child td { border-bottom:none; }
.q-table .num { text-align:right; font-variant-numeric: tabular-nums; }
.chips { display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 12px 0; }
.chip { background:#131b31; border:1px solid #26304a; color:#c7d2ee; border-radius:10px; padding:6px 12px; font-size:.8rem; }
.chip b { color:#7dd3fc; }
.inv-strip { background:#111729; border:1px solid #26304a; border-radius:12px; padding:12px 16px; margin: 6px 0 10px 0; color:#e5e9f5; }
.inv-strip .muted { color:#8fa0c4; font-size:.8rem; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#121a2e; border-radius:10px 10px 0 0; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background:#1b2746 !important; }
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
def reset_workspace():
    ss = st.session_state
    ss.invoices = {}
    ss.exceptions = []
    ss.gl = []
    ss.decisions = []
    ss.je_counter = 0
    ss.workpapers = {}


def load_batch(name: str):
    reset_workspace()
    ok = 0
    for inv in BATCHES[name]():
        done, _ = ingest_invoice(inv)
        ok += int(done)
    st.session_state.batch_name = name
    st.session_state.flash = f"Loaded “{name}”: {ok} invoices processed by the 3-way engine."


def init_state():
    ss = st.session_state
    ss.setdefault("api_key", os.environ.get("META_API_KEY", ""))
    ss.setdefault("base_url", DEFAULT_BASE_URL)
    ss.setdefault("model", MODEL_OPTIONS[0])
    ss.setdefault("voice_model", DEFAULT_VOICE_MODEL)
    ss.setdefault("nlq", DEFAULT_QUERY)
    ss.setdefault("voice_text", "CFO Agent, approve invoice 402")
    ss.setdefault("voice_hash", "")
    ss.setdefault("voice_ver", 0)
    ss.setdefault("voice_status", "Record a command, or edit the transcript manually.")
    if "invoices" not in ss:
        reset_workspace()
        load_batch("Fraud & Shortage Batch")
    if "doc" not in ss:
        ss.doc = {
            "name": "Sample scanned invoice (demo)", "image": None,
            "extracted": json.loads(json.dumps(SAMPLE_EXTRACTION)),
            "regions": build_regions("demo-sample"), "mode": "simulated", "ver": 0,
        }
    if "analysis" not in ss:
        ss.analysis = run_analysis(DEFAULT_QUERY, force_offline=True)


# ──────────────────────────────────────────────────────────────────────────────
# AI client wrappers (real calls when a key is present, local fallback otherwise)
# ──────────────────────────────────────────────────────────────────────────────
def get_client():
    key = (st.session_state.get("api_key") or "").strip()
    if not key or OpenAI is None:
        return None
    base = (st.session_state.get("base_url") or "").strip() or None
    return OpenAI(api_key=key, base_url=base, timeout=45.0, max_retries=1)


def live_mode() -> bool:
    return get_client() is not None


def chat(client, messages, max_tokens=900, temperature=0.2) -> str:
    resp = client.chat.completions.create(
        model=st.session_state.model, messages=messages, temperature=temperature,
        max_tokens=max(max_tokens, 4000),
    )
    return resp.choices[0].message.content or ""

# ──────────────────────────────────────────────────────────────────────────────
# 3-way matching engine
# ──────────────────────────────────────────────────────────────────────────────
def compare_lines(inv: dict) -> pd.DataFrame:
    cols = ["sku", "desc", "category", "po_qty", "grn_qty", "inv_qty", "qty_delta", "po_price",
            "inv_price", "price_pct", "billed", "matched", "status"]
    po = PURCHASE_ORDERS[inv["po"]]
    po_idx = {l["sku"]: l for l in po["lines"]}
    rows = []
    for ln in inv["lines"]:
        qty, price = float(ln["qty"]), float(ln["price"])
        pl = po_idx.get(ln["sku"])
        if pl:
            po_qty, grn_qty, po_price = float(pl["po_qty"]), float(pl["grn_qty"]), float(pl["po_price"])
            desc, cat = pl["desc"], pl["category"]
        else:
            po_qty = grn_qty = po_price = 0.0
            desc, cat = "(SKU not on PO)", "Inventory"
        qty_delta = qty - grn_qty
        price_pct = (price - po_price) / po_price * 100 if po_price > 0 else 0.0
        matched_price = min(price, po_price) if po_price > 0 else 0.0
        over_qty = qty_delta > 1e-9
        over_price = price_pct > PRICE_TOLERANCE_PCT
        if pl is None:
            status = "❌ Not on PO"
        elif over_qty and over_price:
            status = "❌ Qty + Price"
        elif over_qty:
            status = "⚠️ Over-billed vs GRN"
        elif over_price:
            status = "⚠️ Price above PO"
        else:
            status = "✅ Match"
        rows.append({
            "sku": ln["sku"], "desc": desc, "category": cat, "po_qty": po_qty, "grn_qty": grn_qty,
            "inv_qty": qty, "qty_delta": qty_delta, "po_price": po_price, "inv_price": price,
            "price_pct": round(price_pct, 2), "billed": r2(qty * price),
            "matched": r2(min(qty, grn_qty) * matched_price), "status": status,
        })
    return pd.DataFrame(rows, columns=cols)


def invoice_total(inv: dict) -> float:
    return r2(float(compare_lines(inv)["billed"].sum()) + float(inv["freight"]))


def matched_total(inv: dict) -> float:
    cap = PURCHASE_ORDERS[inv["po"]]["freight_cap"]
    return r2(float(compare_lines(inv)["matched"].sum()) + min(float(inv["freight"]), cap))


def detect_exceptions(inv: dict) -> list[dict]:
    df = compare_lines(inv)
    po = PURCHASE_ORDERS[inv["po"]]
    vendor = VENDORS[po["vendor_id"]]
    out = []

    def add(kind, sev, amount, detail):
        out.append({
            "exc_id": f"{inv['invoice_id']}:{kind}", "invoice_id": inv["invoice_id"], "vendor": inv["vendor"],
            "type": kind, "severity": sev, "amount": r2(amount), "detail": detail,
        })

    short = df[df["qty_delta"] > 1e-9]
    if not short.empty:
        amt = float((short["qty_delta"] * short["inv_price"]).sum())
        detail = "; ".join(
            f"{r.sku}: billed {r.inv_qty:g} vs received {r.grn_qty:g} (+{r.qty_delta:g})" for r in short.itertuples())
        add("QUANTITY_SHORTAGE", "HIGH" if amt >= 5000 else "MEDIUM" if amt >= 1000 else "LOW", amt, detail)

    pv = df[(df["price_pct"] > PRICE_TOLERANCE_PCT) & (df["po_price"] > 0)]
    if not pv.empty:
        amt = float((pv["inv_qty"] * (pv["inv_price"] - pv["po_price"])).sum())
        mx = float(pv["price_pct"].max())
        detail = "; ".join(
            f"{r.sku}: ${r.inv_price:,.2f} vs PO ${r.po_price:,.2f} (+{r.price_pct:.1f}%)" for r in pv.itertuples())
        add("PRICE_VARIANCE", "HIGH" if mx >= 10 else "MEDIUM" if mx >= 5 else "LOW", amt, detail)

    freight, cap = float(inv["freight"]), float(po["freight_cap"])
    unapproved = max(0.0, freight - cap)
    if freight > 0 and unapproved > 0.005:
        add("UNAPPROVED_FREIGHT", "HIGH" if unapproved >= 1000 else "MEDIUM" if unapproved >= 500 else "LOW",
            unapproved, f"Freight billed ${freight:,.2f} vs PO-authorised ${cap:,.2f}")

    if str(inv["bank"]).strip() != vendor["bank"]:
        add("BANK_DETAIL_MISMATCH", "CRITICAL", invoice_total(inv),
            f"Remit-to account {mask(inv['bank'])} does not match vendor master {mask(vendor['bank'])}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# General ledger
# ──────────────────────────────────────────────────────────────────────────────
def billed_parts(inv: dict) -> list[tuple[str, float]]:
    parts: dict[str, float] = {}
    for r in compare_lines(inv).itertuples():
        acct = CATEGORY_ACCOUNT.get(r.category, "1400")
        parts[acct] = parts.get(acct, 0.0) + r.billed
    if inv["freight"] > 0:
        parts["5100"] = parts.get("5100", 0.0) + float(inv["freight"])
    return list(parts.items())


def matched_parts(inv: dict) -> list[tuple[str, float]]:
    parts: dict[str, float] = {}
    for r in compare_lines(inv).itertuples():
        acct = CATEGORY_ACCOUNT.get(r.category, "1400")
        parts[acct] = parts.get(acct, 0.0) + r.matched
    approved_freight = min(float(inv["freight"]), PURCHASE_ORDERS[inv["po"]]["freight_cap"])
    if approved_freight > 0:
        parts["5100"] = parts.get("5100", 0.0) + approved_freight
    return list(parts.items())


def scale_parts(parts: list[tuple[str, float]], target: float) -> list[tuple[str, float]]:
    target = r2(target)
    total = sum(x for _, x in parts)
    if total <= 0 or not parts:
        return [("1400", target)]
    scaled = [(a, r2(x * target / total)) for a, x in parts]
    diff = r2(target - sum(x for _, x in scaled))
    scaled[-1] = (scaled[-1][0], r2(scaled[-1][1] + diff))
    return scaled


def post_gl(inv: dict, parts: list[tuple[str, float]], actor: str, memo: str) -> tuple[str, float]:
    ss = st.session_state
    parts = [(a, r2(x)) for a, x in parts if abs(x) > 0.004]
    total = r2(sum(x for _, x in parts))
    ss.je_counter += 1
    je = f"JE-{ss.je_counter:05d}"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = {"je_id": je, "timestamp": ts, "invoice_id": inv["invoice_id"], "vendor": inv["vendor"],
            "memo": memo, "posted_by": actor}
    for acct, amt in parts:
        ss.gl.append({**base, "account_code": acct, "account_name": ACCOUNTS[acct], "debit": amt, "credit": 0.0})
    ss.gl.append({**base, "account_code": "2000", "account_name": ACCOUNTS["2000"], "debit": 0.0, "credit": total})
    return je, total


def log_decision(inv_id: str, action: str, actor: str, amount: float, note: str = ""):
    st.session_state.decisions.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "invoice_id": inv_id,
        "action": action, "actor": actor, "amount": r2(amount), "note": note,
    })


def ingest_invoice(inv: dict) -> tuple[bool, str]:
    ss = st.session_state
    inv = dict(inv)
    if inv["invoice_id"] in ss.invoices:
        return False, f"Invoice {inv['invoice_id']} already exists in this batch (duplicate blocked)."
    exs = detect_exceptions(inv)
    ss.invoices[inv["invoice_id"]] = inv
    if not exs:
        inv["status"] = "AUTO_POSTED"
        je, total = post_gl(inv, billed_parts(inv), "STP-Engine", f"AP posting {inv['invoice_id']} - 3-way match passed")
        inv["posted_amount"] = total
        log_decision(inv["invoice_id"], "AUTO_POST", "STP-Engine", total, f"Clean 3-way match ({je})")
        return True, f"{inv['invoice_id']} matched cleanly and was auto-posted ({je})."
    inv["status"] = "PENDING_REVIEW"
    ss.exceptions.extend(exs)
    return True, f"{inv['invoice_id']} routed to the exception queue ({len(exs)} exception(s))."


def approve_invoice(inv_id: str, actor: str = "CFO (UI)") -> str:
    inv = st.session_state.invoices[inv_id]
    je, total = post_gl(inv, billed_parts(inv), actor, f"AP posting {inv_id} - approved as billed after exception review")
    inv["status"], inv["posted_amount"] = "APPROVED_POSTED", total
    log_decision(inv_id, "APPROVE_AND_POST", actor, total, f"Approved as billed ({je})")
    return f"✅ {inv_id} approved and posted: {je}, {money(total)}"


def adjust_invoice(inv_id: str, amount: float, note: str, actor: str = "CFO (UI)") -> str:
    inv = st.session_state.invoices[inv_id]
    parts = scale_parts(matched_parts(inv), amount)
    je, total = post_gl(inv, parts, actor, f"AP posting {inv_id} - manual adjustment: {note or 'n/a'}")
    inv["status"], inv["posted_amount"] = "ADJUSTED_POSTED", total
    log_decision(inv_id, "MANUAL_ADJUSTMENT", actor, total, note or "Manual adjustment")
    return f"🛠️ {inv_id} adjusted and posted: {je}, {money(total)}"


def dispute_draft(inv: dict, exs: list[dict]) -> str:
    lines = "\n".join(f"- {e['type']}: {e['detail']} (impact {money(e['amount'])})" for e in exs)
    return (
        f"Subject: Invoice {inv['invoice_id']} on hold - 3-way match discrepancies (PO {inv['po']})\n\n"
        f"Hello {inv['vendor']} Accounts Receivable team,\n\n"
        f"We have placed invoice {inv['invoice_id']} on hold because it failed our PO / goods-receipt / invoice match:\n"
        f"{lines}\n\n"
        "Please send a corrected invoice or credit note. For any change to remittance details, please confirm "
        "through your known account manager - we will not update bank details from email alone.\n\n"
        "Regards,\nAccounts Payable"
    )


def dispute_invoice(inv_id: str, actor: str = "CFO (UI)") -> str:
    inv = st.session_state.invoices[inv_id]
    exs = [e for e in st.session_state.exceptions if e["invoice_id"] == inv_id]
    inv["status"], inv["posted_amount"] = "DISPUTED", 0.0
    inv["dispute_draft"] = dispute_draft(inv, exs)
    log_decision(inv_id, "DISPUTE_VENDOR", actor, 0.0, "Payment held; vendor dispute drafted")
    return f"📨 {inv_id} disputed. Payment held and vendor notice drafted."


def compute_kpis() -> dict:
    ss = st.session_state
    invs = list(ss.invoices.values())
    total = len(invs)
    auto = sum(1 for i in invs if i["status"] == "AUTO_POSTED")
    pending = sum(1 for i in invs if i["status"] == "PENDING_REVIEW")
    disc = sum(e["amount"] for e in ss.exceptions)
    open_disc = sum(e["amount"] for e in ss.exceptions if ss.invoices[e["invoice_id"]]["status"] == "PENDING_REVIEW")
    je_count = len({r["je_id"] for r in ss.gl})
    return {
        "total": total, "auto": auto, "pending": pending, "stp": (auto / total * 100) if total else 0.0,
        "disc": disc, "open_disc": open_disc, "n_exc": len(ss.exceptions), "je": je_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Muse Spark audit workpapers
# ──────────────────────────────────────────────────────────────────────────────
def heuristic_workpaper(inv: dict, exs: list[dict], df: pd.DataFrame) -> dict:
    po = PURCHASE_ORDERS[inv["po"]]
    vendor = VENDORS[po["vendor_id"]]
    paras, anomalies = [], []
    for e in exs:
        t = e["type"]
        if t == "QUANTITY_SHORTAGE":
            paras.append(
                f"Quantity check failed: the vendor billed more units than the goods receipt ({po['grn']}) confirms "
                f"({e['detail']}). Paying as billed would overpay by {money(e['amount'])}. Receiving evidence takes "
                "precedence over the vendor's invoice.")
            anomalies.append(f"Billed quantity exceeds received quantity - {money(e['amount'])} at risk")
        elif t == "PRICE_VARIANCE":
            paras.append(
                f"Price check failed: unit prices exceed the PO price beyond the {PRICE_TOLERANCE_PCT:g}% tolerance "
                f"({e['detail']}). The overcharge is {money(e['amount'])}. Unless a signed price amendment exists, "
                "pay at the PO price.")
            anomalies.append(f"Unit price above contract/PO - {money(e['amount'])} overcharge")
        elif t == "UNAPPROVED_FREIGHT":
            paras.append(
                f"Freight check failed: {e['detail']}. {money(e['amount'])} was never authorised on the PO and needs "
                "budget-owner approval or removal.")
            anomalies.append(f"Unauthorised freight surcharge - {money(e['amount'])}")
        elif t == "BANK_DETAIL_MISMATCH":
            paras.append(
                f"Remittance risk: {e['detail']}. A changed bank account on an otherwise normal invoice is the classic "
                "business-email-compromise / vendor-impersonation pattern. Hold payment and verify by call-back to a "
                "known vendor contact before any change.")
            anomalies.append("Bank account differs from vendor master - possible payment fraud")
    billed, matched = invoice_total(inv), matched_total(inv)
    gap = r2(billed - matched)
    if len(exs) >= 2:
        anomalies.append(f"{len(exs)} independent exception classes on one invoice")
    if gap > 0:
        anomalies.append(f"Billed {money(billed)} vs matched payable {money(matched)} (gap {money(gap)}, {gap / billed * 100:.1f}%)")
    top = max(SEV_RANK[e["severity"]] for e in exs)
    bonus = {4: 0.25, 3: 0.18, 2: 0.10, 1: 0.04}[top]
    conf = round(min(0.99, 0.62 + 0.08 * len(exs) + bonus), 2)
    kinds = {e["type"] for e in exs}
    if "BANK_DETAIL_MISMATCH" in kinds or any(e["type"] == "PRICE_VARIANCE" and e["severity"] == "HIGH" for e in exs):
        rec = "DISPUTE_VENDOR"
    elif kinds & {"QUANTITY_SHORTAGE", "UNAPPROVED_FREIGHT"} and top >= 2:
        rec = "MANUAL_ADJUSTMENT"
    else:
        rec = "APPROVE_AND_POST" if top == 1 and gap < 100 else "MANUAL_ADJUSTMENT"
    return {
        "reasoning": " ".join(paras), "confidence": conf, "anomalies": anomalies, "recommendation": rec,
        "source": "Local heuristic engine (offline demo mode)",
    }


def llm_workpaper(client, inv: dict, exs: list[dict], df: pd.DataFrame, fallback: dict) -> dict:
    vendor = VENDORS[PURCHASE_ORDERS[inv["po"]]["vendor_id"]]
    payload = {
        "invoice": {k: inv[k] for k in ("invoice_id", "po", "vendor", "date", "freight")},
        "invoice_bank_last4": str(inv["bank"])[-4:], "vendor_master_bank_last4": vendor["bank"][-4:],
        "three_way_lines": df.drop(columns=["status"]).round(2).to_dict("records"),
        "exceptions": exs, "price_tolerance_pct": PRICE_TOLERANCE_PCT,
    }
    system = (
        "You are an Accounts Payable audit agent performing a PO vs GRN vs invoice 3-way match review. "
        "Return ONLY a JSON object with keys: reasoning (string, 3-6 sentences, cite numbers), "
        "confidence (float 0-1 that the flagged discrepancies are real), anomalies (array of short strings), "
        "recommendation (one of APPROVE_AND_POST, DISPUTE_VENDOR, MANUAL_ADJUSTMENT). No markdown, no prose outside JSON.")
    text = chat(client, [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, default=str)}], max_tokens=900)
    d = parse_json_loose(text)
    rec = str(d.get("recommendation", "")).upper()
    return {
        "reasoning": str(d.get("reasoning") or fallback["reasoning"]),
        "confidence": round(min(1.0, max(0.0, float(d.get("confidence", fallback["confidence"])))), 2),
        "anomalies": [str(a) for a in (d.get("anomalies") or fallback["anomalies"])][:8],
        "recommendation": rec if rec in ("APPROVE_AND_POST", "DISPUTE_VENDOR", "MANUAL_ADJUSTMENT") else fallback["recommendation"],
        "source": f"Muse Spark · {st.session_state.model} (live)",
    }


def get_workpaper(inv_id: str) -> dict:
    ss = st.session_state
    client = get_client()
    key = f"{inv_id}|{ss.model}|{bool(client)}"
    if key in ss.workpapers:
        return ss.workpapers[key]
    inv = ss.invoices[inv_id]
    exs = [e for e in ss.exceptions if e["invoice_id"] == inv_id]
    df = compare_lines(inv)
    wp = heuristic_workpaper(inv, exs, df)
    if client:
        try:
            wp = llm_workpaper(client, inv, exs, df, wp)
        except Exception as ex:
                        wp["source"] = f"Local heuristic engine (live call failed: {type(ex).__name__}: {str(ex)[:90]})"
    ss.workpapers[key] = wp
    return wp


def cb_regen(inv_id: str):
    ss = st.session_state
    for k in [k for k in ss.workpapers if k.startswith(f"{inv_id}|")]:
        ss.workpapers.pop(k, None)


# ──────────────────────────────────────────────────────────────────────────────
# Voice command handling
# ──────────────────────────────────────────────────────────────────────────────
def transcribe_audio(data: bytes, filename: str) -> tuple[str | None, str]:
    client = get_client()
    if client is None:
        return None, "offline"
    try:
        bio = io.BytesIO(data)
        bio.name = filename or "voice.wav"
        resp = client.audio.transcriptions.create(model=st.session_state.voice_model, file=bio)
        return getattr(resp, "text", str(resp)), "live"
    except Exception as ex:
        return None, f"error: {type(ex).__name__}"


def parse_voice_command(text: str) -> dict:
    t = (text or "").lower()
    action = None
    if re.search(r"\b(approve|post|accept)\b", t):
        action = "APPROVE"
    elif re.search(r"\b(dispute|reject|decline|hold)\b", t):
        action = "DISPUTE"
    m = re.search(r"(?:invoice|inv)\s*(?:number|no\.?|#)?\s*[-#:]?\s*(\d+)", t) or re.search(r"\b(\d{3,})\b", t)
    num = m.group(1) if m else None
    inv_id = None
    if num:
        for i in st.session_state.invoices:
            if re.sub(r"\D", "", i) == num or i.endswith(num):
                inv_id = i
                break
    return {"action": action, "number": num, "invoice_id": inv_id}


def has_critical(inv_id: str) -> bool:
    return any(e["severity"] == "CRITICAL" for e in st.session_state.exceptions if e["invoice_id"] == inv_id)


def cb_approve(inv_id):
    st.session_state.flash = approve_invoice(inv_id)


def cb_dispute(inv_id):
    st.session_state.flash = dispute_invoice(inv_id)


def cb_adjust(inv_id):
    ss = st.session_state
    amt = float(ss.get(f"adj_amt_{inv_id}", 0.0) or 0.0)
    ss.flash = adjust_invoice(inv_id, amt, ss.get(f"adj_note_{inv_id}", ""))


def cb_voice(action: str, inv_id: str, transcript: str):
    ss = st.session_state
    inv = ss.invoices.get(inv_id)
    if not inv or inv["status"] != "PENDING_REVIEW":
        ss.flash = f"Voice command ignored: {inv_id} is not pending review."
    elif action == "APPROVE":
        if has_critical(inv_id):
            log_decision(inv_id, "VOICE_APPROVE_BLOCKED", "Voice (CFO)", 0.0, f"CRITICAL exception; transcript: {transcript}")
            ss.flash = f"⛔ Voice approval blocked for {inv_id}: CRITICAL exception requires manual verification."
        else:
            ss.flash = approve_invoice(inv_id, actor="Voice (CFO)")
            log_decision(inv_id, "VOICE_COMMAND", "Voice (CFO)", 0.0, f"Transcript: {transcript}")
    elif action == "DISPUTE":
        ss.flash = dispute_invoice(inv_id, actor="Voice (CFO)")
        log_decision(inv_id, "VOICE_COMMAND", "Voice (CFO)", 0.0, f"Transcript: {transcript}")


# ──────────────────────────────────────────────────────────────────────────────
# Document ingestion (SAM 3.1 + Muse Image simulation, Muse Spark extraction)
# ──────────────────────────────────────────────────────────────────────────────
def build_regions(seed: str) -> list[dict]:
    rng = random.Random(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    out = []
    for name, (x0, y0, x1, y1), color in BASE_REGIONS:
        def j(v):
            return min(max(v + rng.uniform(-0.008, 0.008), 0.0), 1.0)
        out.append({"label": name, "box": (j(x0), j(y0), j(x1), j(y1)), "color": color,
                    "conf": round(rng.uniform(0.93, 0.995), 3)})
    return out


EXTRACTION_PROMPT = (
    "You are a document-parsing agent for accounts payable. Extract the invoice shown in the image. "
    "Return ONLY a JSON object with exactly these keys: document_type, vendor_name, invoice_number, po_number, "
    "invoice_date (YYYY-MM-DD), currency, bank_account, freight (number), "
    "line_items (array of {sku, description, qty, unit_price}), ocr_confidence (0-1). No prose.")


def normalize_extraction(d: dict) -> dict:
    items = []
    for it in d.get("line_items") or []:
        try:
                        items.append({"sku": str(it.get("sku") or ""), "description": str(it.get("description") or ""),
                          "qty": float(it.get("qty", 0) or 0), "unit_price": float(it.get("unit_price", 0) or 0)})
        except Exception:
            continue
    return {
        "document_type": str(d.get("document_type", "INVOICE")), "vendor_name": str(d.get("vendor_name", "")),
        "invoice_number": str(d.get("invoice_number", "")), "po_number": str(d.get("po_number", "")),
        "invoice_date": str(d.get("invoice_date", "")), "currency": str(d.get("currency", "USD")),
        "bank_account": str(d.get("bank_account") or ""), "freight": float(d.get("freight", 0) or 0),
        "line_items": items or [{"sku": "", "description": "", "qty": 0.0, "unit_price": 0.0}],
        "ocr_confidence": float(d.get("ocr_confidence", 0.9) or 0.9),
    }


def sample_extraction_for(name: str) -> dict:
    d = json.loads(json.dumps(SAMPLE_EXTRACTION))
    n = int(hashlib.md5(name.encode()).hexdigest()[:4], 16) % 900
    d["invoice_number"] = f"INV-{2000 + n}"
    return d


def extract_document(name: str, data: bytes, mime: str) -> tuple[dict, str]:
    client = get_client()
    if client and mime.startswith("image/"):
        try:
            b64 = base64.b64encode(data).decode()
            text = chat(client, [{"role": "user", "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
                max_tokens=4000, temperature=0.0)
            return normalize_extraction(parse_json_loose(text)), "live"
        except Exception as ex:
                        return sample_extraction_for(name), f"simulated (live call failed: {type(ex).__name__}: {str(ex)[:120]})"
    note = "simulated" if not client else "simulated (live extraction supports image files only)"
    return sample_extraction_for(name), note


def build_seg_figure(img, regions: list[dict], visible: list[str], ex: dict):
    W, H = (img.size if img is not None else (850, 1100))
    fig = go.Figure()
    if img is not None:
        fig.add_layout_image(dict(source=img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H,
                                  sizing="stretch", layer="below", xanchor="left", yanchor="top"))
    else:
        fig.add_shape(type="rect", x0=0, y0=0, x1=W, y1=H, fillcolor="#f8fafc", line=dict(color="#94a3b8", width=1), layer="below")

        def pt(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def t(nx, ny, text, size=12, bold=False, anchor="left", color="#0f172a"):
            txt = pt(text)
            fig.add_annotation(x=nx * W, y=ny * H, text=f"<b>{txt}</b>" if bold else txt, showarrow=False,
                               xanchor=anchor, yanchor="middle", font=dict(size=size, color=color))

        items = ex.get("line_items", [])[:6]
        sub = sum(float(i.get("qty", 0) or 0) * float(i.get("unit_price", 0) or 0) for i in items)
        freight = float(ex.get("freight", 0) or 0)
        t(0.07, 0.075, ex.get("vendor_name") or "VENDOR NAME", 17, True)
        t(0.07, 0.118, "Accounts Receivable · Remit-to on file", 10, color="#64748b")
        t(0.93, 0.075, "INVOICE", 22, True, "right", "#334155")
        t(0.07, 0.205, "BILL TO", 10, True, color="#64748b")
        t(0.07, 0.24, "Meridian Holdings Ltd", 12)
        t(0.07, 0.272, "Plot 14, Okhla Phase II, New Delhi", 10, color="#64748b")
        t(0.57, 0.208, f"Invoice No: {ex.get('invoice_number', '')}", 11)
        t(0.57, 0.24, f"PO No: {ex.get('po_number', '')}", 11)
        t(0.57, 0.272, f"Date: {ex.get('invoice_date', '')}", 11)
        fig.add_shape(type="line", x0=0.05 * W, x1=0.95 * W, y0=0.372 * H, y1=0.372 * H, line=dict(color="#cbd5e1", width=1))
        for x, lbl, anc in [(0.07, "SKU", "left"), (0.25, "Description", "left"), (0.62, "Qty", "right"),
                            (0.78, "Unit Price", "right"), (0.93, "Amount", "right")]:
            t(x, 0.35, lbl, 10, True, anc, "#475569")
        for k, it in enumerate(items):
            y = 0.40 + 0.036 * k
            q, p = float(it.get("qty", 0) or 0), float(it.get("unit_price", 0) or 0)
            t(0.07, y, it.get("sku", ""), 10)
            t(0.25, y, it.get("description", ""), 10)
            t(0.62, y, f"{q:g}", 10, anchor="right")
            t(0.78, y, f"{p:,.2f}", 10, anchor="right")
            t(0.93, y, f"{q * p:,.2f}", 10, anchor="right")
        t(0.62, 0.665, "Subtotal", 10, color="#475569")
        t(0.93, 0.665, f"{sub:,.2f}", 10, anchor="right")
        t(0.62, 0.70, "Freight", 10, color="#475569")
        t(0.93, 0.70, f"{freight:,.2f}", 10, anchor="right")
        t(0.62, 0.737, "TOTAL", 11, True)
        t(0.93, 0.737, f"{sub + freight:,.2f} {ex.get('currency', '')}", 11, True, "right")
        t(0.07, 0.815, "REMIT TO · BANK DETAILS", 10, True, color="#64748b")
        t(0.07, 0.85, f"Account: {ex.get('bank_account', '')}", 11)
        t(0.07, 0.882, f"Beneficiary: {ex.get('vendor_name', '')}", 10, color="#64748b")

    for reg in regions:
        if reg["label"] not in visible:
            continue
        x0, y0, x1, y1 = (reg["box"][0] * W, reg["box"][1] * H, reg["box"][2] * W, reg["box"][3] * H)
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=reg["color"], width=3),
                      fillcolor=rgba(reg["color"], 0.13))
        fig.add_annotation(x=x0, y=y0, text=f"<b>{reg['label']} · {reg['conf']:.0%}</b>", showarrow=False,
                           xanchor="left", yanchor="bottom", bgcolor=reg["color"], font=dict(size=11, color="#0b1220"),
                           borderpad=2)
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers", marker=dict(size=40, opacity=0),
            hovertext=f"SAM 3.1 mask: {reg['label']}<br>confidence {reg['conf']:.1%}<br>"
                      f"area {(reg['box'][2] - reg['box'][0]) * (reg['box'][3] - reg['box'][1]) * 100:.1f}% of page",
            hoverinfo="text", showlegend=False))
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, W], visible=False, constrain="domain"),
        yaxis=dict(range=[H, 0], visible=False, scaleanchor="x", scaleratio=1), showlegend=False,
        hovermode="closest")
    return fig, W, H


# ──────────────────────────────────────────────────────────────────────────────
# Analytics agent (Muse Code)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_history() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = [str(p) for p in pd.period_range("2025-11", "2026-10", freq="M")]
    vendors = [("Apex Industrial Supply", "Mechanical", 1.2, 0.05), ("Northwind Components Ltd", "Electronics", 0.8, 0.25),
               ("Orion Packaging Co", "Packaging", 2.0, -0.05), ("Helix Chemicals Inc", "Chemicals", 1.0, 0.12),
               ("Vertex Logistics Parts", "Mechanical", 1.5, 0.02)]
    rows = []
    for i, m in enumerate(months):
        for v, cat, base, drift in vendors:
            n_inv = int(rng.integers(40, 120))
            po_value = float(rng.uniform(80_000, 260_000))
            pv = max(0.0, base + drift * i + float(rng.normal(0, 0.4)))
            inv_value = po_value * (1 + pv / 100)
            short = int(rng.poisson(35 if v.startswith("Helix") else 18))
            stp = float(np.clip(0.62 + 0.01 * i - pv * 0.02 + rng.normal(0, 0.02), 0.4, 0.95) * 100)
            rows.append({
                "month": m, "vendor": v, "category": cat, "invoices": n_inv, "po_value": round(po_value, 2),
                "invoice_value": round(inv_value, 2), "price_variance_pct": round(pv, 2), "shortage_units": short,
                "discrepancy_usd": round(inv_value - po_value + short * float(rng.uniform(8, 30)), 2),
                "stp_rate": round(stp, 1), "qty_shortage_ct": int(rng.poisson(6)), "price_variance_ct": int(rng.poisson(4 + pv)),
                "unapproved_freight_ct": int(rng.poisson(3)), "bank_detail_mismatch_ct": int(rng.poisson(0.3)),
            })
    return pd.DataFrame(rows)


HISTORY_COLUMNS = (
    "month (str 'YYYY-MM'), vendor, category, invoices, po_value, invoice_value, price_variance_pct, shortage_units, "
    "discrepancy_usd, stp_rate (percent), qty_shortage_ct, price_variance_ct, unapproved_freight_ct, bank_detail_mismatch_ct")

TEMPLATES = {
    "price": (
        "Group price variance by vendor and month, then draw one line per vendor.",
        'trend = df.groupby(["month", "vendor"], as_index=False)["price_variance_pct"].mean()\n'
        'fig = px.line(trend, x="month", y="price_variance_pct", color="vendor", markers=True,\n'
        '              title="Vendor price variance trend (% over PO price)")\n'
        'fig.update_yaxes(title="Avg price variance %")'),
    "exceptions": (
        "Sum exception counts per taxonomy class by month and stack them.",
        'cols = {"qty_shortage_ct": "QUANTITY_SHORTAGE", "price_variance_ct": "PRICE_VARIANCE",\n'
        '        "unapproved_freight_ct": "UNAPPROVED_FREIGHT", "bank_detail_mismatch_ct": "BANK_DETAIL_MISMATCH"}\n'
        'm = df.groupby("month", as_index=False)[list(cols)].sum().melt(id_vars="month", var_name="type", value_name="count")\n'
        'm["type"] = m["type"].map(cols)\n'
        'fig = px.bar(m, x="month", y="count", color="type", barmode="stack", title="Exceptions by taxonomy class")'),
    "discrepancy": (
        "Total discrepancy dollars per vendor, ranked.",
        'agg = df.groupby("vendor", as_index=False)["discrepancy_usd"].sum().sort_values("discrepancy_usd", ascending=False)\n'
        'fig = px.bar(agg, x="vendor", y="discrepancy_usd", color="discrepancy_usd", color_continuous_scale="Reds",\n'
        '             title="Discrepancy exposure by vendor (USD, trailing 12 months)")'),
    "stp": (
        "Average straight-through-processing rate per month.",
        's = df.groupby("month", as_index=False)["stp_rate"].mean()\n'
        'fig = px.line(s, x="month", y="stp_rate", markers=True, title="Straight-through processing rate (%)")\n'
        'fig.update_traces(fill="tozeroy")\n'
        'fig.update_yaxes(range=[0, 100], title="STP %")'),
    "spend": (
        "Treemap of invoiced spend by category and vendor, coloured by price variance.",
        'fig = px.treemap(df, path=["category", "vendor"], values="invoice_value", color="price_variance_pct",\n'
        '                 color_continuous_scale="RdYlGn_r", title="Invoiced spend by category / vendor (colour = price variance %)")'),
    "shortage": (
        "Pivot shortage units into a vendor x month heatmap.",
        'piv = df.pivot_table(index="vendor", columns="month", values="shortage_units", aggfunc="sum")\n'
        'fig = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", title="Quantity shortage heatmap (units)")'),
}


def template_key(query: str) -> str:
    q = query.lower()
    if re.search(r"\bstp\b|straight|automation rate", q):
        return "stp"
    if re.search(r"shortage|heatmap|quantity|short", q):
        return "shortage"
    if re.search(r"spend|treemap|category|categories", q):
        return "spend"
    if re.search(r"discrepanc|leak|loss|exposure|overcharge", q):
        return "discrepancy"
    if re.search(r"exception|taxonomy|breakdown|type", q):
        return "exceptions"
    return "price"


BANNED = [r"__", r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(", r"\bcompile\s*\(", r"\bos\.", r"\bsys\.", r"subprocess",
          r"requests", r"socket", r"shutil", r"pathlib", r"globals", r"locals", r"getattr", r"setattr", r"\bread_",
          r"to_csv", r"to_pickle", r"to_parquet", r"to_excel", r"to_sql", r"\binput\s*\("]
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map",
    "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip")}


def run_generated_code(code: str, df: pd.DataFrame) -> go.Figure:
    """Best-effort guarded execution (NOT a hardened sandbox - see README note)."""
    code = "\n".join(l for l in code.splitlines() if not re.match(r"^\s*(import|from)\s+\w", l))
    low = code.lower()
    for pat in BANNED:
        if re.search(pat, low):
            raise ValueError(f"blocked pattern '{pat}' in generated code")
    g = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "px": px, "go": go, "df": df.copy()}
    exec(compile(code, "<muse-code>", "exec"), g)
    fig = g.get("fig")
    if not isinstance(fig, go.Figure):
        raise ValueError("generated code did not assign a Plotly figure to `fig`")
    return fig


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10), height=480, legend_title_text="")
    return fig


def llm_generate_code(client, query: str) -> str:
    system = (
        "You are Muse Code, a data-analysis code generator. Write Python that uses pandas (pd), numpy (np), "
        "plotly.express (px) and plotly.graph_objects (go) on an existing DataFrame named `df` with columns: "
        f"{HISTORY_COLUMNS}. Do NOT import anything, read/write files, or access the network. "
        "Assign the final Plotly Figure to a variable named `fig`. Return ONLY Python code.")
    text = chat(client, [{"role": "system", "content": system}, {"role": "user", "content": query}],
                max_tokens=900, temperature=0.1)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_analysis(query: str, force_offline: bool = False) -> dict:
    df = build_history()
    client = None if force_offline else get_client()
    code, source, err = None, "Muse Code fallback templates (offline)", None
    plan = TEMPLATES[template_key(query)][0]
    if client:
        try:
            code = llm_generate_code(client, query)
            source, plan = f"Muse Code · {st.session_state.model} (live)", "Model-generated analysis plan (see code)."
        except Exception as ex:
            err = f"Live generation failed ({type(ex).__name__}); used local template."
    if code is None:
        code = TEMPLATES[template_key(query)][1]
    try:
        fig = run_generated_code(code, df)
    except Exception as ex:
        err = ((err + " ") if err else "") + f"Generated code was rejected or failed ({ex}); used local template."
        code = TEMPLATES[template_key(query)][1]
        source, plan = "Muse Code fallback templates (offline)", TEMPLATES[template_key(query)][0]
        fig = run_generated_code(code, df)
    return {"query": query, "code": code, "fig": style_fig(fig), "source": source, "err": err, "plan": plan}


def cb_example(q: str):
    st.session_state.nlq = q


# ──────────────────────────────────────────────────────────────────────────────
# UI: sidebar, header, KPIs
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown("### 🧾 Autonomous CFO")
        st.caption("Deep 3-Way Matching Engine · Meta AI stack")
        st.markdown("#### 🔑 Model API")
        st.text_input("Meta Model API / Baseten API key", type="password", key="api_key",
                      help="Leave empty to run the offline demo engine. Can also be set via the META_API_KEY env var.")
        st.text_input("API base URL (OpenAI-compatible)", key="base_url",
                      help="Point this at your Meta Model API or Baseten OpenAI-compatible endpoint, "
                           "e.g. https://inference.baseten.co/v1")
        st.selectbox("Target model", MODEL_OPTIONS, key="model")
        st.text_input("Voice transcription model", key="voice_model")
        if st.button("Test connection", key="btn_test"):
            client = get_client()
            if client is None:
                st.warning("No API key set (or the `openai` package is missing) - running in offline demo mode.")
            else:
                try:
                    with st.spinner("Calling model…"):
                        chat(client, [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=8, temperature=0)
                    st.success("Connection OK - live mode active.")
                except Exception as ex:
                    st.error(f"Connection failed: {type(ex).__name__}: {str(ex)[:180]}")
        if ss.get("api_key") and OpenAI is None:
            st.error("`openai` package not installed: `pip install openai`")
        st.divider()
        st.markdown("#### 📦 Batch & demo data")
        st.caption(f"Active batch: **{ss.get('batch_name', '-')}**")
        st.button("Load Perfect Match Batch", on_click=load_batch, args=("Perfect Match Batch",), key="btn_perfect")
        st.button("Load Fraud & Shortage Batch", on_click=load_batch, args=("Fraud & Shortage Batch",), key="btn_fraud")
        st.divider()
        st.caption("Matching rules: price tolerance 2%, billed qty must not exceed GRN qty, freight must be "
                   "PO-authorised, remit-to account must equal vendor master.")


def render_header():
    pill = ('<span class="mode-pill mode-live">LIVE · Meta API</span>' if live_mode()
            else '<span class="mode-pill mode-offline">OFFLINE DEMO MODE</span>')
    st.markdown(f'<p class="hero-title">Autonomous CFO: Deep 3-Way Matching Engine{pill}</p>'
                '<p class="hero-sub">PO vs. goods receipt vs. invoice reconciliation with human-in-the-loop exception '
                'review, audit workpapers and a balanced GL trail.</p>', unsafe_allow_html=True)


def render_kpis():
    k = compute_kpis()
    cards = [
        ("Total invoices processed", f"{k['total']}", f"{k['auto']} auto-posted · {k['pending']} pending review", "#38bdf8"),
        ("Straight-through processing rate", f"{k['stp']:.1f}%", "invoices posted with no human touch", "#22c55e"),
        ("Total discrepancies flagged", money(k["disc"]), f"{k['n_exc']} exceptions · {money(k['open_disc'])} still open", "#f97316"),
        ("GL entries posted", f"{k['je']}", "double-entry journal entries", "#a78bfa"),
    ]
    cols = st.columns(4)
    for col, (label, value, sub, accent) in zip(cols, cards):
        col.markdown(f'<div class="kpi-card" style="--accent:{accent}"><div class="kpi-label">{h(label)}</div>'
                     f'<div class="kpi-value">{h(value)}</div><div class="kpi-sub">{h(sub)}</div></div>',
                     unsafe_allow_html=True)
    st.write("")


# ──────────────────────────────────────────────────────────────────────────────
# UI: Tab 1 - Ingestion & visual parsing
# ──────────────────────────────────────────────────────────────────────────────
def tab_ingestion():
    ss = st.session_state
    doc = ss.doc
    st.markdown("#### 📥 Ingestion & Visual Parsing")
    up_col, btn_col = st.columns([3, 1.3])
    with up_col:
        up = st.file_uploader("Drag & drop a scanned invoice, receipt or purchase order",
                              type=["png", "jpg", "jpeg", "webp", "pdf"], key="uploader")
    with btn_col:
        st.write("")
        st.write("")
        parse_clicked = st.button("Run SAM 3.1 + Muse Image parsing", type="primary", disabled=up is None, key="btn_parse")
        demo_clicked = st.button("Use demo sample", key="btn_demo")

    if parse_clicked and up is not None:
        data = up.getvalue()
        mime = up.type or "application/octet-stream"
        img = None
        if Image is not None and mime.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((1000, 1400))
            except Exception:
                img = None
        with st.spinner("Deskewing (Muse Image) → segmenting (SAM 3.1) → extracting fields (Muse Spark)…"):
            ex, mode = extract_document(up.name, data, mime)
        ss.doc = {"name": up.name, "image": img, "extracted": ex, "regions": build_regions(up.name),
                  "mode": mode, "ver": doc["ver"] + 1}
        doc = ss.doc
    if demo_clicked:
        ss.doc = {"name": "Sample scanned invoice (demo)", "image": None,
                  "extracted": json.loads(json.dumps(SAMPLE_EXTRACTION)), "regions": build_regions("demo-sample"),
                  "mode": "simulated", "ver": doc["ver"] + 1}
        doc = ss.doc

    ex, ver = doc["extracted"], doc["ver"]
    st.markdown(
        '<div class="chips">'
        '<span class="chip"><b>1</b> Deskew &amp; denoise · Muse Image</span>'
        '<span class="chip"><b>2</b> Region masks · SAM 3.1</span>'
        '<span class="chip"><b>3</b> Field extraction · Muse Spark</span>'
        f'<span class="chip">Source: <b>{h(doc["name"])}</b></span>'
        f'<span class="chip">Extraction mode: <b>{h(doc["mode"])}</b></span></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        visible = st.multiselect("Visible SAM 3.1 regions", REGION_LABELS, default=REGION_LABELS, key="seg_visible")
        fig, W, H = build_seg_figure(doc["image"], doc["regions"], visible, ex)
        _stretch(st.plotly_chart, fig, key="seg_fig", config={"displayModeBar": False})
        st.caption("Region boxes are simulated from a layout prior to demonstrate the SAM 3.1 overlay; "
                   "hover a box for its mask confidence.")
        reg_df = pd.DataFrame([{
            "Region": r["label"], "Confidence": r["conf"],
            "BBox (px)": f"({r['box'][0] * W:.0f}, {r['box'][1] * H:.0f}) - ({r['box'][2] * W:.0f}, {r['box'][3] * H:.0f})",
            "Page area %": round((r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) * 100, 1)} for r in doc["regions"]])
        _stretch(st.dataframe, reg_df, hide_index=True)

    with right:
        st.markdown("##### ✏️ Extracted fields (editable)")
        c1, c2 = st.columns(2)
        inv_no = c1.text_input("Invoice number", value=ex["invoice_number"], key=f"ex_inv_{ver}")
        po_no = c2.text_input("PO number", value=ex["po_number"], key=f"ex_po_{ver}")
        vendor_name = c1.text_input("Vendor name", value=ex["vendor_name"], key=f"ex_vendor_{ver}")
        inv_date = c2.text_input("Invoice date", value=ex["invoice_date"], key=f"ex_date_{ver}")
        bank = c1.text_input("Remit-to bank account", value=ex["bank_account"], key=f"ex_bank_{ver}")
        currency = c2.text_input("Currency", value=ex["currency"], key=f"ex_cur_{ver}")
        freight = st.number_input("Freight billed", min_value=0.0, value=float(ex["freight"]), step=0.01,
                                  format="%.2f", key=f"ex_freight_{ver}")
        st.markdown("**Line items**")
        items_df = pd.DataFrame(ex["line_items"], columns=["sku", "description", "qty", "unit_price"])
        edited = _stretch(
            st.data_editor, items_df, num_rows="dynamic", key=f"ex_lines_{ver}", hide_index=True,
            column_config={
                "sku": st.column_config.TextColumn("SKU"),
                "description": st.column_config.TextColumn("Description"),
                "qty": st.column_config.NumberColumn("Qty", min_value=0.0, step=1.0),
                "unit_price": st.column_config.NumberColumn("Unit price", min_value=0.0, step=0.01, format="%.2f")})
        lines = []
        for r in edited.itertuples():
            if str(r.sku).strip() and str(r.sku).lower() != "nan" and not pd.isna(r.qty) and not pd.isna(r.unit_price):
                lines.append({"sku": str(r.sku).strip(), "description": "" if pd.isna(r.description) else str(r.description),
                              "qty": float(r.qty), "unit_price": float(r.unit_price)})
        subtotal = r2(sum(l["qty"] * l["unit_price"] for l in lines))
        payload = {
            "document_type": ex["document_type"], "vendor": {"name": vendor_name},
            "invoice": {"number": inv_no, "po_number": po_no, "date": inv_date, "currency": currency, "freight": freight},
            "bank_details": {"account_number": bank}, "line_items": lines,
            "totals": {"subtotal": subtotal, "grand_total": r2(subtotal + freight)},
            "extraction": {"mode": doc["mode"], "ocr_confidence": ex["ocr_confidence"], "source_file": doc["name"]},
        }
        with st.expander("Extracted JSON preview", expanded=True):
            st.json(payload)
        if st.button("Send to 3-Way Engine", type="primary", key=f"btn_send_{ver}"):
            if not inv_no.strip():
                st.error("Enter an invoice number first.")
            elif po_no not in PURCHASE_ORDERS:
                st.error(f"PO “{po_no}” was not found in the ERP. Known POs: {', '.join(PURCHASE_ORDERS)}")
            elif not lines:
                st.error("Add at least one line item.")
            else:
                vid = PURCHASE_ORDERS[po_no]["vendor_id"]
                inv = {"invoice_id": inv_no.strip(), "po": po_no, "vendor_id": vid, "vendor": VENDORS[vid]["name"],
                       "date": inv_date, "bank": bank.strip(), "freight": float(freight), "source": "ingestion",
                       "lines": [{"sku": l["sku"], "qty": l["qty"], "price": l["unit_price"]} for l in lines]}
                ok, msg = ingest_invoice(inv)
                if ok:
                    ss.flash = msg
                    st.rerun()
                else:
                    st.error(msg)


# ──────────────────────────────────────────────────────────────────────────────
# UI: Tab 2 - Exception review queue
# ──────────────────────────────────────────────────────────────────────────────
def queue_table_html(rows: list[dict]) -> str:
    head = ("<tr><th>Exception</th><th>Invoice</th><th>Vendor</th><th>Type</th><th>Severity</th>"
            "<th class='num'>Amount</th><th>Status</th></tr>")
    body = ""
    for e in rows:
        status = STATUS_LABEL.get(st.session_state.invoices[e["invoice_id"]]["status"], "")
        body += (f"<tr><td>{h(e['exc_id'])}</td><td>{h(e['invoice_id'])}</td><td>{h(e['vendor'])}</td>"
                 f"<td>{h(e['type'])}</td><td>{badge(e['severity'])}</td><td class='num'>{h(money(e['amount']))}</td>"
                 f"<td><span class='pill'>{h(status)}</span></td></tr>")
    return f"<div class='q-wrap'><table class='q-table'>{head}{body}</table></div>"


def render_voice_section():
    ss = st.session_state
    st.markdown("### 🎙️ Voice approval · Muse Voice Transcribe")
    st.caption("Say e.g. “CFO Agent, approve invoice 402”. Voice approvals are blocked for CRITICAL exceptions.")
    audio = st.audio_input("Record a voice command", key="voice_audio") if hasattr(st, "audio_input") else \
        st.file_uploader("Upload a voice command (wav/mp3/m4a)", type=["wav", "mp3", "m4a", "ogg", "webm"], key="voice_audio")
    if audio is not None:
        data = audio.getvalue()
        hsh = hashlib.md5(data).hexdigest()
        if ss.voice_hash != hsh:
            ss.voice_hash = hsh
            with st.spinner("Transcribing with Muse Voice…"):
                text, status = transcribe_audio(data, getattr(audio, "name", "voice.wav"))
            if text:
                ss.voice_text = text
                ss.voice_ver += 1
                ss.voice_status = "Transcribed live with Muse Voice Transcribe."
            elif status == "offline":
                ss.voice_status = "Offline mode: audio captured but not transcribed. Type or edit the transcript below."
            else:
                ss.voice_status = f"Transcription failed ({status}). Type or edit the transcript below."
    st.caption(ss.voice_status)
    transcript = st.text_input("Transcript (editable)", value=ss.voice_text, key=f"voice_txt_{ss.voice_ver}")
    cmd = parse_voice_command(transcript)
    inv = ss.invoices.get(cmd["invoice_id"]) if cmd["invoice_id"] else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", cmd["action"] or "Not recognised")
    c2.metric("Invoice", cmd["invoice_id"] or ("Not found" if cmd["number"] else "-"))
    c3.metric("Status", STATUS_LABEL.get(inv["status"], "-") if inv else "-")
    ready = bool(cmd["action"] and inv and inv["status"] == "PENDING_REVIEW")
    if ready and cmd["action"] == "APPROVE" and has_critical(cmd["invoice_id"]):
        st.warning("This invoice has a CRITICAL exception. Voice approval will be refused and logged; "
                   "use the manual approval controls with out-of-band bank verification.")
    st.button("Execute voice command", disabled=not ready, on_click=cb_voice, key="btn_voice",
              args=(cmd["action"], cmd["invoice_id"], transcript))


def tab_exceptions():
    ss = st.session_state
    st.markdown("#### 🚨 Exception Review Queue")
    if not ss.exceptions:
        st.success("No exceptions in this batch - every invoice passed the 3-way match and was auto-posted. "
                   "Load the Fraud & Shortage Batch to see the review workflow.")
        return
    c1, c2, c3 = st.columns([2.2, 2, 1.2])
    types = c1.multiselect("Exception taxonomy", EXC_TYPES, default=EXC_TYPES, key="f_types")
    sevs = c2.multiselect("Severity", SEVERITIES, default=SEVERITIES, key="f_sev")
    scope = c3.radio("Show", ["Pending", "All"], horizontal=True, key="f_scope")
    rows = [e for e in ss.exceptions if e["type"] in types and e["severity"] in sevs
            and (scope == "All" or ss.invoices[e["invoice_id"]]["status"] == "PENDING_REVIEW")]
    rows.sort(key=lambda e: (-SEV_RANK[e["severity"]], -e["amount"]))
    if not rows:
        st.info("Nothing matches the current filters. The queue is clear for this view.")
        render_voice_section()
        return
    st.markdown(queue_table_html(rows), unsafe_allow_html=True)
    st.write("")
    ids = list(dict.fromkeys(e["invoice_id"] for e in rows))
    labels = {i: f"{i} · {ss.invoices[i]['vendor']} · {money(invoice_total(ss.invoices[i]))}" for i in ids}
    if ss.get("sel_inv") not in labels.values():
        ss["sel_inv"] = labels[ids[0]]
    chosen = st.selectbox("Select invoice to review", list(labels.values()), key="sel_inv")
    sel = next(i for i, lab in labels.items() if lab == chosen)
    inv = ss.invoices[sel]
    inv_exs = [e for e in ss.exceptions if e["invoice_id"] == sel]
    df = compare_lines(inv)
    po = PURCHASE_ORDERS[inv["po"]]
    vendor = VENDORS[po["vendor_id"]]
    badges = " ".join(badge(e["severity"]) + f" <span class='pill'>{h(e['type'])}</span>" for e in inv_exs)
    st.markdown(
        f"<div class='inv-strip'><b>{h(inv['invoice_id'])}</b> · {h(inv['vendor'])} · PO {h(inv['po'])} · "
        f"{h(po['grn'])} · {h(inv['date'])} · <span class='pill'>{h(STATUS_LABEL[inv['status']])}</span>"
        f"<div class='muted' style='margin-top:6px'>{badges}</div></div>", unsafe_allow_html=True)

    lc, rc = st.columns([1.35, 1])
    with lc:
        st.markdown("##### Side-by-side 3-way comparison")
        show = df.rename(columns={
            "sku": "SKU", "desc": "Description", "po_qty": "PO Qty", "grn_qty": "GRN Received", "inv_qty": "Invoice Billed",
            "qty_delta": "Qty Δ (Inv−GRN)", "po_price": "PO Price", "inv_price": "Invoice Price", "price_pct": "Price Δ (%)",
            "billed": "Billed", "matched": "Matched", "status": "Line status"}).drop(columns=["category"])
        _stretch(st.dataframe, show, hide_index=True, column_config={
            "PO Price": st.column_config.NumberColumn(format="$%.2f"),
            "Invoice Price": st.column_config.NumberColumn(format="$%.2f"),
            "Billed": st.column_config.NumberColumn(format="$%.2f"),
            "Matched": st.column_config.NumberColumn(format="$%.2f"),
            "Price Δ (%)": st.column_config.NumberColumn(format="%.2f")})
        billed, matched = invoice_total(inv), matched_total(inv)
        m1, m2, m3 = st.columns(3)
        m1.metric("Invoice total", money(billed))
        m2.metric("Matched payable", money(matched))
        m3.metric("Overpayment at risk", money(max(0.0, billed - matched)))
        k1, k2 = st.columns(2)
        fr_ok = float(inv["freight"]) <= po["freight_cap"] + 1e-9
        k1.metric("Freight billed vs authorised", f"{money(inv['freight'])} / {money(po['freight_cap'])}",
                  "within cap" if fr_ok else "unauthorised", delta_color="normal" if fr_ok else "inverse")
        bank_ok = str(inv["bank"]).strip() == vendor["bank"]
        k2.metric("Remit-to vs vendor master", f"{mask(inv['bank'])} / {mask(vendor['bank'])}",
                  "match" if bank_ok else "MISMATCH", delta_color="normal" if bank_ok else "inverse")
    with rc:
        wp = get_workpaper(sel)
        with st.container(border=True):
            st.markdown("##### 🧠 Muse Spark audit workpaper")
            st.caption(md_safe(wp["source"]))
            w1, w2 = st.columns(2)
            w1.metric("Confidence", f"{wp['confidence']:.2f}")
            w2.metric("Recommended action", wp["recommendation"].replace("_", " ").title())
            st.progress(float(wp["confidence"]))
            st.markdown("**Reasoning**")
            st.markdown(md_safe(wp["reasoning"]))
            st.markdown("**Flagged anomalies**")
            for a in wp["anomalies"]:
                st.markdown(f"- {md_safe(a)}")
            st.button("↻ Regenerate workpaper", on_click=cb_regen, args=(sel,), key=f"btn_regen_{sel}")

    st.markdown("### ⚡ Resolution actions")
    if inv["status"] != "PENDING_REVIEW":
        st.info(f"{sel} is already resolved: {STATUS_LABEL[inv['status']]}.")
        if inv.get("dispute_draft"):
            st.text_area("Vendor dispute draft", inv["dispute_draft"], height=260, key=f"draft_{sel}")
    else:
        critical = has_critical(sel)
        with st.expander("Manual adjustment parameters"):
            st.number_input("Payable amount to post (USD)", min_value=0.01, value=max(matched_total(inv), 0.01),
                            step=0.01, format="%.2f", key=f"adj_amt_{sel}")
            st.text_input("Adjustment note", value="Adjusted to GRN-received quantities at PO prices", key=f"adj_note_{sel}")
        ack = True
        if critical:
            ack = st.checkbox("I verified the vendor's bank details out-of-band (call-back to a known contact)",
                              key=f"ack_{sel}")
        b1, b2, b3 = st.columns(3)
        b1.button("✅ Approve & Post GL", type="primary", disabled=not ack, on_click=cb_approve, args=(sel,), key=f"btn_ok_{sel}")
        b2.button("📨 Dispute Vendor", on_click=cb_dispute, args=(sel,), key=f"btn_dis_{sel}")
        b3.button("🛠️ Manual Adjustment", on_click=cb_adjust, args=(sel,), key=f"btn_adj_{sel}")
        if not ack:
            st.caption("Approve is locked until the bank-detail verification is confirmed.")
    st.divider()
    render_voice_section()


# ──────────────────────────────────────────────────────────────────────────────
# UI: Tab 3 - Analytics agent
# ──────────────────────────────────────────────────────────────────────────────
def tab_analytics():
    ss = st.session_state
    st.markdown("#### 📊 Dynamic Analytics Agent · Muse Code")
    st.caption("Ask a question in plain English. The agent writes Pandas/Plotly code, validates it and renders the chart.")
    examples = ["Show me vendor price variance trends over time", "Break down exceptions by taxonomy class per month",
                "Which vendors have the highest discrepancy exposure?", "Show the STP rate trend",
                "Heatmap of quantity shortages by vendor and month", "Treemap of spend by category and vendor"]
    cols = st.columns(3)
    for i, q in enumerate(examples):
        cols[i % 3].button(q, on_click=cb_example, args=(q,), key=f"ex_q_{i}")
    q = st.text_input("Your question", key="nlq")
    if st.button("Generate code & run", type="primary", key="btn_run_analysis"):
        with st.spinner("Muse Code is writing and running the analysis…"):
            ss.analysis = run_analysis(q.strip() or DEFAULT_QUERY)
    a = ss.analysis
    st.markdown(f"**Agent trace** · {md_safe(a['source'])}")
    st.caption(f"1) Parse intent: “{md_safe(a['query'])}” → 2) Plan: {md_safe(a['plan'])} → "
               "3) Generate code → 4) Static safety check → 5) Execute and render")
    if a["err"]:
        st.warning(a["err"])
    cc, gc = st.columns([1, 1.5])
    with cc:
        st.markdown("##### Generated code")
        st.code(a["code"], language="python")
    with gc:
        _stretch(st.plotly_chart, a["fig"], key="analysis_fig")
    with st.expander("Dataset schema (12-month vendor history, synthetic)"):
        st.code(HISTORY_COLUMNS)
        _stretch(st.dataframe, build_history().head(10), hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# UI: Tab 4 - GL audit ledger
# ──────────────────────────────────────────────────────────────────────────────
def tab_ledger():
    ss = st.session_state
    st.markdown("#### 📕 General Ledger · Audit Ledger")
    if not ss.gl:
        st.info("No journal entries yet. Clean invoices post automatically; exceptions post after review.")
        return
    gl = pd.DataFrame(ss.gl)
    f1, f2 = st.columns(2)
    inv_f = f1.multiselect("Filter by invoice", sorted(gl["invoice_id"].unique()), key="gl_inv")
    acc_f = f2.multiselect("Filter by account", sorted(gl["account_code"].unique()), key="gl_acc",
                           format_func=lambda a: f"{a} · {ACCOUNTS[a]}")
    view = gl
    if inv_f:
        view = view[view["invoice_id"].isin(inv_f)]
    if acc_f:
        view = view[view["account_code"].isin(acc_f)]
    td, tc = r2(gl["debit"].sum()), r2(gl["credit"].sum())
    per_je = gl.groupby("je_id")[["debit", "credit"]].sum()
    all_balanced = bool(((per_je["debit"] - per_je["credit"]).abs() < 0.005).all())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total debits", money(td))
    m2.metric("Total credits", money(tc))
    m3.metric("Out of balance", money(td - tc))
    m4.metric("Every entry balanced", "Yes ✅" if all_balanced else "NO ❌")
    _stretch(st.dataframe, view[["je_id", "timestamp", "invoice_id", "vendor", "account_code", "account_name",
                                  "debit", "credit", "memo", "posted_by"]], hide_index=True, column_config={
        "debit": st.column_config.NumberColumn(format="$%.2f"), "credit": st.column_config.NumberColumn(format="$%.2f")})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "batch": ss.get("batch_name"),
        "model": ss.model, "mode": "live" if live_mode() else "offline",
        "kpis": compute_kpis(), "gl_entries": ss.gl, "decision_log": ss.decisions, "exceptions": ss.exceptions,
        "audit_workpapers": ss.workpapers,
        "invoices": [{k: v for k, v in i.items() if k != "dispute_draft"} for i in ss.invoices.values()],
    }
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Export Audit Trail (CSV)", gl.to_csv(index=False).encode("utf-8"),
                       file_name=f"gl_audit_trail_{stamp}.csv", mime="text/csv", key="dl_csv")
    d2.download_button("⬇️ Export Audit Trail (JSON)", json.dumps(bundle, indent=2, default=str).encode("utf-8"),
                       file_name=f"gl_audit_trail_{stamp}.json", mime="application/json", key="dl_json")
    with st.expander("Decision log (who did what, when)"):
        if ss.decisions:
            _stretch(st.dataframe, pd.DataFrame(ss.decisions), hide_index=True)
        else:
            st.caption("No decisions logged yet.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(md_safe(flash))
    render_sidebar()
    render_header()
    render_kpis()
    t1, t2, t3, t4 = st.tabs(["📥 Ingestion & Visual Parsing", "🚨 Exception Review Queue",
                              "📊 Dynamic Analytics Agent", "📕 General Ledger Audit"])
    with t1:
        tab_ingestion()
    with t2:
        tab_exceptions()
    with t3:
        tab_analytics()
    with t4:
        tab_ledger()


main()