# Autonomous CFO: Deep 3-Way Matching Engine

**Stop paying invoices that don't match what you ordered and received.**

Built for the Meta Global AI Developer Hackathon Â· Track: Indie developer Â· Theme: Build for Your World

> **Problem statement:** [REPLACE THIS LINE with the problem statement you chose from the event page, in your own words]

---

## The problem

Accounts Payable teams pay vendors against invoices, but invoices are often wrong or fraudulent:

- the vendor bills for more units than were actually received,
- unit prices creep above the agreed purchase order (PO) price,
- freight and surcharges appear that nobody authorised,
- the bank account on the invoice is quietly changed (a common business-email-compromise fraud).

Catching these means a person manually comparing three documents (PO, goods receipt note, invoice) for every invoice. It is slow, boring and easy to get wrong.

## What Autonomous CFO does

It matches every invoice against its **PO** and **goods receipt note (GRN)**, automatically posts the invoices that match cleanly, and sends the rest to a human review queue with an AI-written audit workpaper explaining what is wrong and what to do.

| Tab | What it does |
|---|---|
| **1. Ingestion & Visual Parsing** | Upload a scanned invoice. Muse Spark reads the image and extracts the fields into an editable JSON form before anything reaches the engine. |
| **2. Exception Review Queue** | Severity-ranked exceptions, a side-by-side PO vs GRN vs invoice table, a Muse Spark audit workpaper (reasoning, confidence, anomalies, recommended action), and Approve / Dispute / Manual Adjustment actions. Includes voice-command approval. |
| **3. Dynamic Analytics Agent** | Ask a question in plain English. The model writes Pandas/Plotly code, the app checks it and renders the chart. |
| **4. General Ledger Audit** | Balanced double-entry journal entries (Debit Inventory / COGS / Freight-In, Credit Accounts Payable), a decision log, and one-click CSV / JSON audit-trail export. |

### Exception taxonomy

| Exception | Triggered when | Severity |
|---|---|---|
| `QUANTITY_SHORTAGE` | Billed quantity is greater than the quantity received on the GRN | LOW / MEDIUM / HIGH by dollar value |
| `PRICE_VARIANCE` | Invoice unit price is more than 2% above the PO price | LOW / MEDIUM / HIGH by percentage |
| `UNAPPROVED_FREIGHT` | Freight billed is above what the PO authorises | LOW / MEDIUM / HIGH by dollar value |
| `BANK_DETAIL_MISMATCH` | Remit-to account differs from the vendor master | CRITICAL |

## Design principle: the engine decides, the model explains

Matching, severity and posting are **deterministic code**. The language model never decides a payment amount. It reads the evidence and writes the explanation, which means a model mistake cannot change what gets paid.

Human-in-the-loop safeguards:

- CRITICAL exceptions cannot be approved until the reviewer confirms the bank details were verified out-of-band (call-back to a known contact).
- Voice approval is **refused** for CRITICAL exceptions and the attempt is logged.
- Every action (auto-post, approve, adjust, dispute, voice command) is written to a decision log and exported with the audit trail.
- Every journal entry balances; the ledger tab checks this and shows any out-of-balance amount.
- Duplicate invoice IDs are blocked, and an invoice whose PO is not in the ERP is rejected.

## Where Meta's models are used

All live calls go through an OpenAI-compatible endpoint (served on Baseten) to **Muse Spark Open** (`meta-models/muse-spark-open`).

| Feature | What the model does |
|---|---|
| Audit workpapers | Reads the PO / GRN / invoice evidence and the detected exceptions, then returns reasoning, a confidence score, flagged anomalies and a recommended action as JSON. |
| Invoice extraction (vision) | Reads an uploaded invoice image and returns structured fields (vendor, invoice number, PO number, date, bank account, line items). |
| Analytics agent | Turns a natural-language question into Pandas/Plotly code that is statically checked and executed to draw the chart. |

## Honest status: what is live and what is simulated

**Live (real model calls):** audit workpapers, image-to-JSON invoice extraction, analytics code generation.

**Simulated or mock (not claimed as real):**

- The **SAM 3.1 segmentation overlay** is a UI prototype using fixed layout boxes. It does not call SAM 3.1 yet.
- **Voice approval:** the command parser and safety controls are real, but transcription has not been confirmed on this endpoint, so the transcript can be typed or edited.
- **ERP data** (purchase orders, goods receipts, vendor master) and the **12-month analytics history** are mock data.
- **PDF input** is not parsed yet; image files (PNG / JPG) are.
- Without an API key the app runs on a local fallback engine, so it always works offline.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/Dhruv2020-code/Meta-AI-Hackthon.git
cd Meta-AI-Hackthon
python -m venv venv
```

Activate the environment:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Enable live mode (optional)

Create a `.env` file next to `app.py`:

```
META_API_KEY=your_api_key_here
```

Then in the app sidebar set:

| Field | Value |
|---|---|
| API base URL | `https://inference.baseten.co/v1` |
| Target model | `meta-models/muse-spark-open` |

Click **Test connection**. A green **LIVE** badge appears next to the title when it works. Any OpenAI-compatible endpoint and model ID can be used instead.

Without a key, the badge shows **OFFLINE DEMO MODE** and everything still runs on the local engine.

## Try it in two minutes

1. Open the sidebar and click **Load Fraud & Shortage Batch** (8 invoices).
2. Go to **Exception Review Queue** and open **INV-405**: the invoice's bank account does not match the vendor master. Read the workpaper.
3. Open **INV-402**, then use the voice box with `CFO Agent, approve invoice 402` and click **Execute voice command**.
4. Try the same command for INV-405 and watch it get refused.
5. Go to **Ingestion & Visual Parsing**, upload `sample_invoice_PO-9001.png` and click **Run SAM 3.1 + Muse Image parsing**, then **Send to 3-Way Engine**. INV-3001 appears in the queue with a quantity shortage (60 billed, 55 received).
6. Open **Dynamic Analytics Agent** and ask: `Show me vendor price variance trends over time`.
7. Open **General Ledger Audit** and export the audit trail.

## Project structure

```
Meta-AI-Hackthon/
â”œâ”€â”€ app.py                        # the whole app (engine, UI, model calls)
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ sample_invoice_PO-9001.png    # synthetic invoice for the demo
â”œâ”€â”€ LICENSE
â””â”€â”€ README.md
```

## Roadmap

1. Replace mock data with CSV / Excel upload, then Tally / Zoho Books / ERP integration.
2. Real SAM 3.1 region masks and PDF support.
3. Per-field confidence, with low-confidence fields forced to human review.
4. Approval hierarchy by amount, duplicate-invoice detection, and vendor dispute emails written by the model.
5. Measure precision and recall of flagged exceptions with a real pilot business.

## Notes

- All data in the demo is synthetic. No real invoices or personal data are included.
- Generated analytics code is checked against a blocklist and run with restricted builtins. This is a best-effort guard, not a hardened sandbox, so do not run it on untrusted multi-user deployments as is.
- Never commit your `.env` file or API key.

## License

MIT. See [LICENSE](LICENSE).
