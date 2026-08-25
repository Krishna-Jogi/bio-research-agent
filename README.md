# Field Notes

A free AI research assistant for BSc/MSc biology, bioscience, and
pharmacovigilance students — built to give quick, accurate answers grounded
in real government and international regulatory documents, instead of
relying on an AI model's own possibly-wrong general knowledge.


## The problem this solves

Students need fast answers to regulatory and technical questions, but the
real source material is scattered across dozens of government PDFs (FSSAI,
WHO, CPCB, ICH, DBT, IPC), and generic AI chatbots will confidently answer
from general knowledge with no real source behind the claim — which is a
serious problem for anything regulatory or safety-related.

Field Notes is built specifically to avoid that: every regulatory answer is
grounded in a real, retrieved document, and the system is honest — in
code, not just in the AI's prose — about when it couldn't find one.

## What it covers

All 9 required academic domains:

| Domain | How it's covered |
|---|---|
| Pharmacovigilance | RAG — IPC/PvPI guidance documents |
| Nutraceuticals | RAG — FSSAI Health Supplements Regulations, 2016 |
| Water | RAG — WHO GMP: Water for Pharmaceutical Use |
| Environment | RAG — CPCB pharmaceutical effluent/emission standards |
| Chemical Engineering | RAG — ICH Q8(R2) Pharmaceutical Development |
| Plant Tissue Culture (PTC) | RAG — THSTI Biosafety Guidelines |
| Research | Live search via Europe PMC |
| Clinical Research | Live search via Europe PMC |
| Mathematical Concepts | Symbolic math via SymPy |

## How it works

1. A student asks a question through the frontend.
2. An LLM agent (via OpenRouter, free tier) decides whether it needs to
   call a tool — search a regulatory document, look up research papers,
   check drug-safety data, or run a calculation — or whether it can answer
   directly.
3. For regulatory questions, the relevant government/international
   document is retrieved from a local vector database (ChromaDB) using
   semantic search over pre-ingested, chunked PDF text.
4. The agent writes an answer grounded in what was actually retrieved.

## Why this is trustworthy, not just another AI wrapper

Free-tier AI models don't always cite sources accurately on their own —
this was confirmed directly during development. So
instead of trusting the model's citation, the **code itself** tracks every
real title and URL a tool actually returns, and appends a verified,
guaranteed-correct source list to every answer — independent of what the
model's own prose says.

Other guardrails built in, based on real failures found during testing:

- **Unverified-answer disclosure**: if no tool was used, the answer says
  so explicitly, in code, not left to the model's discretion.
- **Domain-search caps**: prevents the agent from looping through repeated
  or wrong-domain searches instead of answering.
- **Leaked-reasoning detection**: some free reasoning models can leak
  internal chain-of-thought into the visible answer; this is detected and
  triggers a retry instead of showing the person a confusing monologue.
- **Model fallback**: OpenRouter has twice retired specific free model
  slugs mid-project without warning. The agent now defaults to an
  auto-router that adapts as models change, with a named model as backup.
- **Graceful network failure handling**: an external API timeout no longer
  crashes the whole question.

## Tech stack

- **Backend**: Python, FastAPI
- **Frontend**: HTML/CSS/JS (single file, no framework)
- **Vector database**: ChromaDB (local, persistent)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`, runs locally)
- **PDF processing**: pypdf
- **Math engine**: SymPy
- **LLM access**: OpenRouter (free tier)

## Running it locally

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENROUTER_API_KEY=your_key_here
```

Ingest the source documents into the vector database (one-time, or after
adding new sources):
```powershell
python -m app.rag.ingest
```

Before demoing or presenting, confirm the model connection is live:
```powershell
python -m app.health_check
```

Start the backend:
```powershell
uvicorn app.main:app --reload
```

Then open `frontend/index.html` in a browser.

## Status

All 9 required domains are implemented and have been individually tested
with real questions, verified for accuracy against the actual source
documents.

