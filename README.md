# Indian CA RAG API

A RAG-based Q&A system that answers Indian tax and compliance questions using your own knowledge base.

## Setup

```bash
cd ca_rag
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
uvicorn main:app --reload
```

API docs at: http://localhost:8000/docs

## Usage

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the TDS rate for professional fees under section 194J?"}'
```

### Ingest a PDF (e.g. Finance Act, ICAI circular)
```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@finance_act_2024.pdf"
```

### Ingest URLs (e.g. Income Tax India site pages)
```bash
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://incometaxindia.gov.in/pages/tax-services.aspx"]}'
```

## Suggested Knowledge Sources

| Source | URL |
|--------|-----|
| Income Tax India | https://incometaxindia.gov.in |
| ICAI | https://icai.org |
| GST Council | https://gstcouncil.gov.in |
| MCA (Companies Act) | https://mca.gov.in |

## Architecture

```
User → /ask → retrieve(ChromaDB) → Claude (CA persona) → Answer + Sources
             ↑
PDF / URL → ingest → chunk → embed (Voyage) → ChromaDB
```
