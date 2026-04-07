# Document Q&A with Citations

A Gradio app that lets you upload a document, ask questions about it, and get answers with inline citations. Hover over any **[N]** badge in the answer to see the exact source passage.

## How it works

1. **Upload** a `.txt`, `.md`, or `.pdf` document
2. The document is split into chunks and indexed using BM25 (and optionally vector search via VoyageAI)
3. **Ask a question** — the top matching chunks are retrieved and sent to Claude with citations enabled
4. Claude returns an answer where each cited span is highlighted and linked to its source passage via a hover tooltip

## Setup

```bash
cd docu-query
source ../venv/bin/activate
pip install gradio pypdf
python app.py
```

Then open [http://localhost:7860](http://localhost:7860).

### Environment variables

Create a `.env` file in the project root (or `docu-query/`) with:

```
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here   # optional — enables hybrid search
```

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `anthropic` | Yes | Claude API + Citations |
| `gradio` | Yes | Web UI |
| `python-dotenv` | Yes | Load `.env` |
| `pypdf` | No | PDF text extraction |
| `voyageai` | No | Vector embeddings for hybrid search |

## Search modes

- **BM25 only** — keyword-based search, works out of the box with no extra API keys
- **Hybrid (BM25 + Vector)** — automatically enabled when a `VOYAGE_API_KEY` is present; uses Reciprocal Rank Fusion (RRF) to merge results from both indexes for better retrieval quality

## Supported file types

| Format | Notes |
|---|---|
| `.txt` | Plain text |
| `.md` | Markdown — sections split on `##` headings |
| `.pdf` | Requires `pip install pypdf` |
