import html
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import gradio as gr
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic client ──────────────────────────────────────────────────────────
anthropic_client = Anthropic()
MODEL = "claude-sonnet-4-5"

# ── Optional: VoyageAI for vector search ─────────────────────────────────────
try:
    import voyageai
    voyage_client = voyageai.Client()
    VOYAGE_AVAILABLE = True
except Exception:
    VOYAGE_AVAILABLE = False


def generate_embedding(chunks):
    if not VOYAGE_AVAILABLE:
        return None
    is_list = isinstance(chunks, list)
    data = chunks if is_list else [chunks]
    result = voyage_client.embed(data, model="voyage-3-large", input_type="query")
    return result.embeddings if is_list else result.embeddings[0]


# ── VectorIndex ───────────────────────────────────────────────────────────────
class VectorIndex:
    def __init__(self, embedding_fn=None):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict] = []
        self._vector_dim: Optional[int] = None
        self._embedding_fn = embedding_fn

    def add_documents(self, documents):
        if not self._embedding_fn:
            return
        vectors = self._embedding_fn([d["content"] for d in documents])
        if vectors is None:
            return
        for vec, doc in zip(vectors, documents):
            if not self.vectors:
                self._vector_dim = len(vec)
            self.vectors.append(list(vec))
            self.documents.append(doc)

    def search(self, query, k=1):
        if not self.vectors or not self._embedding_fn:
            return []
        qv = self._embedding_fn(query)
        if qv is None:
            return []
        dists = sorted(
            [(self._cos_dist(qv, sv), doc) for sv, doc in zip(self.vectors, self.documents)],
            key=lambda x: x[0],
        )
        return [(doc, d) for d, doc in dists[:k]]

    def _cos_dist(self, v1, v2):
        m1 = math.sqrt(sum(x * x for x in v1))
        m2 = math.sqrt(sum(x * x for x in v2))
        if m1 == 0 or m2 == 0:
            return 1.0
        return 1.0 - max(-1.0, min(1.0, sum(a * b for a, b in zip(v1, v2)) / (m1 * m2)))


# ── BM25Index ─────────────────────────────────────────────────────────────────
class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.documents: List[Dict] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len = 0.0
        self._idf: Dict[str, float] = {}
        self._built = False
        self.k1, self.b = k1, b

    def _tok(self, text):
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def add_documents(self, documents):
        for doc in documents:
            tokens = self._tok(doc["content"])
            self.documents.append(doc)
            self._corpus_tokens.append(tokens)
            self._doc_len.append(len(tokens))
            for t in set(tokens):
                self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1
        self._built = False

    def _build(self):
        n = len(self.documents)
        self._avg_doc_len = sum(self._doc_len) / n if n else 0
        self._idf = {
            t: math.log(((n - f + 0.5) / (f + 0.5)) + 1)
            for t, f in self._doc_freqs.items()
        }
        self._built = True

    def search(self, query, k=1):
        if not self.documents:
            return []
        if not self._built:
            self._build()
        qt = self._tok(query)
        scores = []
        for i, tokens in enumerate(self._corpus_tokens):
            tc = Counter(tokens)
            score = sum(
                self._idf.get(t, 0) * tc.get(t, 0) * (self.k1 + 1)
                / (tc.get(t, 0) + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_doc_len) + 1e-9)
                for t in qt
            )
            if score > 1e-9:
                scores.append((math.exp(-0.1 * score), self.documents[i]))
        scores.sort(key=lambda x: x[0])
        return [(doc, s) for s, doc in scores[:k]]


# ── Retriever (RRF fusion) ────────────────────────────────────────────────────
class Retriever:
    def __init__(self, *indexes):
        self._indexes = list(indexes)

    def add_documents(self, documents):
        for idx in self._indexes:
            idx.add_documents(documents)

    def search(self, query, k=5, k_rrf=60):
        all_results = [idx.search(query, k=k * 5) for idx in self._indexes]
        ranks: Dict[int, Dict] = {}
        for src, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                did = id(doc)
                if did not in ranks:
                    ranks[did] = {"doc": doc, "r": [float("inf")] * len(self._indexes)}
                ranks[did]["r"][src] = rank + 1
        scored = [
            (info["doc"], sum(1.0 / (k_rrf + r) for r in info["r"] if r != float("inf")))
            for info in ranks.values()
        ]
        scored = sorted([(d, s) for d, s in scored if s > 0], key=lambda x: x[1], reverse=True)
        return scored[:k]


# ── Document loading & chunking ───────────────────────────────────────────────
def extract_text(file_path: str) -> Tuple[str, str]:
    """Return (text, error_message). One will be empty."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), ""
    if ext == ".pdf":
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(file_path)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            return text, ""
        except ImportError:
            return "", "pypdf not installed — run: pip install pypdf"
        except Exception as e:
            return "", f"PDF read error: {e}"
    return "", f"Unsupported file type: {ext}"


def smart_chunk(text: str) -> List[str]:
    """Split by markdown sections, then paragraphs, capped at ~1200 chars."""
    # Try markdown sections first
    sections = re.split(r"\n#{1,3} ", text)
    if len(sections) > 3:
        return [s.strip() for s in sections if s.strip()]

    # Fall back to paragraphs, merging short ones
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < 1200:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks or [text]


def build_retriever(chunks: List[str]) -> "Retriever":
    docs = [{"content": c} for c in chunks]
    bm25 = BM25Index()
    vec = VectorIndex(embedding_fn=generate_embedding if VOYAGE_AVAILABLE else None)
    bm25.add_documents(docs)
    if VOYAGE_AVAILABLE:
        vec.add_documents(docs)
        return Retriever(bm25, vec)
    return Retriever(bm25)


# ── HTML rendering ────────────────────────────────────────────────────────────
def render_answer_html(content_blocks) -> str:
    """
    Build answer HTML with inline tooltip citations.
    Hovering over a [N] badge shows the cited passage in a dark popup.
    """
    parts = []
    cit_counter = 0

    for block in content_blocks:
        if block.type != "text" or not block.text.strip():
            continue

        safe_text = html.escape(block.text)

        if block.citations:
            badges = []
            for cit in block.citations:
                cit_counter += 1
                snippet = html.escape(cit.cited_text.strip().replace("\n", " "))
                if len(snippet) > 350:
                    snippet = snippet[:350] + "…"
                source = html.escape(cit.document_title or "Source")
                badges.append(
                    f'<span class="cite-wrap">'
                    f'<sup class="cite-badge">[{cit_counter}]'
                    f'<span class="cite-tip">'
                    f'<span class="cite-tip-source">{source}</span>'
                    f'<span class="cite-tip-text">"{snippet}"</span>'
                    f'</span>'
                    f'</sup>'
                    f'</span>'
                )
            parts.append(
                f'<span class="cited-text">{safe_text}</span>' + "".join(badges)
            )
        else:
            parts.append(safe_text)

    return "<p class='answer-body'>" + " ".join(parts) + "</p>"


def build_chunks_html(results: List[Tuple[Dict, float]]) -> str:
    if not results:
        return "<p class='muted'>No chunks retrieved.</p>"
    items = []
    for i, (doc, score) in enumerate(results, 1):
        preview = html.escape(doc["content"].strip()[:380].replace("\n", " "))
        if len(doc["content"]) > 380:
            preview += "…"
        items.append(
            f'<div class="chunk-card">'
            f'<div class="chunk-header">'
            f'<span class="chunk-num">Chunk {i}</span>'
            f'<span class="chunk-score">RRF {score:.4f}</span>'
            f'</div>'
            f'<p class="chunk-preview">{preview}</p>'
            f'</div>'
        )
    return "\n".join(items)


# ── Query handler ─────────────────────────────────────────────────────────────
def query_document(user_query: str, top_k: int, state: Dict):
    if not user_query.strip():
        return build_chunks_html([]), "<p class='muted'>Enter a question above.</p>"
    if state.get("retriever") is None:
        return build_chunks_html([]), "<p class='muted'>Please upload a document first.</p>"

    retriever: Retriever = state["retriever"]
    results = retriever.search(user_query, k=int(top_k))
    chunks_html = build_chunks_html(results)

    doc_blocks = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": doc["content"]},
            "title": f"Chunk {i + 1}",
            "citations": {"enabled": True},
        }
        for i, (doc, _) in enumerate(results)
    ]
    doc_blocks.append({"type": "text", "text": user_query})

    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": doc_blocks}],
    )

    answer_html = render_answer_html(response.content)
    return chunks_html, answer_html


def upload_document(file, state: Dict):
    if file is None:
        return state, "<span class='status-idle'>No file selected.</span>"

    text, err = extract_text(file.name)
    if err:
        return state, f"<span class='status-err'>{html.escape(err)}</span>"
    if not text.strip():
        return state, "<span class='status-err'>File appears to be empty.</span>"

    chunks = smart_chunk(text)
    retriever = build_retriever(chunks)
    doc_name = os.path.basename(file.name)
    new_state = {"retriever": retriever, "doc_name": doc_name, "n_chunks": len(chunks)}

    mode = "Hybrid (BM25 + Vector)" if VOYAGE_AVAILABLE else "BM25"
    msg = (
        f"<span class='status-ok'>"
        f"Loaded <strong>{html.escape(doc_name)}</strong> — "
        f"{len(chunks)} chunks indexed &nbsp;·&nbsp; {mode}"
        f"</span>"
    )
    return new_state, msg


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', system-ui, sans-serif; }
.gradio-container { max-width: 1100px !important; margin: 0 auto; }

/* ── Header ── */
.app-header { text-align: center; padding: 28px 0 12px; }
.app-header h1 { font-size: 1.9rem; font-weight: 800; color: #111827; margin: 0; }
.app-header p  { color: #6b7280; margin: 6px 0 0; font-size: 0.95rem; }

/* ── Upload bar ── */
.upload-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 4px 0; }
.status-ok  { color: #16a34a; font-size: 0.88rem; font-weight: 500; }
.status-err { color: #dc2626; font-size: 0.88rem; font-weight: 500; }
.status-idle{ color: #9ca3af; font-size: 0.88rem; }

/* ── Chunk cards ── */
.chunk-card {
  border: 1px solid #e5e7eb; border-radius: 10px;
  padding: 12px 15px; margin-bottom: 10px; background: #f9fafb;
}
.chunk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.chunk-num   { font-weight: 700; font-size: 0.83rem; color: #374151; }
.chunk-score { font-size: 0.75rem; color: #6b7280; background: #e5e7eb;
               padding: 2px 7px; border-radius: 999px; }
.chunk-preview { font-size: 0.86rem; color: #4b5563; margin: 0; line-height: 1.5; }

/* ── Answer body ── */
.answer-body {
  font-size: 1rem; line-height: 1.85; color: #111827;
  margin: 0; padding: 0;
}
.cited-text {
  background: linear-gradient(120deg, #fef9c3 0%, #fde68a 100%);
  border-radius: 3px; padding: 1px 3px;
}

/* ── Citation badge + tooltip ── */
.cite-wrap { display: inline; position: relative; }

.cite-badge {
  display: inline-block;
  font-size: 0.68rem; font-weight: 800;
  color: #fff; background: #2563eb;
  border-radius: 4px; padding: 1px 5px;
  margin-left: 2px; cursor: default;
  vertical-align: super; line-height: 1;
  position: relative;
  z-index: 0;
}

/* Tooltip bubble */
.cite-tip {
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  position: absolute;
  z-index: 9999;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  width: 310px;
  background: #1e293b;
  color: #f1f5f9;
  border-radius: 10px;
  padding: 11px 14px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.35);
  transition: opacity 0.15s ease;
  font-style: normal;
  font-weight: 400;
  line-height: normal;
  text-align: left;
  white-space: normal;
  vertical-align: baseline;
}
.cite-tip::after {
  content: "";
  position: absolute;
  top: 100%; left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: #1e293b;
}
.cite-tip-source {
  display: block;
  font-size: 0.72rem;
  font-weight: 700;
  color: #93c5fd;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 5px;
}
.cite-tip-text {
  display: block;
  font-size: 0.83rem;
  color: #e2e8f0;
  font-style: italic;
  line-height: 1.5;
}
.cite-badge:hover .cite-tip {
  visibility: visible;
  opacity: 1;
}

/* ── Answer box wrapper ── */
#answer-wrap {
  border: 1px solid #e5e7eb; border-radius: 10px;
  padding: 18px 20px; background: #fff; min-height: 140px;
}

/* ── Section labels ── */
.sec-label {
  font-size: 0.74rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 10px;
}
.muted { color: #9ca3af; font-size: 0.9rem; }
"""

# ── Gradio layout ─────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Document Q&A with Citations") as demo:

    state = gr.State({"retriever": None, "doc_name": None, "n_chunks": 0})

    gr.HTML("""
      <div class="app-header">
        <h1>Document Q&amp;A with Citations</h1>
        <p>Upload a document, ask a question — hover over <strong style="color:#2563eb">[N]</strong> to see the source passage.</p>
      </div>
    """)

    # ── Upload row ────────────────────────────────────────────────────────────
    with gr.Row(elem_classes="upload-row"):
        file_upload = gr.File(
            label="Upload document (.txt, .md, .pdf)",
            file_types=[".txt", ".md", ".pdf"],
            scale=3,
        )
        upload_status = gr.HTML(
            value="<span class='status-idle'>No document loaded.</span>",
            scale=4,
        )

    # ── Query row ─────────────────────────────────────────────────────────────
    with gr.Row():
        query_box = gr.Textbox(
            placeholder="e.g.  What happened in the INC-2023-Q4-011 incident?",
            label="Your question",
            lines=2,
        )

    ask_btn = gr.Button("Ask", variant="primary", size="lg")

    # ── Answer ────────────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-label">Answer</div>')
    answer_out = gr.HTML(
        value="<div id='answer-wrap'><p class='muted'>Answer will appear here.</p></div>"
    )


    # ── Events ────────────────────────────────────────────────────────────────
    file_upload.upload(
        fn=upload_document,
        inputs=[file_upload, state],
        outputs=[state, upload_status],
    )

    def ask(query, s):
        _, answer_html = query_document(query, 3, s)
        return f"<div id='answer-wrap'>{answer_html}</div>"

    ask_btn.click(fn=ask, inputs=[query_box, state], outputs=[answer_out])
    query_box.submit(fn=ask, inputs=[query_box, state], outputs=[answer_out])

if __name__ == "__main__":
    demo.launch()
