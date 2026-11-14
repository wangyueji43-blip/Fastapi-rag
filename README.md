# Scientific Paper RAG System

A research-paper question answering system built with **FastAPI**, **Docling**, **Sentence Transformers**, **Hybrid Retrieval (Dense + BM25 + RRF)**, and **Cross-Encoder Reranking**. The system retrieves evidence from scientific PDFs and uses an OpenAI-compatible LLM API to generate grounded answers with source metadata.

> **Project status:** Single-paper research prototype. The current evaluation uses one DCMAGRN paper (155 chunks). Multi-PDF ingestion, production deployment, and large-scale retrieval have not yet been validated.

## Features

- **Document processing:** Docling-based parsing and structure-aware chunking, retaining page numbers, section headings, and content type.
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions).
- **Retrieval:** Dense cosine similarity, BM25 keyword retrieval, and Reciprocal Rank Fusion (RRF).
- **Optional reranking:** `cross-encoder/ms-marco-MiniLM-L6-v2` reorders a larger retrieved candidate pool.
- **Grounded question answering:** Prompt construction from retrieved chunks and answer generation through an OpenAI-compatible endpoint.
- **REST API:** `/rag/search` and `/rag/ask`, with input validation and optional `rerank` control.
- **Testing:** FastAPI route tests and small-scale retrieval/answer comparison experiments.

## Architecture

```text
                      Offline indexing
 PDF (one document) -> Docling parser -> Chunking -> chunks.json
                                                  |
                                                  v
                                          Embedding model
                                                  |
                                                  v
                              embeddings.npy + manifest.json

                       Online retrieval / QA
 User query -> FastAPI -> RetrievalService
                             |             |
                             |        scope filtering
                             |             |
                             +------ Dense + BM25 ------+
                                                        |
                                               RRF (rerank=false)
                                                        |
                              if rerank=true: candidate recall + union
                                                        |
                                               Cross-Encoder reranking
                                                        |
                                          Relevant chunks + metadata
                                                        |
                                 /rag/search: return evidence
                                 /rag/ask: PromptBuilder -> LLM -> answer
```

The offline pipeline creates an index ahead of time. API requests **do not parse or re-embed PDFs on every question**.

## Repository layout

```text
.
├── api/                        # FastAPI RAG routes
├── rag/                        # Retrieval, reranking, indexing, prompting
│   ├── index_store.py
│   ├── docling_retrieval.py
│   ├── docling_fusion.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   └── rag_service.py
├── schemas/                    # Request/response schemas
├── scripts/                    # Index builders and evaluation scripts
├── evaluation/                 # Small retrieval development set
├── requirements/               # Separate Docling environment definitions
├── test/                       # Automated tests
├── config.py                   # Settings and model defaults
├── main.py                     # FastAPI application
├── llm_service.py              # Basic chat API support
├── requirements.txt            # Main API/retrieval dependencies
└── .env.example                # Local API configuration template
```

Some legacy PDF-processing and retrieval modules are kept for experiments; the active Docling retrieval path uses `rag/index_store.py`, `rag/docling_retrieval.py`, and `rag/docling_fusion.py`.

## Setup

### 1. Clone and prepare Python

```bash
git clone https://github.com/Tamato/fastapi-rag-agent.git
cd fastapi-rag-agent

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python **3.10** was used during development. The main requirements record the project's Python dependencies; PyTorch installation can vary by operating system and CPU/CUDA environment. For the separate Docling parsing environment, refer to `requirements/docling-conda.yml` and related files. **A clean-room installation on other machines has not yet been verified.**

### 2. Configure the LLM

```bash
cp .env.example .env
```

Edit `.env` locally:

```dotenv
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
# Optional: override the default model in config.py
# LLM_MODEL=your_model_name
```

`LLM_MODEL` has a default in `config.py`. Replace it or override it to match a model offered by your API provider. **Never commit `.env` or API keys.**

### 3. Prepare the PDF

The current preprocessing scripts use **fixed input/output paths**, not CLI arguments. Place the sample paper at:

```text
data/papers/sample.pdf
```

If your checkout's scripts specify a different PDF input path, either use that exact path or update the path in the script before running it. The sample PDF and generated data are not included in the public repository; obtain a PDF you are permitted to use.

### 4. Build the chunks and embedding index

Run preprocessing in the appropriate environment (the Docling stage may require its separate environment):

```bash
python -m scripts.build_docling_chunks
python -m scripts.build_docling_embedding
```

Expected paths for the current setup:

```text
data/processed/docling/chunks.json
data/processed/docling/index/chunks.json
data/processed/docling/index/embeddings.npy
data/processed/docling/index/manifest.json
```

In the reference experiment, the pipeline produced **155 chunks** and a **155 × 384** normalized embedding matrix. Different documents may yield different counts.

> These scripts currently execute the build process even if called with `--help`; do not append `--help` expecting a usage screen.

### 5. Start the API

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` in the environment where the server is reachable. When running on a remote machine, use an appropriate SSH port forward or network configuration.

## API examples

### `POST /rag/search`

Retrieve evidence without calling the answer-generating LLM:

```bash
curl -X POST http://127.0.0.1:8000/rag/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Which datasets were used to benchmark DCMAGRN?",
    "method": "hybrid",
    "top_k": 3,
    "scope": "body",
    "rerank": true
  }'
```

Parameters:

| Field | Meaning |
| --- | --- |
| `query` | User question |
| `method` | `dense`, `bm25`, or `hybrid` |
| `top_k` | Number of returned chunks |
| `scope` | `body`, `reference`, or `all` |
| `rerank` | `false` for the original retrieval path; `true` to enable Cross-Encoder reranking |

The response includes ranked evidence with `chunk_id`, `content`, `pages`, `headings`, `source`, `score`, and `score_type`. In hybrid mode, `score_type` is `rrf` without reranking and `cross_encoder` with reranking.

### `POST /rag/ask`

Retrieve evidence, construct a prompt, and generate an answer:

```bash
curl -X POST http://127.0.0.1:8000/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Which datasets were used to benchmark DCMAGRN?",
    "method": "hybrid",
    "top_k": 3,
    "scope": "body",
    "rerank": true
  }'
```

The response includes `query`, `answer`, `sources`, and `model`. The `/rag/ask` endpoint requires a valid LLM endpoint and API key. The basic chat routes in `main.py` are separate from the RAG-specific routes.

## Evaluation

The project includes `evaluation/dcmagrn_retrieval_dev.json`, a **four-question development set** with partially annotated relevant evidence. These results are exploratory, not a statistically robust benchmark.

| Method | Known-evidence Hit@3 |
| --- | ---: |
| Hybrid RRF | 3/4 |
| Hybrid + Cross-Encoder | 4/4 |

For the question *“Which datasets were used to benchmark DCMAGRN?”*, an earlier Hybrid RRF run returned:

```text
docling_0044 -> docling_0048 -> docling_0051
```

With reranking enabled, it returned:

```text
docling_0060 -> docling_0005 -> docling_0018
```

The reranked evidence included the GEO accessions **GSE178341** and **GSE200997**, which the answer-generating model was then able to report. These IDs and chunk rankings are tied to the current indexed paper and configuration, not guaranteed for other PDFs.

Run the API regression suite:

```bash
python -m pytest test/test_reg_search_api.py -q
```

At the latest local check, **12 tests passed**. Note that these API tests use a fake retrieval service; the actual retrieval/reranking behavior was also checked separately against the indexed document.

## Limitations and planned extensions

- The reference index covers **one PDF**, not a general-purpose multi-document knowledge base.
- PDF parsing of multi-column scientific pages may introduce occasional reading-order errors.
- The retrieval development set has only four questions and incomplete evidence annotations.
- The current BM25 implementation builds its corpus during retrieval; larger collections would benefit from a persistent/indexed implementation.
- The app assumes an existing offline index; automated PDF ingestion and incremental multi-document updates are not yet implemented.
- API deployment, concurrency, authentication, and clean-environment reproducibility need further validation before production use.

A natural extension is **per-PDF parsing and chunking, followed by aggregation into a shared index with document IDs for filtering and citation tracking**.

## Security and data handling

- Keep `.env`, credentials, model weights, and generated indices out of Git.
- Do not upload copyrighted PDFs without redistribution permission.
- Review experimental outputs before publishing to avoid exposing private content or service endpoints.

## License

No license has been selected yet. Add a `LICENSE` file after choosing the terms under which you want to share this project.
