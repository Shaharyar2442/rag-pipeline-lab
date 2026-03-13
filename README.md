# 🔍 RAG in the Wild — Advanced Retrieval-Augmented Generation

> A comparative study of four advanced RAG strategies on a real-world noisy web corpus, built for the **CS-4015 Agentic AI** course.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-F26522)](https://groq.com)

---

## 📋 Overview

This project implements and evaluates **four advanced RAG (Retrieval-Augmented Generation) pipelines** on the [CRAG dataset](https://github.com/facebookresearch/CRAG) — a challenging benchmark of real web search results spanning finance, sports, movies, music, and general knowledge.

### The Challenge

Users ask factual questions like *"Who directed Inception?"* or *"Which company in the Dow Jones is the best performer?"* — but the system doesn't search the web live. Instead, it relies on a **pre-crawled corpus** where relevance isn't guaranteed: snippets may be ads, tangentially related articles, or fragments that only *look* relevant.

### What's Implemented

| Pipeline | Strategy | Key Technique |
|---|---|---|
| 🔀 **RAG Fusion** | Multi-query retrieval | Generate query variants → retrieve for each → merge via Reciprocal Rank Fusion |
| 🧠 **HyDE** | Hypothetical Document Embedding | Generate a hypothetical answer → embed it → retrieve real docs similar to the hypothesis |
| ✅ **CRAG** | Corrective RAG | Score retrieval confidence → gate through high/medium/low paths → fallback to LLM-only |
| 🕸️ **Graph RAG** | Graph-augmented retrieval | Build entity co-occurrence graph → expand search via graph neighbors → re-rank |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  CRAG Dataset │────▶│  Corpus Builder  │────▶│  FAISS Index     │
│  (4.8 GB)     │     │  (9,532 chunks)  │     │  (384-dim, IP)   │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
┌──────────────┐     ┌─────────────────┐              │
│  User Query   │────▶│  Pipeline Engine │◀─────────────┘
│  (Frontend)   │     │  (4 strategies)  │
└──────────────┘     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  Groq LLM       │
                     │  (Llama 3.3 70B)│
                     └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Python packages
pip install -r requirements.txt

# spaCy model (for Graph RAG NER)
python -m spacy download en_core_web_sm

# Frontend
cd frontend && npm install
```

### 2. Configure

Copy `config/config.example.yaml` → `config/config.yaml` and set your API key:

```yaml
dataset_path: "dataset/crag_task_1_and_2_dev_v4.jsonl"
embedding_model: "all-MiniLM-L6-v2"
generation_model: "llama-3.3-70b-versatile"
api_provider: "groq"          # or "gemini"
api_key: "your-api-key-here"
top_k: 5
index_path: "index/crag_index"
```

> ⚠️ **Do not commit `config.yaml` with API keys.** Use Groq (free tier) or Google Gemini — OpenAI is not allowed.

### 3. Download Dataset

Download the [CRAG Task 1 & 2 dataset](https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2), decompress, and place in `dataset/`:

```
dataset/crag_task_1_and_2_dev_v4.jsonl
```

### 4. Run

```bash
# Build index + run evaluation
python run_evaluation.py --limit 20

# Start backend API (loads index once, serves queries)
python backend/app.py

# Start frontend (in a separate terminal)
cd frontend && npx vite --port 3000
```

Then open **http://localhost:3000** 🎉

---

## 📊 Evaluation Results

Evaluated on 20 examples from the CRAG dev set:

| Pipeline | Accuracy | Correct / Total |
|---|---|---|
| RAG Fusion | 5.0% | 1 / 20 |
| **HyDE** ⭐ | **15.0%** | **3 / 20** |
| CRAG | 0.0% | 0 / 20 |
| Graph RAG | 0.0% | 0 / 20 |

> **Why is accuracy low?** The CRAG benchmark is intentionally hard — questions span diverse domains, the global corpus has sparse coverage, and gold answers require exact factual recall. Refer to [`report.md`](report.md) for detailed analysis.

**Recommendation:** HyDE performs best by bridging the query-document vocabulary gap through hypothetical document generation.

---

## 📁 Project Structure

```
├── config/
│   ├── config.example.yaml      # Template config
│   └── config.yaml              # Your config (gitignored)
├── dataset/
│   └── crag_task_1_and_2_dev_v4.jsonl  # CRAG dataset
├── src/
│   ├── config_loader.py         # YAML config loader
│   ├── corpus.py                # FAISS index build/save/load
│   ├── data_loader.py           # CRAG JSONL parser
│   ├── retrieval.py             # Query embedding + search
│   ├── generation.py            # LLM generation (Groq/Gemini)
│   ├── evaluation.py            # Normalized matching + accuracy
│   └── pipelines/
│       ├── rag_fusion.py        # RAG Fusion (RRF)
│       ├── hyde.py              # HyDE
│       ├── crag.py              # Corrective RAG
│       └── graph_rag.py         # Graph RAG (entity graph)
├── backend/
│   └── app.py                   # Flask API server
├── frontend/
│   └── src/App.jsx              # React UI
├── run_evaluation.py            # Evaluation runner
├── report.md                    # Analysis report
└── requirements.txt             # Python dependencies
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (`IndexFlatIP`, cosine similarity) |
| LLM Inference | Groq API (`llama-3.3-70b-versatile`) |
| NER | spaCy (`en_core_web_sm`) |
| Backend | Flask + Flask-CORS |
| Frontend | React + Vite |
| Dataset | [CRAG](https://github.com/facebookresearch/CRAG) by Meta |

---

## 📄 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/query` | Run a query through a selected pipeline |
| `GET` | `/api/samples` | Get sample queries for the frontend |
| `GET` | `/api/health` | Health check |

**Example request:**
```json
POST /api/query
{
  "query": "Who directed Inception?",
  "pipeline": "hyde"
}
```

---

## 📝 Report

See [`report.md`](report.md) for the detailed analysis comparing all four pipelines, including per-pipeline observations, near-miss analysis, and recommendations.

---

## 📜 License

Academic use — CS-4015 Agentic AI, FAST-NUCES.
