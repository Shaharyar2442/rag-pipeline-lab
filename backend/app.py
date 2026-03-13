"""
Flask backend API for the RAG Pipeline frontend.
Loads the global index once at startup and exposes pipeline endpoints.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config_loader import load_config
from src.corpus import get_or_build_index
from src.data_loader import load_examples
from src.pipelines import rag_fusion, hyde, crag, graph_rag

app = Flask(__name__)
CORS(app)  # Allow requests from React dev server

# Global state
_index = None
_config = None
_sample_queries = []

PIPELINES = {
    "rag_fusion": rag_fusion,
    "hyde": hyde,
    "crag": crag,
    "graph_rag": graph_rag,
}


def get_index():
    """Get or initialize the global index."""
    global _index, _config, _sample_queries

    if _index is None:
        print("Loading config...")
        _config = load_config()

        print("Building/loading index...")
        _index = get_or_build_index(_config)

        print("Loading sample queries...")
        _sample_queries = []
        for i, ex in enumerate(load_examples(limit=20)):
            _sample_queries.append({
                "query": ex["query"],
                "domain": ex.get("domain", ""),
                "question_type": ex.get("question_type", ""),
                "answer": ex.get("answer", ""),
            })

        print(f"Backend ready! Index: {_index.faiss_index.ntotal} chunks, "
              f"{len(_sample_queries)} sample queries loaded.")

    return _index, _config


@app.route("/api/query", methods=["POST"])
def handle_query():
    """
    Run a RAG pipeline on a query.

    Request body:
        {
            "query": "Who directed Inception?",
            "pipeline": "rag_fusion",  // rag_fusion | hyde | crag | graph_rag
            "top_k": 5  // optional
        }

    Response:
        {
            "answer": "...",
            "retrieved_chunks": [...],
            "pipeline": "rag_fusion",
            ...pipeline-specific fields
        }
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data["query"]
    pipeline_name = data.get("pipeline", "rag_fusion")
    top_k = data.get("top_k", None)

    if pipeline_name not in PIPELINES:
        return jsonify({
            "error": f"Unknown pipeline: {pipeline_name}. "
                     f"Choose from: {list(PIPELINES.keys())}"
        }), 400

    try:
        index, config = get_index()

        # Override top_k if specified
        run_config = dict(config)
        if top_k is not None:
            run_config["top_k"] = int(top_k)

        # Run the pipeline
        pipeline = PIPELINES[pipeline_name]
        result = pipeline.run(query, index, run_config)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return sample queries from the dataset."""
    get_index()  # Ensure loaded
    return jsonify({"samples": _sample_queries})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting RAG Pipeline backend...")
    print("Loading index on first request (or pre-loading)...")

    # Pre-load the index at startup
    try:
        get_index()
    except Exception as e:
        print(f"Warning: Could not pre-load index: {e}")
        print("Index will be loaded on first request.")

    app.run(host="0.0.0.0", port=5000, debug=False)
