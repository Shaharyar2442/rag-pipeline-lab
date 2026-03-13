"""
Running  all 4 pipelines on the dev set, compute accuracy per pipeline, print or save results.
"""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import load_config
from src.corpus import get_or_build_index
from src.data_loader import load_examples
from src.evaluation import evaluate_single, compute_accuracy

# Import all 4 pipelines
from src.pipelines import rag_fusion, hyde, crag, graph_rag


PIPELINES = {
    "rag_fusion": rag_fusion,
    "hyde": hyde,
    "crag": crag,
    "graph_rag": graph_rag,
}


def run_evaluation(config: dict, limit: int = None, output_path: str = None):
    """
    Run all 4 pipelines on the dev set and compute accuracy.

   
    """
    # Step 1: Build or load the global index
    print("=" * 60)
    print("STEP 1: Loading/Building Global Index")
    print("=" * 60)
    index = get_or_build_index(config)

    # Step 2: Load evaluation examples
    print("\n" + "=" * 60)
    print("STEP 2: Loading Evaluation Examples")
    print("=" * 60)
    examples = list(load_examples(path=config.get("dataset_path"), limit=limit))
    print(f"Loaded {len(examples)} examples")

    # Step 3 & 4: Run each pipeline and evaluate
    all_results = {}

    for pipeline_name, pipeline_module in PIPELINES.items():
        print(f"\n{'=' * 60}")
        print(f"PIPELINE: {pipeline_name.upper()}")
        print(f"{'=' * 60}")

        pipeline_eval_results = []

        for i, example in enumerate(examples):
            query = example["query"]
            gold = example["answer"]
            alts = example.get("alt_ans", [])

            print(f"  [{i+1}/{len(examples)}] {query[:60]}...")

            try:
                result = pipeline_module.run(query, index, config)
                predicted = result.get("answer", "")

                # Evaluate
                eval_result = evaluate_single(predicted, gold, alts)
                eval_result["query"] = query
                eval_result["domain"] = example.get("domain", "")
                eval_result["question_type"] = example.get("question_type", "")
                eval_result["retrieved_chunks"] = result.get("retrieved_chunks", [])

                status = "✓" if eval_result["correct"] else "✗"
                print(f"    {status} Predicted: {predicted[:60]}...")
                print(f"      Gold: {gold[:60]}...")

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                eval_result = {
                    "correct": False,
                    "predicted": f"ERROR: {str(e)}",
                    "gold_answer": gold,
                    "alt_answers": alts,
                    "query": query,
                    "domain": example.get("domain", ""),
                    "question_type": example.get("question_type", ""),
                    "error": str(e),
                }

            pipeline_eval_results.append(eval_result)

        # Compute accuracy for this pipeline
        accuracy_info = compute_accuracy(pipeline_eval_results)
        all_results[pipeline_name] = {
            "accuracy": accuracy_info["accuracy"],
            "correct": accuracy_info["correct"],
            "total": accuracy_info["total"],
            "per_example": pipeline_eval_results,
        }

        print(f"\n  >> {pipeline_name} Accuracy: {accuracy_info['accuracy']:.2%} "
              f"({accuracy_info['correct']}/{accuracy_info['total']})")

    # Step 5: Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Pipeline':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 45)
    for name, result in all_results.items():
        print(f"{name:<15} {result['accuracy']:>9.2%} {result['correct']:>10} {result['total']:>8}")

    # Save results
    if output_path is None:
        output_path = "evaluation_results.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(examples),
        "summary": {
            name: {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
            for name, result in all_results.items()
        },
        "detailed_results": {
            name: result["per_example"]
            for name, result in all_results.items()
        },
    }

    # Remove non-serializable objects from detailed results
    for pipeline_name in output["detailed_results"]:
        for result in output["detailed_results"][pipeline_name]:
            # Ensure all values are JSON-serializable
            for key in list(result.keys()):
                if not isinstance(result[key], (str, int, float, bool, list, dict, type(None))):
                    result[key] = str(result[key])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of examples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_evaluation(config, limit=args.limit, output_path=args.output)
