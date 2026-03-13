"""
Comparing predicted answer to gold answer and alt_ans; compute accuracy.
"""

import re
import string
from typing import List, Dict, Optional


def normalize(text: str) -> str:
    """
    Normalizing text for comparison by lowercasing, removing articles, punctuation, and extra whitespace.
    """
    if not text:
        return ""

    text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_correct(predicted: str, gold_answer: str, alt_answers: Optional[List[str]] = None) -> bool:
    """
    Checking if the predicted answer matches the gold answer or any alternative answer.
    
    Matching logic:
    1. Exact normalized match.
    2. Gold answer is contained in (or contains) the predicted answer.
    3. Same checks for each alternative answer.

    
    Returns:
        True if the prediction is considered correct.
    """
    if not predicted:
        return False

    norm_pred = normalize(predicted)
    norm_gold = normalize(gold_answer)

    if not norm_gold:
        return False

    # Check against gold answer
    if _matches(norm_pred, norm_gold):
        return True

    # Check against alternative answers
    if alt_answers:
        for alt in alt_answers:
            if alt and _matches(norm_pred, normalize(alt)):
                return True

    return False


def _matches(norm_predicted: str, norm_gold: str) -> bool:
    """
    Ftn to check if normalized predicted matches normalized gold.
    Uses exact match AND containment 
    """
    if not norm_gold or not norm_predicted:
        return False

    # Exact match
    if norm_predicted == norm_gold:
        return True

    # Gold is contained in prediction 
    if norm_gold in norm_predicted:
        return True

    # Prediction is contained in gold 
    if norm_predicted in norm_gold:
        return True

    # Check if all words of gold appear in prediction (for multi-word answers)
    gold_words = set(norm_gold.split())
    pred_words = set(norm_predicted.split())
    if gold_words and gold_words.issubset(pred_words):
        return True

    return False


def evaluate_single(predicted: str, gold_answer: str, alt_answers: Optional[List[str]] = None) -> Dict:
    """
    Evaluate a single prediction.

    Returns:
        Dict with 'correct' (bool), 'predicted', 'gold_answer', 'alt_answers'.
    """
    correct = is_correct(predicted, gold_answer, alt_answers)
    return {
        "correct": correct,
        "predicted": predicted,
        "gold_answer": gold_answer,
        "alt_answers": alt_answers or [],
    }


def compute_accuracy(results: List[Dict]) -> Dict:
    """
    Compute accuracy from a list of evaluation results.

    Args:
        results: List of dicts, each with at least a 'correct' key (bool).

    Returns:
        Dict with 'accuracy' (float), 'correct' (int), 'total' (int).
    """
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    return {
        "accuracy": round(correct / total, 4),
        "correct": correct,
        "total": total,
    }


def evaluate_pipeline(
    pipeline_results: List[Dict],
    examples: List[Dict],
) -> Dict:
    """
    Evaluate a full pipeline's predictions against the dataset.

    Args:
        pipeline_results: List of pipeline output dicts (each with 'answer' key).
        examples: List of dataset examples (each with 'answer' and 'alt_ans').

    Returns:
        Dict with per-example results and overall accuracy.
    """
    eval_results = []
    for pred_result, example in zip(pipeline_results, examples):
        predicted = pred_result.get("answer", "")
        gold = example.get("answer", "")
        alts = example.get("alt_ans", [])

        eval_result = evaluate_single(predicted, gold, alts)
        eval_result["query"] = example.get("query", "")
        eval_result["domain"] = example.get("domain", "")
        eval_result["question_type"] = example.get("question_type", "")
        eval_results.append(eval_result)

    accuracy_info = compute_accuracy(eval_results)
    return {
        "per_example": eval_results,
        **accuracy_info,
    }
