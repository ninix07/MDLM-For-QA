"""
Evaluation metrics for SQuAD 2.0 (F1 and Exact Match).
"""

import collections
import re
import string
from typing import List, Dict


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_detailed_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute detailed F1 and EM scores, broken down by class (Answerable vs Unanswerable).
    
    Returns:
        Dict containing:
        - f1, em: Overall scores
        - has_ans_f1, has_ans_em: Scores for answerable questions
        - no_ans_f1, no_ans_em: Scores for unanswerable questions
        - total_preds: Total number of predictions
        - total_has_ans: Total answerable references
        - total_no_ans: Total unanswerable references
        - pred_has_ans: Total predicted as answerable
        - pred_no_ans: Total predicted as unanswerable
    """
    if not predictions or not references:
        return {
            "f1": 0.0, "em": 0.0,
            "has_ans_f1": 0.0, "has_ans_em": 0.0,
            "no_ans_f1": 0.0, "no_ans_em": 0.0,
        }

    total_f1 = 0.0
    total_em = 0.0
    
    has_ans_f1 = 0.0
    has_ans_em = 0.0
    has_ans_count = 0
    
    no_ans_f1 = 0.0
    no_ans_em = 0.0
    no_ans_count = 0
    
    pred_has_ans_count = 0
    pred_no_ans_count = 0

    for pred, ref in zip(predictions, references):
        # Calculate individual metrics
        f1 = compute_f1(ref, pred)
        em = compute_exact(ref, pred)
        
        total_f1 += f1
        total_em += em
        
        # Determine if reference is "No Answer"
        # In our pipeline, NoAns is represented as empty string ""
        is_no_ans_ref = (len(get_tokens(ref)) == 0)
        
        # Determine if prediction is "No Answer"
        is_no_ans_pred = (len(get_tokens(pred)) == 0)
        
        if is_no_ans_pred:
            pred_no_ans_count += 1
        else:
            pred_has_ans_count += 1
            
        if is_no_ans_ref:
            no_ans_count += 1
            no_ans_f1 += f1
            no_ans_em += em
        else:
            has_ans_count += 1
            has_ans_f1 += f1
            has_ans_em += em

    count = len(predictions)
    
    metrics = {
        "f1": 100.0 * total_f1 / count,
        "em": 100.0 * total_em / count,
        "total_preds": count,
        "total_has_ans": has_ans_count,
        "total_no_ans": no_ans_count,
        "pred_has_ans": pred_has_ans_count,
        "pred_no_ans": pred_no_ans_count,
    }
    
    if has_ans_count > 0:
        metrics["has_ans_f1"] = 100.0 * has_ans_f1 / has_ans_count
        metrics["has_ans_em"] = 100.0 * has_ans_em / has_ans_count
    else:
        metrics["has_ans_f1"] = 0.0
        metrics["has_ans_em"] = 0.0
        
    if no_ans_count > 0:
        metrics["no_ans_f1"] = 100.0 * no_ans_f1 / no_ans_count
        metrics["no_ans_em"] = 100.0 * no_ans_em / no_ans_count
    else:
        metrics["no_ans_f1"] = 0.0
        metrics["no_ans_em"] = 0.0
        
    return metrics


# Backward compatibility alias
compute_metrics = compute_detailed_metrics
