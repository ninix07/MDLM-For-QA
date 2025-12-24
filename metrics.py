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


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute average F1 and EM scores.
    
    Args:
        predictions: List of predicted answer strings.
        references: List of ground truth answer strings.
    """
    if not predictions or not references:
        return {"f1": 0.0, "em": 0.0}

    total_f1 = 0.0
    total_em = 0.0
    count = len(predictions)

    for pred, ref in zip(predictions, references):
        total_f1 += compute_f1(ref, pred)
        total_em += compute_exact(ref, pred)

    return {
        "f1": 100.0 * total_f1 / count,
        "em": 100.0 * total_em / count,
    }
