"""
Evaluation script for SQuAD 2.0 with official metrics.
"""

import os
import json
import argparse
import re
import string
from collections import Counter
from typing import Dict, List, Tuple

import torch
from transformers import XLMRobertaTokenizer
from tqdm import tqdm
import wandb
import evaluate

from config import get_config
from data import create_dataloader
from models import LatentDiffusionQA


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_metrics(
    predictions: Dict[str, str],
    ground_truths: Dict[str, List[str]],
    no_answer_probs: Dict[str, float],
    no_answer_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute SQuAD 2.0 metrics.

    Args:
        predictions: Dict mapping qid to predicted answer
        ground_truths: Dict mapping qid to list of ground truth answers
        no_answer_probs: Dict mapping qid to probability of no answer
        no_answer_threshold: Threshold for predicting no answer
    """
    total = 0
    f1_sum = 0.0
    em_sum = 0.0

    has_answer_total = 0
    has_answer_f1 = 0.0
    has_answer_em = 0.0

    no_answer_total = 0
    no_answer_correct = 0

    for qid, pred in predictions.items():
        if qid not in ground_truths:
            continue

        gts = ground_truths[qid]
        total += 1

        # Check if this is a no-answer question
        is_no_answer = len(gts) == 1 and gts[0] == ""

        if is_no_answer:
            no_answer_total += 1
            if pred == "" or (
                qid in no_answer_probs and no_answer_probs[qid] > no_answer_threshold
            ):
                no_answer_correct += 1
                f1_sum += 1.0
                em_sum += 1.0
        else:
            has_answer_total += 1
            # Compute max F1 and EM over all ground truths
            max_f1 = max(f1_score(pred, gt) for gt in gts)
            max_em = max(exact_match_score(pred, gt) for gt in gts)

            f1_sum += max_f1
            em_sum += max_em
            has_answer_f1 += max_f1
            has_answer_em += max_em

    metrics = {
        "exact_match": 100.0 * em_sum / max(total, 1),
        "f1": 100.0 * f1_sum / max(total, 1),
        "total": total,
    }

    if has_answer_total > 0:
        metrics["has_answer_exact_match"] = 100.0 * has_answer_em / has_answer_total
        metrics["has_answer_f1"] = 100.0 * has_answer_f1 / has_answer_total
        metrics["has_answer_total"] = has_answer_total

    if no_answer_total > 0:
        metrics["no_answer_accuracy"] = 100.0 * no_answer_correct / no_answer_total
        metrics["no_answer_total"] = no_answer_total

    return metrics


def load_squad_ground_truths(data_path: str) -> Tuple[Dict, Dict]:
    """Load ground truths from SQuAD file."""
    with open(data_path, "r") as f:
        data = json.load(f)

    ground_truths = {}
    is_impossible = {}

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                is_imp = qa.get("is_impossible", False)
                is_impossible[qid] = is_imp

                if is_imp:
                    ground_truths[qid] = [""]
                else:
                    ground_truths[qid] = [a["text"] for a in qa["answers"]]

    return ground_truths, is_impossible


@torch.no_grad()
def evaluate_model(
    model: LatentDiffusionQA,
    data_loader,
    dataset,
    device: torch.device,
    null_threshold: float = 0.7,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Run model on dataset and collect predictions."""
    model.eval()

    predictions = {}
    no_answer_probs = {}

    for batch in tqdm(data_loader, desc="Evaluating"):
        context_ids = batch["context_input_ids"].to(device)
        context_mask = batch["context_attention_mask"].to(device)
        question_ids = batch["question_input_ids"].to(device)
        question_mask = batch["question_attention_mask"].to(device)
        qids = (
            batch["ids"]
            if "ids" in batch
            else [str(i) for i in range(len(context_ids))]
        )

        outputs = model.generate(
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            null_threshold=null_threshold,
        )

        texts = model.decode_tokens_to_text(outputs["tokens"], outputs["is_null"])

        for i, qid in enumerate(qids):
            predictions[qid] = texts[i]
            no_answer_probs[qid] = outputs["null_similarity"][i].item()

    return predictions, no_answer_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_file", type=str, default="data/dev-v2.0.json")
    parser.add_argument("--output_file", type=str, default="predictions.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--null_threshold", type=float, default=0.7)
    args = parser.parse_args()

    # Handle WandB checkpoint
    if ":" in args.checkpoint and "/" in args.checkpoint:
        print(f"Downloading checkpoint from WandB: {args.checkpoint}")
        run = wandb.init(project="squad-latent-diffusion", job_type="evaluation")
        artifact = run.use_artifact(args.checkpoint)
        artifact_dir = artifact.download()
        # Assuming the checkpoint file is named 'model.pt' or similar in the artifact
        # We'll look for a .pt file
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt") or f.endswith(".pth") or f.endswith(".bin")]
        if not checkpoint_files:
             # If no specific model file found, try using the artifact dir itself if it's a file path or check for standard names
             # But usually artifacts are directories. Let's assume the user points to the artifact and we find the file.
             # If the artifact IS the file, download returns the dir containing it.
             pass
        
        if checkpoint_files:
            args.checkpoint = os.path.join(artifact_dir, checkpoint_files[0])
            print(f"Using checkpoint file: {args.checkpoint}")
        else:
             print(f"Warning: No .pt/.pth/.bin file found in artifact. Using provided path as is if valid.")

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.model.base_encoder)

    # Load model
    print("Loading model...")
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        latent_dim=config.model.vae_latent_dim,
        d_model=config.model.denoiser_dim,
        num_layers=config.model.denoiser_layers,
        num_heads=config.model.denoiser_heads,
        ff_dim=config.model.denoiser_ff_dim,
        max_answer_len=config.model.max_answer_length,
        num_train_timesteps=config.diffusion.num_train_timesteps,
        num_inference_timesteps=config.diffusion.num_inference_timesteps,
        schedule_type=config.diffusion.schedule_type,
        use_vae=config.model.use_vae,
        base_encoder=config.model.base_encoder,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.scheduler.to(device)

    # Load data
    print("Loading data...")
    data_loader, dataset = create_dataloader(
        args.data_file,
        tokenizer,
        args.batch_size,
        max_context_length=config.model.max_context_length,
        max_question_length=config.model.max_question_length,
        max_answer_length=config.model.max_answer_length,
        use_balanced_sampler=False,
        shuffle=False,
    )

    # Load ground truths
    ground_truths, is_impossible = load_squad_ground_truths(args.data_file)

    # Evaluate
    print("Running evaluation...")
    predictions, no_answer_probs = evaluate_model(
        model, data_loader, dataset, device, args.null_threshold
    )

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths, no_answer_probs, 0.5)

    # Compute ROUGE and BLEU
    print("Computing ROUGE and BLEU scores...")
    rouge = evaluate.load("rouge")
    sacrebleu = evaluate.load("sacrebleu")

    # Prepare data for metrics
    # ROUGE/BLEU need list of predictions and list of references (list of lists)
    # We only consider questions that have answers for these metrics usually, 
    # or we can include all. Standard SQuAD eval doesn't usually do ROUGE/BLEU, but for generation it makes sense.
    # We will compute it on all examples.
    
    metric_preds = []
    metric_refs = []
    
    for qid, pred in predictions.items():
        if qid in ground_truths:
            metric_preds.append(pred)
            # references must be a list of strings
            metric_refs.append(ground_truths[qid])

    if metric_preds:
        rouge_results = rouge.compute(predictions=metric_preds, references=metric_refs)
        # BLEU expects references to be list of list of strings
        bleu_results = sacrebleu.compute(predictions=metric_preds, references=metric_refs)
        
        metrics.update(rouge_results)
        metrics["bleu"] = bleu_results["score"]


    print("\n=== Results ===")
    print(f"Exact Match: {metrics['exact_match']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    if "has_answer_exact_match" in metrics:
        print(f"HasAns EM: {metrics['has_answer_exact_match']:.2f}")
        print(f"HasAns F1: {metrics['has_answer_f1']:.2f}")
    if "no_answer_accuracy" in metrics:
        print(f"NoAns Accuracy: {metrics['no_answer_accuracy']:.2f}")
    
    print(f"ROUGE-1: {metrics.get('rouge1', 0):.2f}")
    print(f"ROUGE-2: {metrics.get('rouge2', 0):.2f}")
    print(f"ROUGE-L: {metrics.get('rougeL', 0):.2f}")
    print(f"BLEU: {metrics.get('bleu', 0):.2f}")

    # Save predictions
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {args.output_file}")

    # Save metrics
    metrics_file = args.output_file.replace(".json", "_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
