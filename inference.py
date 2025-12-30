"""
Inference script for interactive question answering.
"""

import argparse
import torch
from transformers import AutoTokenizer

from config import get_config
from models import LatentDiffusionQA


class QAInference:
    """Interactive QA inference wrapper."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        null_threshold: float = 0.3,
    ):
        self.config = get_config()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.null_threshold = null_threshold

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_encoder
        )

        # Load model
        self.model = LatentDiffusionQA(
            tokenizer=self.tokenizer,
            latent_dim=self.config.model.vae_latent_dim,
            d_model=self.config.model.denoiser_dim,
            num_layers=self.config.model.denoiser_layers,
            num_heads=self.config.model.denoiser_heads,
            ff_dim=self.config.model.denoiser_ff_dim,
            max_answer_len=self.config.model.max_answer_length,
            num_train_timesteps=self.config.diffusion.num_train_timesteps,
            num_inference_timesteps=self.config.diffusion.num_inference_timesteps,
            schedule_type=self.config.diffusion.schedule_type,
            use_vae=self.config.model.use_vae,
            base_encoder=self.config.model.base_encoder,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.scheduler.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def answer(
        self,
        context: str,
        question: str,
        show_progress: bool = False,
    ) -> dict:
        """
        Answer a question given context.

        Returns:
            dict with 'answer', 'is_unanswerable', 'confidence'
        """
        # Tokenize
        context_enc = self.tokenizer(
            context,
            max_length=self.config.model.max_context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        question_enc = self.tokenizer(
            question,
            max_length=self.config.model.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        context_ids = context_enc["input_ids"].to(self.device)
        context_mask = context_enc["attention_mask"].to(self.device)
        question_ids = question_enc["input_ids"].to(self.device)
        question_mask = question_enc["attention_mask"].to(self.device)

        # Generate
        outputs = self.model.generate(
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            null_threshold=self.null_threshold,
            show_progress=show_progress,
        )

        # Decode
        texts = self.model.decode_tokens_to_text(outputs["tokens"], outputs["is_null"])

        # Normalize cosine similarity from [-1,1] to [0,1] before calculating confidence
        sim = outputs["null_similarity"][0].item()
        normalized_sim = (sim + 1) / 2  # Map [-1,1] -> [0,1]
        confidence = 1.0 - normalized_sim

        return {
            "answer": texts[0],
            "is_unanswerable": outputs["is_null"][0].item(),
            "null_similarity": sim,
            "confidence": confidence,
        }

    def batch_answer(
        self,
        contexts: list,
        questions: list,
    ) -> list:
        """Answer multiple questions in batch."""
        results = []
        for ctx, q in zip(contexts, questions):
            results.append(self.answer(ctx, q))
        return results


def interactive_mode(qa: QAInference):
    """Run interactive QA session."""
    print("\n=== Interactive QA Mode ===")
    print("Enter 'quit' to exit\n")

    while True:
        print("-" * 50)
        context = input("Context: ").strip()
        if context.lower() == "quit":
            break

        question = input("Question: ").strip()
        if question.lower() == "quit":
            break

        result = qa.answer(context, question, show_progress=True)

        print(f"\nAnswer: {result['answer'] if result['answer'] else '[Unanswerable]'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Null Similarity: {result['null_similarity']:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--null_threshold", type=float, default=0.3)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    qa = QAInference(
        checkpoint_path=args.checkpoint,
        null_threshold=args.null_threshold,
    )

    if args.interactive:
        interactive_mode(qa)
    elif args.context and args.question:
        result = qa.answer(args.context, args.question, show_progress=True)
        print(f"\nContext: {args.context}")
        print(f"Question: {args.question}")
        print(f"Answer: {result['answer'] if result['answer'] else '[Unanswerable]'}")
        print(f"Confidence: {result['confidence']:.2%}")
    else:
        print("Please provide --context and --question, or use --interactive mode")


if __name__ == "__main__":
    main()
