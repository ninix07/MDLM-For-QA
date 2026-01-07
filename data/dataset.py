"""
SQuAD 2.0 Dataset with balanced sampling for answerable/unanswerable questions.
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer


@dataclass
class SQuADExample:
    """Single SQuAD 2.0 example."""

    id: str
    context: str
    question: str
    answer: str
    is_impossible: bool
    answer_start: Optional[int] = None


class SQuAD2Dataset(Dataset):
    """
    SQuAD 2.0 Dataset for Latent Diffusion.

    Handles:
    - Loading and parsing SQuAD 2.0 JSON format
    - Converting unanswerable questions to <NULL_ANS> target
    - Tokenization using XLM-RoBERTa
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_context_length: int = 384,
        max_question_length: int = 64,
        max_answer_length: int = 64,
        null_ans_token: str = "<NULL_ANS>",
    ):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.null_ans_token = null_ans_token

        # Add special token if not present
        if null_ans_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [null_ans_token]})

        self.null_ans_token_id = tokenizer.convert_tokens_to_ids(null_ans_token)

        # Load data
        self.examples = self._load_squad(data_path)

        # Separate answerable and unanswerable for balanced sampling
        self.answerable_indices = [i for i, ex in enumerate(self.examples) if not ex.is_impossible]
        self.unanswerable_indices = [i for i, ex in enumerate(self.examples) if ex.is_impossible]

        print(f"Loaded {len(self.examples)} examples:")
        print(f"  - Answerable: {len(self.answerable_indices)}")
        print(f"  - Unanswerable: {len(self.unanswerable_indices)}")

        # Pre-tokenize all data to store in RAM
        print("Pre-tokenizing all data (this may take a moment...)")
        self._pre_tokenize_all()

    def _load_squad(self, data_path: str) -> List[SQuADExample]:
        """Load SQuAD 2.0 JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]

                for qa in paragraph["qas"]:
                    question = qa["question"]
                    qid = qa["id"]
                    is_impossible = qa.get("is_impossible", False)

                    if is_impossible:
                        # Unanswerable: use <NULL_ANS> as target
                        answer = self.null_ans_token
                        answer_start = None
                    else:
                        # Answerable: use first answer
                        if qa["answers"]:
                            answer = qa["answers"][0]["text"]
                            answer_start = qa["answers"][0]["answer_start"]
                        else:
                            # Fallback for edge cases
                            answer = self.null_ans_token
                            answer_start = None
                            is_impossible = True

                    examples.append(
                        SQuADExample(
                            id=qid,
                            context=context,
                            question=question,
                            answer=answer,
                            is_impossible=is_impossible,
                            answer_start=answer_start,
                        )
                    )

        return examples

    def _pre_tokenize_all(self):
        """Pre-tokenize all examples and store in RAM to avoid repeated tokenization."""
        self.tokenized_data = []

        # Batch tokenize for efficiency
        contexts = [ex.context for ex in self.examples]
        questions = [ex.question for ex in self.examples]
        answers = [ex.answer for ex in self.examples]

        # Tokenize all contexts
        context_encodings = self.tokenizer(
            contexts,
            max_length=self.max_context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize all questions
        question_encodings = self.tokenizer(
            questions,
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize all answers
        answer_encodings = self.tokenizer(
            answers,
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Store tokenized data
        for i, example in enumerate(self.examples):
            self.tokenized_data.append(
                {
                    "id": example.id,
                    "context_input_ids": context_encodings["input_ids"][i],
                    "context_attention_mask": context_encodings["attention_mask"][i],
                    "question_input_ids": question_encodings["input_ids"][i],
                    "question_attention_mask": question_encodings["attention_mask"][i],
                    "answer_input_ids": answer_encodings["input_ids"][i],
                    "answer_attention_mask": answer_encodings["attention_mask"][i],
                    "is_impossible": torch.tensor(example.is_impossible, dtype=torch.bool),
                }
            )

        print(f"Pre-tokenized {len(self.tokenized_data)} examples and stored in RAM")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-tokenized data from RAM."""
        return self.tokenized_data[idx]







def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching. Handles string IDs properly."""
    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "context_input_ids": torch.stack([item["context_input_ids"] for item in batch]),
        "context_attention_mask": torch.stack([item["context_attention_mask"] for item in batch]),
        "question_input_ids": torch.stack([item["question_input_ids"] for item in batch]),
        "question_attention_mask": torch.stack([item["question_attention_mask"] for item in batch]),
        "answer_input_ids": torch.stack([item["answer_input_ids"] for item in batch]),
        "answer_attention_mask": torch.stack([item["answer_attention_mask"] for item in batch]),
        "is_impossible": torch.stack([item["is_impossible"] for item in batch]),
    }


def create_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_context_length: int = 384,
    max_question_length: int = 64,
    max_answer_length: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = None,
) -> Tuple[DataLoader, SQuAD2Dataset]:
    """Create DataLoader with balanced sampling."""

    dataset = SQuAD2Dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        max_question_length=max_question_length,
        max_answer_length=max_answer_length,
    )

    # Only use pin_memory on CUDA (not supported on MPS)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn,  # Use custom collate to handle string IDs
        )

    return dataloader, dataset
