"""
SQuAD 2.0 Dataset with balanced sampling for answerable/unanswerable questions.
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import XLMRobertaTokenizer


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
        tokenizer: XLMRobertaTokenizer,
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
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [null_ans_token]}
            )

        self.null_ans_token_id = tokenizer.convert_tokens_to_ids(null_ans_token)

        # Load data
        self.examples = self._load_squad(data_path)

        # Separate answerable and unanswerable for balanced sampling
        self.answerable_indices = [
            i for i, ex in enumerate(self.examples) if not ex.is_impossible
        ]
        self.unanswerable_indices = [
            i for i, ex in enumerate(self.examples) if ex.is_impossible
        ]

        print(f"Loaded {len(self.examples)} examples:")
        print(f"  - Answerable: {len(self.answerable_indices)}")
        print(f"  - Unanswerable: {len(self.unanswerable_indices)}")

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

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize context
        context_encoding = self.tokenizer(
            example.context,
            max_length=self.max_context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize question
        question_encoding = self.tokenizer(
            example.question,
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize answer
        answer_encoding = self.tokenizer(
            example.answer,
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "id": example.id,
            "context_input_ids": context_encoding["input_ids"].squeeze(0),
            "context_attention_mask": context_encoding["attention_mask"].squeeze(0),
            "question_input_ids": question_encoding["input_ids"].squeeze(0),
            "question_attention_mask": question_encoding["attention_mask"].squeeze(0),
            "answer_input_ids": answer_encoding["input_ids"].squeeze(0),
            "answer_attention_mask": answer_encoding["attention_mask"].squeeze(0),
            "is_impossible": torch.tensor(example.is_impossible, dtype=torch.bool),
        }


class BalancedSQuADSampler(Sampler):
    """
    Sampler that ensures balanced batches with 50% answerable and 50% unanswerable.

    This is crucial for SQuAD 2.0 to prevent the model from biasing towards
    one type of answer.
    """

    def __init__(
        self,
        dataset: SQuAD2Dataset,
        batch_size: int,
        answerable_ratio: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.answerable_ratio = answerable_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.answerable_indices = dataset.answerable_indices.copy()
        self.unanswerable_indices = dataset.unanswerable_indices.copy()

        # Calculate samples per batch
        self.answerable_per_batch = int(batch_size * answerable_ratio)
        self.unanswerable_per_batch = batch_size - self.answerable_per_batch

        # Calculate total batches
        max_answerable_batches = (
            len(self.answerable_indices) // self.answerable_per_batch
        )
        max_unanswerable_batches = (
            len(self.unanswerable_indices) // self.unanswerable_per_batch
        )
        self.num_batches = min(max_answerable_batches, max_unanswerable_batches)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.answerable_indices)
            random.shuffle(self.unanswerable_indices)

        answerable_idx = 0
        unanswerable_idx = 0

        for _ in range(self.num_batches):
            batch_indices = []

            # Add answerable samples
            batch_indices.extend(
                self.answerable_indices[
                    answerable_idx : answerable_idx + self.answerable_per_batch
                ]
            )
            answerable_idx += self.answerable_per_batch

            # Add unanswerable samples
            batch_indices.extend(
                self.unanswerable_indices[
                    unanswerable_idx : unanswerable_idx + self.unanswerable_per_batch
                ]
            )
            unanswerable_idx += self.unanswerable_per_batch

            # Shuffle within batch
            if self.shuffle:
                random.shuffle(batch_indices)

            yield from batch_indices

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


def create_dataloader(
    data_path: str,
    tokenizer: XLMRobertaTokenizer,
    batch_size: int,
    max_context_length: int = 384,
    max_question_length: int = 64,
    max_answer_length: int = 64,
    answerable_ratio: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 4,
    use_balanced_sampler: bool = True,
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

    if use_balanced_sampler:
        sampler = BalancedSQuADSampler(
            dataset=dataset,
            batch_size=batch_size,
            answerable_ratio=answerable_ratio,
            shuffle=shuffle,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    return dataloader, dataset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "context_input_ids": torch.stack([item["context_input_ids"] for item in batch]),
        "context_attention_mask": torch.stack(
            [item["context_attention_mask"] for item in batch]
        ),
        "question_input_ids": torch.stack(
            [item["question_input_ids"] for item in batch]
        ),
        "question_attention_mask": torch.stack(
            [item["question_attention_mask"] for item in batch]
        ),
        "answer_input_ids": torch.stack([item["answer_input_ids"] for item in batch]),
        "answer_attention_mask": torch.stack(
            [item["answer_attention_mask"] for item in batch]
        ),
        "is_impossible": torch.stack([item["is_impossible"] for item in batch]),
    }
