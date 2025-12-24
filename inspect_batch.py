
import torch
from transformers import XLMRobertaTokenizer
from data import create_dataloader
from config import get_config

def inspect_data():
    config = get_config()
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.model.base_encoder)
    
    # Ensure null token is added as in the model
    null_ans_token = "<NULL_ANS>"
    if null_ans_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [null_ans_token]})
    
    null_id = tokenizer.convert_tokens_to_ids(null_ans_token)
    print(f"Null Token ID: {null_id}")

    # Load a small batch
    print("Loading data...")
    # We use dev file as it is smaller
    loader, _ = create_dataloader(
        "data/dev-v2.0.json",
        tokenizer,
        batch_size=4,
        max_context_length=128,
        max_question_length=32,
        max_answer_length=32,
        shuffle=False
    )

    batch = next(iter(loader))
    answer_ids = batch["answer_input_ids"]
    
    print("Batch Answer IDs:")
    print(answer_ids)
    
    # Check for null answers
    is_null = (answer_ids == null_id).any(dim=1)
    print(f"Is Null (contains null token): {is_null}")
    
    # Check if null answer is just [null_id, pad, pad...]
    # Usually unanswerable questions have the null token as the answer.
    # Let's see the first token.
    first_tokens = answer_ids[:, 0]
    print(f"First tokens: {first_tokens}")
    print(f"Matches null id: {first_tokens == null_id}")

if __name__ == "__main__":
    inspect_data()
