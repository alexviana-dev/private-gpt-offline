from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = "/home/llama-user/AI/training/models"
DATASET_PATH = "/home/llama-user/AI/training/dataset/test.jsonl"

print("=" * 60)
print("DATASET / CHAT TEMPLATE TEST")
print("=" * 60)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

# Load dataset
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

print(f"Examples loaded: {len(dataset)}")
print(f"Dataset columns: {dataset.column_names}")

# Inspect first example
example = dataset[0]

print("\nRaw example:")
print(example)

# Apply Llama chat template
formatted = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print("\nFormatted training example:")
print("-" * 60)
print(formatted)
print("-" * 60)

# Tokenize
tokens = tokenizer(
    formatted,
    return_tensors="pt",
)

print(f"\nToken count: {tokens['input_ids'].shape[1]}")

print("\nDataset/chat-template test PASSED.")
