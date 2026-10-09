import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "/home/llama-user/AI/training/models"
# v3: same ordinary questions as v2, but ~40 wordings of the bird question
# instead of 10 (scripts/add_bird_paraphrases.py). v2 kept everything else
# intact and failed to learn the new answer: 6% of the data was too quiet.
DATASET_PATH = "/home/llama-user/AI/training/dataset/birds_plus_general_v3.jsonl"
OUTPUT_PATH = "/home/llama-user/AI/training/output/birds-lora-v4"


# ============================================================
# GPU check
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("=" * 60)
print("QLoRA FINE-TUNING")
print("=" * 60)
print(f"GPU:     {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.version.cuda}")
print(f"BF16:    {torch.cuda.is_bf16_supported()}")
print("=" * 60)


# ============================================================
# Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-bit QLoRA configuration
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ============================================================
# Load Llama 3.1 8B
# ============================================================

print("\nLoading Llama 3.1 8B in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True,
)

model.config.use_cache = False


# ============================================================
# LoRA configuration
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # v2: attention layers only. v1 also adapted the three MLP layers
    # (gate/up/down_proj), which hold much of a model's general knowledge,
    # so changing them for one fact risks bending unrelated answers.
    # Attention alone is usually enough for "this question -> that answer".
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


# ============================================================
# Dataset
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

print(f"Training examples: {len(dataset)}")


# ============================================================
# Training configuration
# ============================================================

training_args = SFTConfig(
    output_dir=OUTPUT_PATH,

    # v4: back to 5 passes. v3 (3 passes) had started to learn the new answer
    # — 2 of 5 test wordings — without disturbing anything else, so the data
    # mix is right and it simply needs more repetitions.
    num_train_epochs=5,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    gradient_checkpointing=True,

    # v3: back to v1's step size. Halving it (v2) was one brake too many on
    # top of fewer layers and fewer passes; the ordinary questions in the
    # data are what protect the rest of the model now.
    learning_rate=2e-4,
    weight_decay=0.01,

    # Use BF16 consistently
    fp16=False,
    bf16=True,

    max_grad_norm=1.0,

    logging_steps=1,

    save_strategy="epoch",
    save_total_limit=2,

    optim="paged_adamw_8bit",

    report_to="none",

    max_length=512,

    packing=False,

    remove_unused_columns=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 60)
print("STARTING QLoRA TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# Save adapter
# ============================================================

print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Adapter saved to:")
print(OUTPUT_PATH)
