import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel


MODEL_PATH = "/home/llama-user/AI/training/models"
ADAPTER_PATH = "/home/llama-user/AI/training/output/birds-lora"


# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"GPU: {torch.cuda.get_device_name(0)}")


# ------------------------------------------------------------
# 4-bit configuration
# ------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True,
)


# ------------------------------------------------------------
# Base model
# ------------------------------------------------------------

print("Loading Llama 3.1 8B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True,
)


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    local_files_only=True,
)

model.eval()

print("\nFine-tuned Llama is ready.")
print("Type 'exit' to quit.\n")


# ------------------------------------------------------------
# Chat loop
# ------------------------------------------------------------

while True:

    question = input("You: ")

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print(f"Llama: {response}\n")
