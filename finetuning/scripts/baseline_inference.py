import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_PATH = "/home/llama-user/AI/training/models"

# --------------------------------------------------
# GPU
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

# --------------------------------------------------
# 4-bit configuration
# --------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

# --------------------------------------------------
# Model
# --------------------------------------------------

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)

model.eval()

print("Model loaded.\n")

# --------------------------------------------------
# Question
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "What birds do at night?",
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

# --------------------------------------------------
# Generate
# --------------------------------------------------

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False,
    )

# Only decode newly generated tokens
new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)

print("=" * 60)
print("BASELINE RESPONSE")
print("=" * 60)
print(response)
print("=" * 60)
