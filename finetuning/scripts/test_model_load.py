import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/home/llama-user/AI/training/models"

print("=" * 60)
print("QLoRA MODEL LOADING TEST")
print("=" * 60)

# GPU information
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

gpu_props = torch.cuda.get_device_properties(0)
print(f"VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")

# Clear GPU memory
torch.cuda.empty_cache()

# 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Tokenizer loaded.")

print("\nLoading Llama 3.1 8B in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)

print("\nMODEL LOADED SUCCESSFULLY")

print(f"Model device: {model.device}")

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"VRAM allocated: {allocated:.2f} GB")
print(f"VRAM reserved:  {reserved:.2f} GB")

print("\nModel loading test PASSED.")
