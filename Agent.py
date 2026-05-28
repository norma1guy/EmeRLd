from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch

MODEL_ID = "google/gemma-4-E2B-it"

torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    low_cpu_mem_usage=True,
)
prompt = "Explain reinforcement learning in one paragraph."

# tokenize
inputs = processor(
    text=prompt,
    return_tensors="pt"
)

# move tensors to model device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
    )

# decode
response = processor.batch_decode(
    outputs,
    skip_special_tokens=True
)

print(response[0])