# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = HookedSAETransformer.from_pretrained(
    "google/gemma-2-2b",
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
