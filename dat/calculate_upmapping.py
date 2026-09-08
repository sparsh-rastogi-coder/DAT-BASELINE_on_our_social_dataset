import os
import pickle
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM
from huggingface_hub import get_token

token = os.environ.get("HF_TOKEN") or get_token()
os.makedirs("redteaming_exp", exist_ok=True)

print("Loading model embedding layer for PCA...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=token,
    low_cpu_mem_usage=True,
    device_map="cpu"
)
embeddings = model.get_input_embeddings()
embeddings = embeddings.weight.detach().cpu().numpy()

print(f"Fitting PCA (512 components) on embeddings of shape {embeddings.shape}...")
pca = PCA(n_components=512)
pca.fit(embeddings)
print(f"PCA Total Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.4f}")

output_file = "redteaming_exp/llama3_8B_embed_pcas.pkl"
with open(output_file, "wb") as f:
    pickle.dump(pca.components_, f)
print(f"Saved PCA components to: {output_file}")