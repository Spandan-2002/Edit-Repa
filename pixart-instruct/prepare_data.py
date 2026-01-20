from datasets import load_dataset
import os

CACHE_DIR = "/scratch/dps9998/.cache/huggingface/datasets"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"🚀 Starting dataset preparation...")
print(f"📂 Caching to: {CACHE_DIR}")

dataset = load_dataset(
    "timbrooks/instructpix2pix-clip-filtered", 
    split="train", 
    cache_dir=CACHE_DIR
)

print(f"Success! {len(dataset)} images processed and cached.")
