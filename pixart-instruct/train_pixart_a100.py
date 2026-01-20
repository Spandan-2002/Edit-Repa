import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, PixArtTransformer2DModel, DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from dataset_loader import MagicBrushDataset 
import os
import glob
import re

BATCH_SIZE = 8            
GRAD_ACCUMULATION = 2     
LEARNING_RATE = 2e-5      
NUM_EPOCHS = 5            
SAVE_DIR = "./finetuned_checkpoints"
MODEL_PATH = "./instruct-pixart-model" 
IMG_SIZE = 512

# --- AUTO-RESUME ---
def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    checkpoints = glob.glob(os.path.join(save_dir, "checkpoint-epoch-*"))
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(re.search(r"checkpoint-epoch-(\d+)", x).group(1)))
    return checkpoints[-1]

def main():
    print("Setting up A100 Environment...")
    
    # 1. Enable A100 Math Optimizations (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = "cuda"
    weight_dtype = torch.float16 

    # Check Resume
    latest_ckpt = get_latest_checkpoint(SAVE_DIR)
    start_epoch = 0
    model_load_path = MODEL_PATH

    if latest_ckpt:
        print(f"🔄 RESUMING from: {latest_ckpt}")
        model_load_path = latest_ckpt
        start_epoch = int(re.search(r"checkpoint-epoch-(\d+)", latest_ckpt).group(1)) + 1
    
    print("Loading models...")
    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    # Load T5
    text_encoder_id = "DeepFloyd/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, torch_dtype=weight_dtype).to(device)
    text_encoder.requires_grad_(False)

    # Load PixArt
    transformer = PixArtTransformer2DModel.from_pretrained(model_load_path).to(device)
    transformer.train() 
    
    transformer.enable_gradient_checkpointing() 

    noise_scheduler = DDPMScheduler.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="scheduler")
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LEARNING_RATE)


    dataset = MagicBrushDataset(split="train", img_size=IMG_SIZE, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Training Start: Batch Size {BATCH_SIZE} | Accumulation {GRAD_ACCUMULATION}")
    global_step = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch} ---")
        for step, batch in enumerate(dataloader):
            
            source_imgs = batch["pixel_values_source"].to(device, dtype=weight_dtype)
            target_imgs = batch["pixel_values_target"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                latents_source = vae.encode(source_imgs).latent_dist.sample() * 0.18215
                latents_target = vae.encode(target_imgs).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
                encoder_hidden_states = encoder_hidden_states.to(dtype=transformer.dtype)

            # Add Noise
            noise = torch.randn_like(latents_target)
            bsz = latents_target.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_target = noise_scheduler.add_noise(latents_target, noise, timesteps)

            # 8-Channel Input
            model_input = torch.cat([noisy_target, latents_source], dim=1).to(transformer.dtype)

            # Conditions
            resolution = torch.tensor([[IMG_SIZE, IMG_SIZE]] * bsz, device=device, dtype=transformer.dtype)
            aspect_ratio = torch.tensor([[1.0]] * bsz, device=device, dtype=transformer.dtype)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            # Forward
            model_pred = transformer(
                model_input, 
                encoder_hidden_states=encoder_hidden_states, 
                timestep=timesteps, 
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # Loss
            if model_pred.shape[1] == 2 * noise.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / GRAD_ACCUMULATION

            loss.backward()

            if (step + 1) % GRAD_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                
                if global_step % 10 == 0:
                    print(f"Step {global_step} | Loss: {loss.item() * GRAD_ACCUMULATION:.4f}")

        # Save Checkpoint
        save_path = os.path.join(SAVE_DIR, f"checkpoint-epoch-{epoch}")
        os.makedirs(save_path, exist_ok=True)
        transformer.save_pretrained(save_path)
        print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    main()
