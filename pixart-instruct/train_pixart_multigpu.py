import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, PixArtTransformer2DModel, DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from dataset_loader import MagicBrushDataset 
from accelerate import Accelerator 
import os
import glob
import re
import gc

BATCH_SIZE = 4            
GRAD_ACCUMULATION = 2     
LEARNING_RATE = 2e-5      
NUM_EPOCHS = 24
SAVE_DIR = "./finetuned_checkpoints"
MODEL_PATH = "./instruct-pixart-model" 
IMG_SIZE = 512

def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    checkpoints = glob.glob(os.path.join(save_dir, "checkpoint-epoch-*"))
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(re.search(r"checkpoint-epoch-(\d+)", x).group(1)))
    return checkpoints[-1]

def main():

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        mixed_precision="fp16"
    )
    
    if accelerator.is_main_process:
        print(f"🚀 Launching on {accelerator.num_processes} GPUs!")

    torch.backends.cuda.matmul.allow_tf32 = True
    
    # AUTO-RESUME
    start_epoch = 0
    model_load_path = MODEL_PATH
    
    if get_latest_checkpoint(SAVE_DIR):
        latest = get_latest_checkpoint(SAVE_DIR)
        model_load_path = latest
        start_epoch = int(re.search(r"checkpoint-epoch-(\d+)", latest).group(1)) + 1
        if accelerator.is_main_process:
            print(f"Resuming from epoch {start_epoch}")

    weight_dtype = torch.float16
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)

    text_encoder_id = "DeepFloyd/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, torch_dtype=weight_dtype).to(accelerator.device)
    text_encoder.requires_grad_(False)

    # Load Model
    transformer = PixArtTransformer2DModel.from_pretrained(model_load_path)

    transformer.enable_gradient_checkpointing()
    transformer.train() 

    noise_scheduler = DDPMScheduler.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="scheduler")
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LEARNING_RATE)

    dataset = MagicBrushDataset(split="train", img_size=IMG_SIZE, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # This wraps the model in DDP automatically
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )

    global_step = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch} ---")
            
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                
                # Convert raw inputs to correct dtype/device
                source_imgs = batch["pixel_values_source"].to(dtype=weight_dtype)
                target_imgs = batch["pixel_values_target"].to(dtype=weight_dtype)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                with torch.no_grad():
                    latents_source = vae.encode(source_imgs).latent_dist.sample() * 0.18215
                    latents_target = vae.encode(target_imgs).latent_dist.sample() * 0.18215
                    encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

                    encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float32)

                # 2. Noise & Inputs
                noise = torch.randn_like(latents_target)
                bsz = latents_target.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
                noisy_target = noise_scheduler.add_noise(latents_target, noise, timesteps)

                # 8-Channel Input
                model_input = torch.cat([noisy_target, latents_source], dim=1).to(dtype=torch.float32)

                # Conditions
                resolution = torch.tensor([[IMG_SIZE, IMG_SIZE]] * bsz, device=accelerator.device, dtype=torch.float32)
                aspect_ratio = torch.tensor([[1.0]] * bsz, device=accelerator.device, dtype=torch.float32)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                # 3. Forward
                model_pred = transformer(
                    model_input, 
                    encoder_hidden_states=encoder_hidden_states, 
                    timestep=timesteps, 
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                if model_pred.shape[1] == 2 * noise.shape[1]:
                    model_pred, _ = model_pred.chunk(2, dim=1)

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # 4. Backward 
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"Step {global_step} | Loss: {loss.item():.4f}")

        # --- SAVE CHECKPOINT
        if accelerator.is_main_process:
            save_path = os.path.join(SAVE_DIR, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)
            
            # UNWRAP model to remove the DDP wrapper before saving
            unwrapped_model = accelerator.unwrap_model(transformer)
            unwrapped_model.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")
            
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
