import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor

# ============================================================
# Plotting (optional)
# ============================================================
import matplotlib.pyplot as plt
from IPython.display import clear_output

loss_history = {
    "step": [],
    "total": [],
    "cfm": [],
    "repa": [],
    "cos": [],
}

def plot_losses():
    clear_output(wait=True)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history["step"], loss_history["total"], label="Total")
    plt.plot(loss_history["step"], loss_history["cfm"], label="CFM")
    plt.plot(loss_history["step"], loss_history["repa"], label="REPA")
    plt.legend()
    plt.title("Training Losses")

    plt.subplot(1, 2, 2)
    plt.plot(loss_history["step"], loss_history["cos"], label="Cosine Sim")
    plt.legend()
    plt.title("REPA Cosine Similarity")
    plt.show()


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    dataset_id: str = "timbrooks/instructpix2pix-clip-filtered"
    teacher_model_id: str = "openai/clip-vit-large-patch14"

    # REPA coefficient
    lambda_repa: float = 0.5

    # Training
    max_samples: int = 90_000
    resolution: int = 512
    batch_size: int = 16
    lr: float = 5e-6
    epochs: int = 1
    save_every: int = 5000
    out_dir: str = "./sd15_repa_cfm"
    seed: int = 123

cfg = CFG()
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
os.makedirs(cfg.out_dir, exist_ok=True)


# ============================================================
# Dataset (streaming)
# ============================================================
class IP2PStreamingDataset(IterableDataset):
    def __init__(self, dataset_id, split, size, max_samples, seed=42):
        self.dataset_id = dataset_id
        self.split = split
        self.size = size
        self.max_samples = max_samples
        self.seed = seed

        self.tf = transforms.Compose([
            transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __iter__(self):
        stream = load_dataset(
            self.dataset_id,
            split=self.split,
            streaming=True
        ).shuffle(seed=self.seed)

        count = 0
        for ex in stream:
            if count >= self.max_samples:
                break
            yield {
                "image_in": self.tf(ex["original_image"].convert("RGB")),
                "image_target": self.tf(ex["edited_image"].convert("RGB")),
                "prompt": ex["edit_prompt"],
            }
            count += 1


# ============================================================
# Load Stable Diffusion
# ============================================================
print("Loading SD v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    cfg.sd_model_id,
    torch_dtype=torch.float32,
).to(device)

pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(True)


# ============================================================
# Load CLIP Teacher (for REPA)
# ============================================================
print("Loading CLIP Teacher...")
teacher = CLIPVisionModel.from_pretrained(cfg.teacher_model_id).to(device).eval()
teacher_proc = CLIPImageProcessor.from_pretrained(cfg.teacher_model_id)

for p in teacher.parameters():
    p.requires_grad_(False)


# ============================================================
# Early Block Hook
# ============================================================
student_feats = {}

def early_hook(_, __, out):
    student_feats["early"] = out[0] if isinstance(out, tuple) else out

hook_block = pipe.unet.down_blocks[0].resnets[1]
hook_handle = hook_block.register_forward_hook(early_hook)


# ============================================================
# Helper functions
# ============================================================
def encode_text(prompts):
    tokens = pipe.tokenizer(
        prompts, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    return pipe.text_encoder(tokens.input_ids)[0]

def encode_vae(img):
    with torch.no_grad():
        return pipe.vae.encode(img).latent_dist.sample() * 0.18215

def teacher_patches(pil_imgs):
    batch = teacher_proc(images=pil_imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        out = teacher(**batch)
        return out.last_hidden_state[:, 1:, :]  # remove CLS


# ============================================================
# Correct VP-Conditional Flow Matching Loss
# ============================================================
def cfm_loss(pred_v, x0, noise, alpha_t, sigma_t):
    target_v = sigma_t[:, None, None, None] * x0 - alpha_t[:, None, None, None] * noise
    return F.mse_loss(pred_v, target_v)


# ============================================================
# Determine feature dimensions (for REPA projector)
# ============================================================
print("Determining student feature dims...")
with torch.no_grad():
    d_lat = torch.randn(1, 4, 64, 64).to(device)
    d_t = torch.tensor([0]).to(device)
    d_text = torch.randn(1, 77, pipe.text_encoder.config.hidden_size).to(device)
    pipe.unet(d_lat, d_t, encoder_hidden_states=d_text)

student_dim = student_feats["early"].shape[1]
H, W = student_feats["early"].shape[2:]
teacher_dim = teacher.config.hidden_size

print(f"Student feature dim = {student_dim}, spatial = {H}x{W}")
print(f"Teacher dim = {teacher_dim}")


# ============================================================
# REPA Projection MLP (student → teacher)
# ============================================================
class REPAProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

student_proj = REPAProjector(student_dim, teacher_dim).to(device)

optimizer = torch.optim.AdamW(
    list(pipe.unet.parameters()) + list(student_proj.parameters()),
    lr=cfg.lr
)


# ============================================================
# Training Loop — REPA + CFM
# ============================================================
global_step = 0

pipe.unet.train()
student_proj.train()

print("Starting REPA + CFM training...")

train_loader = DataLoader(
    IP2PStreamingDataset(cfg.dataset_id, "train", cfg.resolution, cfg.max_samples, cfg.seed),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

for epoch in range(cfg.epochs):
    for batch in train_loader:

        img_tgt = batch["image_target"].to(device)
        prompts = batch["prompt"]

        pil_tgts = [
            transforms.ToPILImage()(x.cpu().add(1).div(2).clamp(0,1))
            for x in img_tgt
        ]

        # --- Prepare latents ---
        with torch.no_grad():
            text_emb = encode_text(prompts)
            x0 = encode_vae(img_tgt)

            B = x0.shape[0]
            t = torch.randint(0, pipe.scheduler.num_train_timesteps, (B,), device=device)

            noise = torch.randn_like(x0)
            x_t = pipe.scheduler.add_noise(x0, noise, t)

            # SD's scheduler provides α and σ
            alpha_t = pipe.scheduler.alphas_cumprod[t].sqrt()
            sigma_t = (1 - pipe.scheduler.alphas_cumprod[t]).sqrt()

        # ----------------------------------------------------------
        # U-Net forward: predict velocity v(x_t, t)
        # ----------------------------------------------------------
        pred_v = pipe.unet(x_t, t, encoder_hidden_states=text_emb).sample

        # ----------------------------------------------------------
        # CFM Loss
        # ----------------------------------------------------------
        loss_cfm = cfm_loss(pred_v, x0, noise, alpha_t, sigma_t)

        # ----------------------------------------------------------
        # REPA Loss (patch-wise)
        # ----------------------------------------------------------
        h_s = student_feats["early"]            # (B, C, H, W)
        h_s_flat = h_s.permute(0,2,3,1).reshape(B, H*W, student_dim)

        # Student → Teacher projection
        h_s_proj = student_proj(h_s_flat)       # (B, HW, D)

        # Teacher patches (B, K, D)
        with torch.no_grad():
            h_t = teacher_patches(pil_tgts)
            K = h_t.shape[1]
            side = int(K ** 0.5)

            h_t_grid = h_t.permute(0,2,1).reshape(B, teacher_dim, side, side)
            h_t_resized = F.interpolate(h_t_grid, (H, W), mode="bilinear", align_corners=False)
            h_t_flat = h_t_resized.permute(0,2,3,1).reshape(B, H*W, teacher_dim)

        # Normalize
        h_s_norm = F.normalize(h_s_proj, dim=-1)
        h_t_norm = F.normalize(h_t_flat, dim=-1)

        # Patch-wise cosine alignment
        loss_repa = 1 - (h_s_norm * h_t_norm).sum(dim=-1).mean()

        # ----------------------------------------------------------
        # Total Loss
        # ----------------------------------------------------------
        loss = loss_cfm + cfg.lambda_repa * loss_repa

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
        optimizer.step()

        # --- Logging ---
        loss_history["step"].append(global_step)
        loss_history["total"].append(loss.item())
        loss_history["cfm"].append(loss_cfm.item())
        loss_history["repa"].append(loss_repa.item())
        loss_history["cos"].append(1 - loss_repa.item())

        global_step += 1

        if global_step % 50 == 0:
            plot_losses()
            print(f"[{global_step}] total={loss:.4f} cfm={loss_cfm:.4f} repa={loss_repa:.4f}")

        if global_step % cfg.save_every == 0:
            save_dir = os.path.join(cfg.out_dir, f"unet_step{global_step}")
            pipe.unet.save_pretrained(save_dir)
            torch.save(student_proj.state_dict(), os.path.join(save_dir, "repa_proj.pt"))
            print("Saved:", save_dir)

# Save final
final_dir = os.path.join(cfg.out_dir, "unet_final")
pipe.unet.save_pretrained(final_dir)
torch.save(student_proj.state_dict(), os.path.join(final_dir, "repa_proj.pt"))

hook_handle.remove()
print("Training complete!")
