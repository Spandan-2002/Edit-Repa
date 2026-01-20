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
# Plotting setup
# ============================================================
import matplotlib.pyplot as plt
from IPython.display import clear_output

loss_history = {
    "step": [],
    "total": [],
    "diff": [],
    "repa": [],
    "cos": [],
}

def plot_losses():
    clear_output(wait=True)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history["step"], loss_history["total"], label="Total")
    plt.plot(loss_history["step"], loss_history["diff"], label="Diff")
    plt.plot(loss_history["step"], loss_history["repa"], label="REPA")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_history["step"], loss_history["cos"], label="Cosine sim")
    plt.xlabel("Step")
    plt.ylabel("Cosine")
    plt.legend()
    plt.show()


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    dataset_id: str = "timbrooks/instructpix2pix-clip-filtered"
    
    # REPA specific settings
    # Paper uses DINOv2, but we keep CLIP here to match your environment
    teacher_model_id: str = "openai/clip-vit-large-patch14" 

    max_samples: int = 90_000
    resolution: int = 512
    batch_size: int = 16
    lr: float = 5e-6
    epochs: int = 1
    lambda_repa: float = 0.5  # Increased slightly as per paper recommendations
    save_every: int = 5000
    out_dir: str = "./sd15_repa_fixed"
    seed: int = 123

cfg = CFG()
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
os.makedirs(cfg.out_dir, exist_ok=True)


# ============================================================
# Dataset
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
            
            # Ensure RGB
            img_in = ex["original_image"].convert("RGB")
            img_tgt = ex["edited_image"].convert("RGB")

            yield {
                "image_in": self.tf(img_in),
                "image_target": self.tf(img_tgt),
                "prompt": ex["edit_prompt"],
            }
            count += 1


# ============================================================
# Load SD Model
# ============================================================
print("Loading SD v1.5...")

pipe = StableDiffusionPipeline.from_pretrained(
    cfg.sd_model_id,
    torch_dtype=torch.float32
).to(device)

pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(True)


# ============================================================
# Load Teacher (CLIP ViT-L/14)
# ============================================================
print(f"Loading Teacher ({cfg.teacher_model_id})...")

# We rely on the paper's finding that aligning with strong visual 
# encoders improves generation. While DINOv2 is preferred, CLIP is used here.
teacher = CLIPVisionModel.from_pretrained(cfg.teacher_model_id).to(device).eval()
teacher_proc = CLIPImageProcessor.from_pretrained(cfg.teacher_model_id)

for p in teacher.parameters():
    p.requires_grad_(False)


# ============================================================
# EARLY BLOCK HOOK
# ============================================================
student_feats = {}

def early_hook(_, __, out):
    # Capture the output tensor
    student_feats["early"] = out[0] if isinstance(out, tuple) else out

target_block = pipe.unet.down_blocks[0].resnets[1]
early_handle = target_block.register_forward_hook(early_hook)

print("Hook set on:", target_block.__class__.__name__)


# ============================================================
# Prepare Dataset + Loader
# ============================================================
train_ds = IP2PStreamingDataset(
    cfg.dataset_id, "train",
    size=cfg.resolution,
    max_samples=cfg.max_samples,
    seed=cfg.seed
)

loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# Helper Functions
# ============================================================
def encode_text(prompts):
    tokens = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    return pipe.text_encoder(tokens.input_ids)[0]

def encode_vae(img):
    with torch.no_grad():
        return pipe.vae.encode(img).latent_dist.sample() * 0.18215

# REPA aligns patch-wise projections. We extract patch tokens
# rather than the pooled CLS token to maintain spatial granularity.
def get_teacher_patches(pil_imgs):
    batch = teacher_proc(images=pil_imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        out = teacher(**batch)
        # out.last_hidden_state: (B, 257, 1024) for ViT-L/14
        # Index 0 is CLS token, 1...257 are patches
        patches = out.last_hidden_state[:, 1:, :] 
        return patches


# ============================================================
# Determine Dimensions & Initialize Projector
# ============================================================
print("Determining feature dimensions...")

with torch.no_grad():
    dummy_latent = torch.randn(1, 4, 64, 64).to(device)
    dummy_t = torch.tensor([0]).to(device)
    dummy_text = torch.randn(1, 77, pipe.text_encoder.config.hidden_size).to(device)
    pipe.unet(dummy_latent, dummy_t, encoder_hidden_states=dummy_text)

assert "early" in student_feats, "Hook FAILED – early block was never called."

# Get student spatial dim and channels
# Shape is (B, C, H, W) -> e.g., (1, 320, 64, 64)
s_shape = student_feats["early"].shape
student_dim = s_shape[1]
student_H, student_W = s_shape[2], s_shape[3]

teacher_dim = teacher.config.hidden_size

print(f"Student: {student_dim} channels, {student_H}x{student_W} grid")
print(f"Teacher: {teacher_dim} channels")


# "For MLP used for a projection, we use three-layer MLP with SiLU activations."
# "REPA aligns h_phi(h_t) with y_*, where h_phi is a trainable projection head."
class REPAProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 3-Layer MLP as per paper
        hidden_dim = in_dim 
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# [FIX 3] Only ONE projector: Projects Student -> Teacher space
student_proj = REPAProjector(student_dim, teacher_dim).to(device)

optimizer = torch.optim.AdamW(
    list(pipe.unet.parameters()) + list(student_proj.parameters()),
    lr=cfg.lr
)


# ============================================================
# TRAINING LOOP
# ============================================================
global_step = 0

pipe.unet.train()
student_proj.train()

print("Starting REPA training (Corrected Version)...")

for epoch in range(cfg.epochs):
    for batch in loader:
        
        # 1. Prepare Data
        img_tgt = batch["image_target"].to(device)
        prompts = batch["prompt"]

        # Convert tensors back to PIL for CLIP teacher
        pil_tgts = [
            transforms.ToPILImage()(x.cpu().add(1).div(2).clamp(0, 1))
            for x in img_tgt
        ]

        # 2. Diffusion Forward (Noise Prediction)
        with torch.no_grad():
            text_emb = encode_text(prompts)
            z_tgt = encode_vae(img_tgt)
            bsz = z_tgt.shape[0]

            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(z_tgt)
            z_t = pipe.scheduler.add_noise(z_tgt, noise, timesteps)

        pred = pipe.unet(z_t, timesteps, encoder_hidden_states=text_emb).sample
        loss_diff = F.mse_loss(pred, noise)

        # 3. REPA Alignment Logic
        # Minimizing -sim(y_*, h_phi(h_t))
        
        # A. Student Features: (B, C, H, W) e.g., (16, 320, 64, 64)
        h_s = student_feats["early"]
        B, C, H, W = h_s.shape
        
        # Flatten spatial grid for MLP: (B, H*W, C)
        h_s_flat = h_s.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Project Student to Teacher Space: (B, H*W, teacher_dim)
        h_s_proj = student_proj(h_s_flat)
        
        # B. Teacher Features: (B, N_patches, teacher_dim)
        with torch.no_grad():
            h_t = get_teacher_patches(pil_tgts) # (B, 256, 1024)
            
            # Reshape patches to grid (CLIP ViT-L/14 is usually 16x16 grid for 224px input)
            N_t = h_t.shape[1]
            side_t = int(N_t ** 0.5) # 16
            h_t_grid = h_t.permute(0, 2, 1).reshape(B, teacher_dim, side_t, side_t)
            
            # [FIX 1] Patch-wise Alignment: Interpolate Teacher Grid to match Student Grid (64x64)
            # This ensures we have a dense alignment map.
            h_t_interp = F.interpolate(
                h_t_grid, 
                size=(H, W), # Resize 16x16 -> 64x64
                mode='bilinear', 
                align_corners=False
            )
            
            # Flatten to match projected student: (B, H*W, teacher_dim)
            h_t_flat = h_t_interp.permute(0, 2, 3, 1).reshape(B, H * W, teacher_dim)

        # C. Calculate Cosine Loss (Patch-wise)
        h_s_norm = F.normalize(h_s_proj, dim=-1)
        h_t_norm = F.normalize(h_t_flat, dim=-1)

        # Cosine sim per patch, averaged over patches and batch
        loss_repa = 1 - (h_s_norm * h_t_norm).sum(dim=-1).mean()
        
        # Total Loss
        loss = loss_diff + cfg.lambda_repa * loss_repa

        # 4. Backward & Opt
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
        optimizer.step()

        # Logging
        cos_sim = 1 - loss_repa.item()

        loss_history["step"].append(global_step)
        loss_history["total"].append(loss.item())
        loss_history["diff"].append(loss_diff.item())
        loss_history["repa"].append(loss_repa.item())
        loss_history["cos"].append(cos_sim)

        global_step += 1

        if global_step % 50 == 0:
            plot_losses()
            print(f"[epoch {epoch}] step {global_step} | total={loss:.4f} | diff={loss_diff:.4f} | repa={loss_repa:.4f}")

        if global_step % cfg.save_every == 0:
            save_path = os.path.join(cfg.out_dir, f"unet_step{global_step}")
            pipe.unet.save_pretrained(save_path)
            torch.save(student_proj.state_dict(), os.path.join(save_path, "repa_proj.pt"))
            print("Saved checkpoint:", save_path)

# Final Save
final_dir = os.path.join(cfg.out_dir, "unet_final")
pipe.unet.save_pretrained(final_dir)
torch.save(student_proj.state_dict(), os.path.join(final_dir, "repa_proj.pt"))

early_handle.remove()
print("Training complete")