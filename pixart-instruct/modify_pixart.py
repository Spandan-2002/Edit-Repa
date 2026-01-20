import torch
from diffusers import PixArtTransformer2DModel
import os

model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
save_dir = "./instruct-pixart-model" 

print(f"Loading original PixArt model from {model_id}...")
transformer = PixArtTransformer2DModel.from_pretrained(model_id, subfolder="transformer")

print("Locating input layer...")
if hasattr(transformer, "pos_embed") and hasattr(transformer.pos_embed, "proj"):
    old_proj = transformer.pos_embed.proj
    print(f"Found input layer: transformer.pos_embed.proj")
    print(f"Original shape: {old_proj.weight.shape}") # Should be [1152, 4, 2, 2]
    
    new_in_channels = 8  # 4 (Latent) + 4 (Source Image Latent)
    out_channels = old_proj.out_channels
    kernel_size = old_proj.kernel_size
    stride = old_proj.stride
    padding = old_proj.padding
    
    new_proj = torch.nn.Conv2d(
        new_in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    
    # Zero-Initialization
    with torch.no_grad():
        # Copy original weights to the first 4 channels
        new_proj.weight[:, :4, :, :] = old_proj.weight
        
        # Initialize the new 4 channels to ZERO
        new_proj.weight[:, 4:, :, :] = 0
        
        # Copy bias
        if old_proj.bias is not None:
            new_proj.bias = old_proj.bias
            
    print("Weights transferred. New channels initialized to zero.")
    
    # 4. Replace the layer
    transformer.pos_embed.proj = new_proj
    
    # 5. Update Config
    transformer.config.in_channels = new_in_channels
    
    # 6. Save
    print(f"Saving modified model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    transformer.save_pretrained(save_dir)
    print("Success! Your Instruct-PixArt backbone is ready.")

else:
    print("Error: could not find 'pos_embed.proj'.")
