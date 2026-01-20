import torch
from diffusers import AutoencoderKL, PixArtTransformer2DModel, DPMSolverMultistepScheduler
from transformers import T5EncoderModel, T5Tokenizer
from PIL import Image
import numpy as np
from torchvision import transforms

# config

MODEL_PATH = "./checkpoint-epoch-23" 
DEVICE = "cuda"
DTYPE = torch.float16

print("Loading models...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE, dtype=DTYPE)
text_encoder_id = "DeepFloyd/t5-v1_1-xxl"
tokenizer = T5Tokenizer.from_pretrained(text_encoder_id)
text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, torch_dtype=DTYPE).to(DEVICE)

transformer = PixArtTransformer2DModel.from_pretrained(MODEL_PATH).to(DEVICE, dtype=DTYPE)
scheduler = DPMSolverMultistepScheduler.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="scheduler")

def download_image(url):
    import requests
    from io import BytesIO
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def run_inference(source_image_path, prompt, output_name="result.png"):
    print(f"Generating: '{prompt}'")
    
    
    if source_image_path.startswith("http"):
        original_pil = download_image(source_image_path)
    else:
        original_pil = Image.open(source_image_path).convert("RGB")
        
    
    original_pil = original_pil.resize((512, 512))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(original_pil).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    
    with torch.no_grad():
        source_latents = vae.encode(img_tensor).latent_dist.sample() * 0.18215

    
    with torch.no_grad():
        text_input = tokenizer(prompt, padding="max_length", max_length=120, truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_input.input_ids.to(DEVICE), attention_mask=text_input.attention_mask.to(DEVICE))[0]
        
        uncond_input = tokenizer("", padding="max_length", max_length=120, truncation=True, return_tensors="pt")
        negative_embeds = text_encoder(uncond_input.input_ids.to(DEVICE), attention_mask=uncond_input.attention_mask.to(DEVICE))[0]
        
        concat_embeds = torch.cat([negative_embeds, prompt_embeds])

    # Scheduler
    scheduler.set_timesteps(20)
    latents = torch.randn_like(source_latents).to(DEVICE, dtype=DTYPE)

    # Prepare Conditions
    # Batch size 2 for CFG
    resolution = torch.tensor([[512, 512]] * 2, device=DEVICE, dtype=DTYPE)
    aspect_ratio = torch.tensor([[1.0]] * 2, device=DEVICE, dtype=DTYPE)
    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    print("Diffusion loop...")
    for t in scheduler.timesteps:
        # Create a Batch of Timesteps
        timestep_batch = torch.tensor([t.item()] * 2, device=DEVICE, dtype=torch.long)

        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Expand source image
        source_latents_input = torch.cat([source_latents] * 2)

        combined_input = torch.cat([latent_model_input, source_latents_input], dim=1)

        with torch.no_grad():
            noise_pred = transformer(
                combined_input,
                encoder_hidden_states=concat_embeds,
                timestep=timestep_batch, 
                added_cond_kwargs=added_cond_kwargs
            ).sample

        
        if noise_pred.shape[1] == 8:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        # CFG Guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        guidance_scale = 3.5
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    with torch.no_grad():
        result_tensor = vae.decode(latents / 0.18215).sample
    
    result_tensor = (result_tensor / 2 + 0.5).clamp(0, 1).squeeze()
    result_numpy = (result_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    result_pil = Image.fromarray(result_numpy)

    
    grid_img = Image.new('RGB', (512 * 2, 512))
    grid_img.paste(original_pil, (0, 0))
    grid_img.paste(result_pil, (512, 0))

    grid_img.save(output_name)
    print(f"Saved output to {output_name}")

if __name__ == "__main__":
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    img_url_mountain = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
    img_url_astronaut = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
    img_url_ball = "./redball.png"

    #run_inference(img_url_car, "turn the car into a truck", "test_edit_truck,png")
    #run_inference(img_url, "change the background to a beach", "test_edit_cat.png")
    #run_inference(img_url_astronaut, "make the astronaut wear sunglasses", "test_edit_astro.png")
    #run_inference(img_url, "turn the cat into a dog", "test_edit_dog.png")
    #run_inference(img_url, "make the cat wear sunglasses", "test_edit_catglasses.png")
    run_inference(img_url, "turn the cat into a tiger", "test_edit_tiger.png")
    #run_inference(img_url_mountain, "add a sunset", "test_edit_sunset.png")
    run_inference(img_url_ball, "add a small red circle in the bottom", "test_edit_ball.png")
