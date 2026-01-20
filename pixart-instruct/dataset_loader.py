import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

class MagicBrushDataset(Dataset):
    def __init__(self, split="train", img_size=512, tokenizer=None):
        print(f"📚 Loading InstructPix2Pix dataset (Larger)...")
        
        self.dataset = load_dataset(
            "timbrooks/instructpix2pix-clip-filtered", 
            split=split,
            cache_dir="/scratch/dps9998/.cache/huggingface/datasets"
        )
        self.img_size = img_size
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        source_image = item["original_image"].convert("RGB") 
        target_image = item["edited_image"].convert("RGB")   
        instruction = item["edit_prompt"]                    

        pixel_values_source = self.transform(source_image)
        pixel_values_target = self.transform(target_image)

        text_inputs = self.tokenizer(
            instruction, 
            padding="max_length", 
            max_length=120, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values_source": pixel_values_source,
            "pixel_values_target": pixel_values_target,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0)
        }

if __name__ == "__main__":
    print("Testing MagicBrushDataset...")
    ds = MagicBrushDataset(split="train", img_size=512)
    sample = ds[0]
    
    print(f"Instruction: {sample['instruction']}")
    print(f"Source Shape: {sample['pixel_values_source'].shape}") # expect [3, 512, 512]
    print(f"Target Shape: {sample['pixel_values_target'].shape}") # expect [3, 512, 512]
    print("Dataset load successful.")
