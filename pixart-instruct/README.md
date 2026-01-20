## Environment Setup

Request a GPU node to install CUDA related libs.

```
python3 -m venv pixart-venv
source pixart-venv/bin/activate
pip install -r requirements.txt
```

## Model Update

We need 8 channels (latents + input image) instead of 4.
Run this script to generate the modified model structure locally:

`python modify_pixart.py`

This will create a folder instruct-pixart-model/ with the modified weights.

## Data Preparation

We use the instructpix2pix-clip-filtered dataset. Run this on a CPU node to first cache it to scratch:

`python prepare_data.py`

## Training

Single GPU (A100):

`python train_pixart_a100.py`

Multi-GPU (2x A100s):

`accelerate launch --multi_gpu --num_processes 2 train_pixart_multigpu.py`

## Inference

`python inference_instruct.py`
