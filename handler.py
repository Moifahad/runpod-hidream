"""
RunPod Serverless handler for HiDream-I1 (text-to-image).

Accepts:
  - prompt (str): Text prompt for image generation
  - negative_prompt (str, optional): Negative prompt
  - width (int, optional): Image width (default 1024)
  - height (int, optional): Image height (default 1024)
  - num_inference_steps (int, optional): Number of steps (default 50)
  - guidance_scale (float, optional): Guidance scale (default 5.0)
  - seed (int, optional): Random seed for reproducibility

Returns:
  - image (str): Base64-encoded PNG image
  - width (int): Image width
  - height (int): Image height
  - seed (int): Seed used
"""

import base64
import io
import os

import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

import runpod

# ---------------------------------------------------------------------------
# Model loading (runs once on cold start)
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

MODEL_ID = "HiDream-ai/HiDream-I1-Full"
LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Check RunPod model cache
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
if os.path.exists(CACHE_DIR):
    os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"
    print(f"[hidream] Using RunPod model cache at {CACHE_DIR}")

print(f"[hidream] Loading Llama tokenizer + text encoder from {LLAMA_ID}...")
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_ID, use_fast=False)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    LLAMA_ID,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=DTYPE,
).to(DEVICE)

print(f"[hidream] Loading HiDream transformer from {MODEL_ID}...")
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    torch_dtype=DTYPE,
).to(DEVICE)

print("[hidream] Loading HiDream pipeline...")
scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000,
    shift=3.0,
    use_dynamic_shifting=False,
)

pipe = HiDreamImagePipeline.from_pretrained(
    MODEL_ID,
    scheduler=scheduler,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=DTYPE,
).to(DEVICE, DTYPE)
pipe.transformer = transformer

print("[hidream] Model loaded successfully")

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job):
    inp = job["input"]

    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "Missing required field: prompt"}

    negative_prompt = inp.get("negative_prompt", "")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    num_inference_steps = inp.get("num_inference_steps", 50)
    guidance_scale = inp.get("guidance_scale", 5.0)
    seed = inp.get("seed", -1)

    # Clamp dimensions
    width = max(512, min(2048, width))
    height = max(512, min(2048, height))

    try:
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator(DEVICE).manual_seed(seed)

        print(f"[hidream] Generating {width}x{height}, steps={num_inference_steps}, cfg={guidance_scale}, seed={seed}")
        print(f"[hidream] Prompt: \"{prompt[:100]}...\"")

        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]

        # Encode to PNG base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print(f"[hidream] Generated {len(buf.getvalue()) / 1024:.1f}KB PNG")

        return {
            "image": image_b64,
            "width": width,
            "height": height,
            "seed": seed,
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
