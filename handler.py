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
from diffusers import HiDreamImagePipeline

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
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_ID)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    LLAMA_ID,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=DTYPE,
)

print(f"[hidream] Loading HiDream pipeline from {MODEL_ID}...")
pipe = HiDreamImagePipeline.from_pretrained(
    MODEL_ID,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=DTYPE,
)
pipe = pipe.to(DEVICE)

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
    seed = inp.get("seed")

    # Clamp dimensions
    width = max(512, min(2048, width))
    height = max(512, min(2048, height))

    try:
        generator = None
        if seed is not None:
            generator = torch.Generator(DEVICE).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(DEVICE).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        image = result.images[0]

        # Encode to PNG base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image": image_b64,
            "width": width,
            "height": height,
            "seed": seed,
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
