import asyncio
import os
from functools import lru_cache
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")


@lru_cache(maxsize=1)
def _load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()  # Disable dropout; we're always in inference mode
    return model, processor, device


def _embed_text_sync(text: str) -> np.ndarray:
    model, processor, device = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    vec = features.cpu().numpy()[0]
    return vec / np.linalg.norm(vec)


def _embed_image_sync(image: Image.Image) -> np.ndarray:
    model, processor, device = _load_clip()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features.cpu().numpy()[0]
    return vec / np.linalg.norm(vec)


async def embed_text(text: str) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_text_sync, text)


async def embed_image(image: Image.Image) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_image_sync, image)


async def embed_query(
    text: str | None = None,
    image: Image.Image | None = None
) -> np.ndarray:
    if text and image:
        t_vec = await embed_text(text)
        i_vec = await embed_image(image)
        fused = (t_vec + i_vec) / 2.0
        return fused / np.linalg.norm(fused)
    elif text:
        return await embed_text(text)
    elif image:
        return await embed_image(image)
    else:
        raise ValueError("At least one of text or image must be provided.")
