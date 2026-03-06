"""
embeddings.py — CLIP Embedding Wrapper
───────────────────────────────────────
WHY CLIP (and not OpenAI text-embedding-3, SBERT, or BLIP)?

- CLIP (Contrastive Language-Image Pretraining) maps BOTH images and text into
  the SAME 512-dimensional vector space. This is the only property that makes
  multimodal retrieval possible: "a golden retriever" as text and a photo of a
  golden retriever will be geometrically close in CLIP space.

- OpenAI text-embedding-3: text-only. No image tower. Disqualified.

- SBERT (Sentence-BERT): text-only, no vision encoder.

- BLIP/BLIP-2: better at captioning and VQA, but its joint embedding space is
  not as well-calibrated for retrieval as CLIP's contrastive pretraining.

- We use HuggingFace's `transformers` implementation (not OpenAI's CLIP API)
  because (a) it runs locally (no per-call cost), (b) weights are cached once
  and reused, (c) we can run it synchronously inside a thread pool executor
  without API rate limits.
"""

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
    """
    Load CLIP model and processor once, cache in memory for the process lifetime.

    Why lru_cache(maxsize=1)?
    Loading CLIP (600MB) takes ~8 seconds. With 4 uvicorn workers each forked
    from the same parent process, lru_cache ensures the model is loaded exactly
    once per worker process — not once per request.
    """
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
    # L2-normalize so cosine similarity == dot product (FAISS IndexFlatIP)
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
    """
    Async wrapper around synchronous CLIP text encoding.

    Why run_in_executor and not just async?
    PyTorch's forward pass releases the GIL only partially. We offload it to a
    ThreadPoolExecutor so FastAPI's event loop is never blocked. This keeps the
    API responsive while embeddings are being computed.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_text_sync, text)


async def embed_image(image: Image.Image) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_image_sync, image)


async def embed_query(
    text: str | None = None,
    image: Image.Image | None = None
) -> np.ndarray:
    """
    Fuse text and image embeddings for multimodal queries.

    Strategy: element-wise average of L2-normalized vectors, then re-normalize.
    Why average and not concatenation?
    - FAISS indexes a fixed-dimension space (512 for ViT-B/32). Concatenation
      would double dimensions to 1024, invalidating any pre-built index.
    - Averaging stays in CLIP's joint embedding space, preserving cosine
      similarity semantics.
    """
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