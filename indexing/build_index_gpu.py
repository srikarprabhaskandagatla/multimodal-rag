import json
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_index_gpu.log"),
    ],
)
logger = logging.getLogger(__name__)

CORPUS_PATH      = os.getenv("CORPUS_PATH", "raw_dataset/data/corpus.jsonl")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "raw_dataset/faiss_index")
CLIP_MODEL_NAME  = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "512"))   # tune to GPU VRAM
NUM_WORKERS      = int(os.getenv("NUM_WORKERS", "8"))    # DataLoader workers
FETCH_WORKERS    = int(os.getenv("FETCH_WORKERS", "32")) # image download threads
EMBEDDING_DIM    = 512   # CLIP ViT-B/32
HNSW_M           = 32


def fetch_image(args: tuple) -> tuple[int, Image.Image | None]:
    idx, url = args
    if not url:
        return idx, None
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "multimodal-rag/1.0"})
        if resp.status_code == 200:
            return idx, Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return idx, None


def fetch_all_images(docs: list[dict]) -> list[Image.Image | None]:
    images = [None] * len(docs)
    tasks = [
        (i, doc.get("image_url") or doc.get("image_path"))
        for i, doc in enumerate(docs)
        if doc.get("image_url") or doc.get("image_path")
    ]
    logger.info("Downloading %d images with %d threads ...", len(tasks), FETCH_WORKERS)
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_image, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            idx, img = future.result()
            images[idx] = img
            done += 1
            if done % 1000 == 0:
                logger.info("  Downloaded %d / %d images", done, len(tasks))
    logger.info("Image download complete.")
    return images


class CorpusDataset(Dataset):
    def __init__(self, docs: list[dict], images: list[Image.Image | None], processor: CLIPProcessor):
        self.docs = docs
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        image = self.images[idx]
        text = doc.get("content") or doc.get("title") or ""

        text = text[:300]

        return {
            "idx": idx,
            "doc_id": doc["doc_id"],
            "text": text,
            "has_image": image is not None,
            "image": image if image is not None else Image.new("RGB", (224, 224)),
        }


def collate_fn(processor: CLIPProcessor):
    def _collate(batch):
        texts  = [b["text"] for b in batch]
        images = [b["image"] for b in batch]
        has_image = [b["has_image"] for b in batch]
        idxs   = [b["idx"] for b in batch]
        doc_ids = [b["doc_id"] for b in batch]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return inputs, idxs, doc_ids, has_image
    return _collate

def build_faiss_index(dim: int) -> faiss.Index:
    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    return faiss.IndexIDMap(hnsw)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning("No GPU detected — this will be slow. Use a GPU node.")

    # Load CLIP
    logger.info("Loading CLIP model: %s", CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    logger.info("CLIP loaded.")

    # Load corpus
    corpus_path = Path(CORPUS_PATH)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    with open(corpus_path) as f:
        docs = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d documents from corpus.", len(docs))

    images = fetch_all_images(docs)

    dataset = CorpusDataset(docs, images, processor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn(processor),
        pin_memory=(device == "cuda"),
    )

    # Embed all docs
    index = build_faiss_index(EMBEDDING_DIM)
    id_map: list[str] = []
    total_embedded = 0

    logger.info("Embedding %d docs in batches of %d ...", len(docs), BATCH_SIZE)

    with torch.no_grad():
        for batch_inputs, idxs, doc_ids, has_images in loader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            text_features = model.get_text_features(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
            )

            image_features = model.get_image_features(
                pixel_values=batch_inputs["pixel_values"],
            )

            has_img_tensor = torch.tensor(has_images, dtype=torch.bool, device=device)
            fused = torch.where(
                has_img_tensor.unsqueeze(1),
                (text_features + image_features) / 2.0,
                text_features,
            )

            # L2 normalize
            fused = torch.nn.functional.normalize(fused, dim=-1)
            vecs = fused.cpu().numpy().astype(np.float32)

            # Add to FAISS
            start_id = len(id_map)
            faiss_ids = np.arange(start_id, start_id + len(doc_ids), dtype=np.int64)
            index.add_with_ids(vecs, faiss_ids)
            id_map.extend(doc_ids)

            total_embedded += len(doc_ids)
            logger.info("Embedded %d / %d docs", total_embedded, len(docs))

    logger.info("FAISS index built: %d vectors", index.ntotal)

    output_dir = Path(FAISS_INDEX_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "id_map.pkl", "wb") as f:
        pickle.dump(id_map, f)

    logger.info("Saved index.faiss and id_map.pkl to %s", output_dir)
    logger.info("Done. %d documents indexed.", len(id_map))


if __name__ == "__main__":
    main()
