import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import faiss
import numpy as np
import pytest

from app.retriever import FAISSRetriever

EMBEDDING_DIM = 512


def _make_temp_index(num_docs: int = 10) -> tuple[str, list[str]]:
    tmpdir = tempfile.mkdtemp()
    hnsw = faiss.IndexHNSWFlat(EMBEDDING_DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIDMap(hnsw)

    id_map = [f"doc_{i:03d}" for i in range(num_docs)]
    vecs = np.random.randn(num_docs, EMBEDDING_DIM).astype(np.float32)
    # L2-normalize so inner product == cosine similarity
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = np.arange(num_docs, dtype=np.int64)
    index.add_with_ids(vecs, ids)

    faiss.write_index(index, str(Path(tmpdir) / "index.faiss"))
    with open(Path(tmpdir) / "id_map.pkl", "wb") as f:
        pickle.dump(id_map, f)

    return tmpdir, id_map


def _fake_metadata(doc_ids: list[str]) -> list[dict]:
    return [
        {
            "doc_id": doc_id,
            "title": f"Title for {doc_id}",
            "source": "test_corpus",
            "doc_type": "text",
            "content_preview": f"Preview text for {doc_id}.",
            "metadata": {},
        }
        for doc_id in doc_ids
    ]


def test_retriever_loads_index():
    tmpdir, id_map = _make_temp_index(num_docs=10)

    retriever = FAISSRetriever()
    with patch.dict("os.environ", {"FAISS_INDEX_PATH": tmpdir}):
        retriever.load()

    assert retriever.index is not None
    assert retriever.index.ntotal == 10
    assert retriever.id_map == id_map


def test_retriever_load_missing_index_raises():
    retriever = FAISSRetriever()
    with patch.dict("os.environ", {"FAISS_INDEX_PATH": "/nonexistent/path"}):
        with pytest.raises(FileNotFoundError):
            retriever.load()


@pytest.mark.asyncio
async def test_retrieve_text_returns_top_k():
    tmpdir, _ = _make_temp_index(num_docs=20)

    retriever = FAISSRetriever()
    with patch.dict("os.environ", {"FAISS_INDEX_PATH": tmpdir, "FAISS_TOP_K": "20"}):
        retriever.load()

    mock_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    mock_vec /= np.linalg.norm(mock_vec)

    with patch("app.retriever.embed_query", new=AsyncMock(return_value=mock_vec)), \
         patch("app.retriever.get_doc_metadata", new=AsyncMock(side_effect=_fake_metadata)):
        results = await retriever.retrieve(text="test query", top_k=5)

    assert len(results) <= 5
    # Results should be sorted by score descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_retrieve_multimodal_fuses_embeddings():
    from PIL import Image as PILImage
    dummy_image = PILImage.new("RGB", (224, 224))

    tmpdir, _ = _make_temp_index(num_docs=5)
    retriever = FAISSRetriever()
    with patch.dict("os.environ", {"FAISS_INDEX_PATH": tmpdir, "FAISS_TOP_K": "5"}):
        retriever.load()

    mock_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    mock_vec /= np.linalg.norm(mock_vec)
    mock_embed = AsyncMock(return_value=mock_vec)

    with patch("app.retriever.embed_query", new=mock_embed), \
         patch("app.retriever.get_doc_metadata", new=AsyncMock(side_effect=_fake_metadata)):
        await retriever.retrieve(text="red sneakers", image=dummy_image, top_k=3)

    # embed_query must have been called with both text and image
    mock_embed.assert_called_once_with(text="red sneakers", image=dummy_image)


@pytest.mark.asyncio
async def test_retrieve_returns_scores():
    tmpdir, _ = _make_temp_index(num_docs=10)
    retriever = FAISSRetriever()
    with patch.dict("os.environ", {"FAISS_INDEX_PATH": tmpdir, "FAISS_TOP_K": "10"}):
        retriever.load()

    mock_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    mock_vec /= np.linalg.norm(mock_vec)

    with patch("app.retriever.embed_query", new=AsyncMock(return_value=mock_vec)), \
         patch("app.retriever.get_doc_metadata", new=AsyncMock(side_effect=_fake_metadata)):
        results = await retriever.retrieve(text="query", top_k=5)

    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)


@pytest.mark.asyncio
async def test_retrieve_empty_db_returns_empty():
    tmpdir, _ = _make_temp_index(num_docs=5)
    retriever = FAISSRetriever()
    with patch.dict("os.environ", {"FAISS_INDEX_PATH": tmpdir, "FAISS_TOP_K": "5"}):
        retriever.load()

    mock_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    mock_vec /= np.linalg.norm(mock_vec)

    with patch("app.retriever.embed_query", new=AsyncMock(return_value=mock_vec)), \
         patch("app.retriever.get_doc_metadata", new=AsyncMock(return_value=[])):
        results = await retriever.retrieve(text="query", top_k=5)

    assert results == []
