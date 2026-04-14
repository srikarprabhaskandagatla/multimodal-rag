"""
cache.py — Redis Query Cache
─────────────────────────────
WHY REDIS (and not Memcached or an in-process dict)?

- In-process dict: not shared across uvicorn workers. Worker A caches a result
  that Worker B never sees — cache hit rate ~25% with 4 workers.

- Memcached: no per-key TTL, no persistence. If the pod restarts, all cached
  embeddings are lost — next request pays full CLIP + FAISS cost again.

- Redis: shared across all workers via TCP, native per-key TTL, RDB snapshots
  for persistence across restarts, and sub-millisecond latency on a local
  network. The correct choice for a multi-worker async ML API.

CACHE KEY DESIGN:
  "rag:{text_hash}:{has_image}"
  - SHA-256 of the text query (fixed-length, safe for any input)
  - has_image flag distinguishes text-only from multimodal queries with same text
  - TTL: 3600s (1 hour) — hot queries re-embed frequently; stale after 1h
"""

import hashlib
import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Module-level async Redis client — created once, shared across all requests
# in the worker process. redis.asyncio is non-blocking and coroutine-safe.
_redis_client: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,  # Return str, not bytes
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


def _make_cache_key(text: str | None, has_image: bool) -> str:
    """
    Build a deterministic cache key from query parameters.
    SHA-256 keeps the key short and safe regardless of query length or content.
    """
    raw = f"{text or ''}:{has_image}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"rag:{digest}:{int(has_image)}"


async def get_cached(
    text: str | None,
    has_image: bool,
) -> list[dict[str, Any]] | None:
    """
    Look up cached retrieval results for this query.
    Returns None on cache miss or any Redis error (fail-open: never block a query).
    """
    try:
        client = _get_redis()
        key = _make_cache_key(text, has_image)
        value = await client.get(key)
        if value:
            logger.debug("Cache hit for key %s", key)
            return json.loads(value)
        return None
    except Exception as e:
        # Redis errors must never crash the API — log and proceed without cache
        logger.warning("Redis get failed (key=%s): %s", key, e)
        return None


async def set_cached(
    text: str | None,
    has_image: bool,
    results: list[dict[str, Any]],
) -> None:
    """
    Store retrieval results in Redis with TTL.
    Fails silently — a cache write error should never fail the user's request.
    """
    try:
        client = _get_redis()
        key = _make_cache_key(text, has_image)
        await client.setex(key, CACHE_TTL, json.dumps(results, default=str))
        logger.debug("Cached %d results for key %s (TTL=%ds)", len(results), key, CACHE_TTL)
    except Exception as e:
        logger.warning("Redis set failed (key=%s): %s", key, e)
