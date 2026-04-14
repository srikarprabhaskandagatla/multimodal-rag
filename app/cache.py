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

_redis_client: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True, 
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


def _make_cache_key(text: str | None, has_image: bool) -> str:
    raw = f"{text or ''}:{has_image}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"rag:{digest}:{int(has_image)}"


async def get_cached(
    text: str | None,
    has_image: bool,
) -> list[dict[str, Any]] | None:
    try:
        client = _get_redis()
        key = _make_cache_key(text, has_image)
        value = await client.get(key)
        if value:
            logger.debug("Cache hit for key %s", key)
            return json.loads(value)
        return None
    except Exception as e:
        logger.warning("Redis get failed (key=%s): %s", key, e)
        return None


async def set_cached(
    text: str | None,
    has_image: bool,
    results: list[dict[str, Any]],
) -> None:
    try:
        client = _get_redis()
        key = _make_cache_key(text, has_image)
        await client.setex(key, CACHE_TTL, json.dumps(results, default=str))
        logger.debug("Cached %d results for key %s (TTL=%ds)", len(results), key, CACHE_TTL)
    except Exception as e:
        logger.warning("Redis set failed (key=%s): %s", key, e)
