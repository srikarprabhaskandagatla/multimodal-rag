"""
prepare_dataset.py — Build corpus.jsonl from Project Gutenberg (via Gutendex API)
───────────────────────────────────────────────────────────────────────────────────
Gutendex is a free, open API for Project Gutenberg (~70k public-domain books).
No API key, no rate limits, reachable from HPC clusters.

API: https://gutendex.com/books/?languages=en&page={n}
Covers: formats["image/jpeg"] field — direct CDN URL, no download needed.

Usage:
    cd /multimodal-rag
    pip install requests tqdm
    python data/prepare_dataset.py

Output:
    /multimodal-rag/data/corpus.jsonl
"""

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/multimodal-rag")
OUTPUT_FILE = BASE_DIR / "data" / "corpus.jsonl"
TARGET_DOCS = 50_000
REQUEST_DELAY = 2.0   # seconds between pages — Gutendex rate limit is strict
TIMEOUT       = 20    # slightly longer to handle slow HPC network
MAX_RETRIES   = 4     # retry each page up to 4 times before skipping
MAX_CONSEC_FAILURES = 10  # stop only if 10 consecutive pages all fail

GUTENDEX_URL = "https://gutendex.com/books/"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "multimodal-rag-research/1.0"})


def fetch_page(url: str) -> tuple[list[dict], str | None]:
    """
    Fetch one page from Gutendex with retries.
    Returns (list of book dicts, next_page_url or None).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=TIMEOUT)
            if resp.status_code == 429:
                # Rate limited — back off much longer
                wait = 30 * attempt  # 30s, 60s, 90s, 120s
                print(f"\n  [rate-limit] 429 on attempt {attempt}/{MAX_RETRIES}. "
                      f"Waiting {wait}s before retry ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", []), data.get("next")
        except Exception as e:
            wait = REQUEST_DELAY * (2 ** attempt)  # exponential back-off
            print(f"\n  [warn] attempt {attempt}/{MAX_RETRIES} failed ({url}): {e}")
            if attempt < MAX_RETRIES:
                print(f"  [info] retrying in {wait:.1f}s ...")
                time.sleep(wait)
    return [], None


def build_record(doc_id: str, book: dict) -> dict | None:
    """Convert a Gutendex book dict into a corpus record."""
    title = book.get("title", "").strip()
    if not title:
        return None

    # Authors
    authors = [a["name"] for a in book.get("authors", [])]

    # Rich content text: title + author + summary + subjects
    parts = [title]
    if authors:
        parts.append("by " + ", ".join(authors[:2]))
    summaries = book.get("summaries", [])
    if summaries:
        parts.append(summaries[0][:600])
    subjects = book.get("subjects", [])[:6]
    if subjects:
        parts.append("Subjects: " + ", ".join(subjects))
    bookshelves = book.get("bookshelves", [])[:3]
    if bookshelves:
        parts.append("Shelves: " + ", ".join(bookshelves))

    content = " ".join(parts)

    # Cover image URL from formats dict
    formats = book.get("formats", {})
    image_url = formats.get("image/jpeg")
    doc_type  = "multimodal" if image_url else "text"

    return {
        "doc_id":     doc_id,
        "title":      title[:200],
        "source":     "gutenberg",
        "doc_type":   doc_type,
        "content":    content[:1000],
        "image_url":  image_url,
        "image_path": None,
        "metadata": {
            "author":       authors[0] if authors else "",
            "subjects":     subjects,
            "bookshelves":  bookshelves,
            "gutenberg_id": book.get("id"),
            "downloads":    book.get("download_count", 0),
        },
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source : Gutendex (Project Gutenberg) — no API key needed")
    print(f"Target : {TARGET_DOCS:,} documents")
    print(f"Output : {OUTPUT_FILE}\n")

    # Resume from existing file if it already has docs
    written = 0
    start_page = 1
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = sum(1 for line in f if line.strip())
        if existing > 0:
            written = existing
            # Gutendex returns ~32 books/page; estimate which page to resume from
            start_page = max(1, (written // 32) + 1)
            print(f"Resuming from {written:,} existing docs (starting at page ~{start_page})\n")

    page = start_page
    consec_failures = 0

    with open(OUTPUT_FILE, "a") as out_f, tqdm(total=TARGET_DOCS, initial=written, unit="docs") as pbar:
        while written < TARGET_DOCS:
            url = f"{GUTENDEX_URL}?languages=en&page={page}"
            books, _ = fetch_page(url)

            if not books:
                consec_failures += 1
                print(f"\n  [skip] page {page} — no data after {MAX_RETRIES} retries "
                      f"({consec_failures}/{MAX_CONSEC_FAILURES} consecutive failures)")
                if consec_failures >= MAX_CONSEC_FAILURES:
                    print(f"\n  [stop] {MAX_CONSEC_FAILURES} consecutive page failures — "
                          f"Gutendex may be down or we've exhausted the catalog.")
                    break
                page += 1
                time.sleep(REQUEST_DELAY * 2)
                continue

            consec_failures = 0  # reset on any success

            for book in books:
                if written >= TARGET_DOCS:
                    break

                doc_id = f"doc_{written + 1:05d}"
                record = build_record(doc_id, book)
                if record is None:
                    continue

                out_f.write(json.dumps(record) + "\n")
                written += 1
                pbar.update(1)

            if written % 500 == 0:
                out_f.flush()

            page += 1
            time.sleep(REQUEST_DELAY)

    # Summary
    multimodal = text_only = 0
    with open(OUTPUT_FILE) as f:
        for line in f:
            rec = json.loads(line)
            if rec["doc_type"] == "multimodal":
                multimodal += 1
            else:
                text_only += 1

    print(f"\nDone!")
    print(f"  Total written : {written:,}")
    print(f"  Multimodal    : {multimodal:,}  (text + cover image URL)")
    print(f"  Text-only     : {text_only:,}  (no cover available)")
    print(f"\nNext step: python indexing/build_index.py")


if __name__ == "__main__":
    main()
