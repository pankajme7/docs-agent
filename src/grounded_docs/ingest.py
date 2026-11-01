"""Download a curated set of public docs and populate the vector index."""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup

from grounded_docs.chunking import chunk_text
from grounded_docs.config import Settings
from grounded_docs.embeddings import EmbeddingModel
from grounded_docs.vector_store import VectorStore

logger = logging.getLogger(__name__)
USER_AGENT = "GroundedDocsLearningProject/0.1 (+https://github.com/pankajme7)"


@dataclass
class Page:
    url: str
    title: str
    text: str


def load_source_urls(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    urls = payload.get("sources", [])
    if not isinstance(urls, list) or not urls:
        raise ValueError(f"No source URLs found in {path}")
    return [str(url) for url in urls]


def extract_page(url: str, html: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script, style, nav, footer, header, aside, noscript, .td-sidebar, .td-navbar"):
        node.decompose()

    article = soup.select_one("main article") or soup.select_one("article") or soup.select_one("main") or soup.body
    if article is None:
        raise ValueError(f"Could not locate main content at {url}")

    title_node = article.find(["h1"]) or soup.find("title")
    title = title_node.get_text(" ", strip=True) if title_node else urlparse(url).path.rstrip("/").split("/")[-1]
    blocks: list[str] = []
    for node in article.find_all(["h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if node.name in {"h2", "h3", "h4"}:
            text = f"\n\n## {text}\n\n"
        elif node.name == "li":
            text = f"- {text}"
        blocks.append(text)
    content = "\n\n".join(blocks)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) < 150:
        raise ValueError(f"Extracted too little text from {url}; page layout may have changed")
    return Page(url=url, title=title, text=content)


def fetch_page(client: httpx.Client, url: str) -> Page:
    response = client.get(url)
    response.raise_for_status()
    return extract_page(url, response.text)


def ingest(settings: Settings, sources_file: Path | None = None, reset: bool = False) -> dict:
    sources_file = sources_file or settings.project_root / "data" / "sources.yaml"
    urls = load_source_urls(sources_file)
    embedding_model = EmbeddingModel(settings.embedding_model)
    store = VectorStore(settings)
    if reset:
        store.reset()

    succeeded = 0
    failed: list[dict[str, str]] = []
    with httpx.Client(
        timeout=httpx.Timeout(35.0, connect=15.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for index, url in enumerate(urls, start=1):
            try:
                page = fetch_page(client, url)
                chunks = chunk_text(page.text, settings.chunk_words, settings.chunk_overlap_words)
                if not chunks:
                    raise ValueError("No text chunks extracted")
                ids = []
                metadatas = []
                for chunk_index, chunk in enumerate(chunks):
                    digest = hashlib.sha1(f"{url}|{chunk_index}|{chunk}".encode()).hexdigest()[:16]
                    ids.append(f"{digest}")
                    metadatas.append({
                        "source_url": url,
                        "title": page.title[:500],
                        "section": "",
                        "chunk_index": chunk_index,
                    })
                vectors = embedding_model.encode_documents(chunks)
                store.replace_source(url, ids, chunks, metadatas, vectors)
                succeeded += 1
                logger.info("[%d/%d] indexed %s (%d chunks)", index, len(urls), page.title, len(chunks))
                time.sleep(0.15)
            except Exception as exc:  # Keep processing the remaining curated pages.
                logger.exception("Failed to ingest %s", url)
                failed.append({"url": url, "error": str(exc)})

    result = {"requested": len(urls), "succeeded": succeeded, "failed": failed, "chunks": store.count()}
    if succeeded == 0:
        raise RuntimeError(f"No pages were ingested successfully. Details: {failed}")
    return result
