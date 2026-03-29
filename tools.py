from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import trafilatura
from ddgs import DDGS

from config import get_settings
from retriever import get_retriever

settings = get_settings()
OUTPUT_DIR = settings.output_path
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for relevant pages and return concise results with URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Read and extract the main textual content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to read"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "Save the final Markdown report to disk and return the saved path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Markdown filename"},
                    "content": {"type": "string", "description": "Markdown content"},
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "Search the local knowledge base using hybrid retrieval and reranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Knowledge base search query"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def web_search(query: str) -> str:
    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=settings.max_search_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
    if not results:
        return f"No web results found for query: {query}"
    return json.dumps(results, ensure_ascii=False, indent=2)[: settings.max_search_content_length]



def read_url(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: failed to download URL: {url}"
        extracted = trafilatura.extract(downloaded, include_links=True, include_formatting=False)
        if not extracted:
            return f"Error: failed to extract readable content from URL: {url}"
        return extracted[: settings.max_url_content_length]
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        return f"Error reading URL {url}: {exc}"



def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^\w\-.а-яА-ЯіїєІЇЄ]+", "_", filename.strip())
    if not cleaned:
        cleaned = "research_report.md"
    if not cleaned.lower().endswith(".md"):
        cleaned += ".md"
    return cleaned



def write_report(filename: str, content: str) -> str:
    try:
        safe_name = sanitize_filename(filename)
        path = OUTPUT_DIR / safe_name
        path.write_text(content, encoding="utf-8")
        return f"Report saved to {path}"
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        return f"Error saving report: {exc}"



def knowledge_search(query: str) -> str:
    """Search the local knowledge base using hybrid retrieval + reranking."""
    try:
        retriever = get_retriever()
        docs = retriever.hybrid_search(query)
    except Exception as exc:
        return f"Error searching knowledge base: {exc}"

    if not docs:
        return f"No local knowledge base results found for query: {query}"

    lines = [f"Found {len(docs)} knowledge base results for query: {query}"]
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:800] + ("..." if len(snippet) > 800 else "")
        lines.append(f"{idx}. [{source}{page_label}] {snippet}")
    return "\n".join(lines)



def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    registry = {
        "web_search": web_search,
        "read_url": read_url,
        "write_report": write_report,
        "knowledge_search": knowledge_search,
    }
    tool = registry.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    try:
        return tool(**arguments)
    except TypeError as exc:
        return f"Error: invalid arguments for tool '{name}': {exc}"
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        return f"Error executing tool '{name}': {exc}"
