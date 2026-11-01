# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local Markdown storage and retrieval helpers for Smart Knowledge Agent."""

import hashlib
import os
from pathlib import Path
import re
from typing import Any

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


def get_knowledge_base_dir() -> Path:
    """Returns the directory path where Markdown knowledge files are stored."""
    env_dir = os.environ.get("KNOWLEDGE_BASE_DIR")
    kb_dir = Path(env_dir).resolve() if env_dir else DEFAULT_KNOWLEDGE_DIR
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir


def sanitize_slug(slug: str, title: str = "") -> str:
    """Sanitizes a slug into a safe lowercase ASCII filename stem."""
    candidate = (slug or "").strip().lower()
    candidate = re.sub(r"[^a-z0-9\-_]+", "-", candidate)
    candidate = re.sub(r"-{2,}", "-", candidate).strip("-_")
    if not candidate:
        digest = hashlib.sha1((title or "knowledge").encode("utf-8")).hexdigest()[:8]
        candidate = f"genai-note-{digest}"
    return candidate[:64]


def save_markdown_knowledge(
    title: str,
    content: str,
    slug: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Saves Generative AI knowledge as a Markdown file in knowledge_base/.

    Args:
        title: Title of the knowledge note.
        content: Markdown body text to save.
        slug: Filename slug (without .md extension).
        tags: Optional list of Generative AI tags.

    Returns:
        Dictionary containing status, filename, file_path, title, and tags.
    """
    kb_dir = get_knowledge_base_dir()
    clean_title = (title or "生成AIナレッジ").strip()
    safe_slug = sanitize_slug(slug, clean_title)
    filename = f"{safe_slug}.md"
    file_path = (kb_dir / filename).resolve()

    # Enforce path containment inside kb_dir
    if kb_dir not in file_path.parents and file_path.parent != kb_dir:
        raise ValueError(f"Invalid target path outside knowledge base: {file_path}")

    clean_tags = [t.strip() for t in (tags or ["Generative AI"]) if t and t.strip()]
    tags_line = ", ".join(clean_tags) if clean_tags else "Generative AI"

    body = (content or "").replace("\\n", "\n").strip()
    if body.startswith(f"# {clean_title}"):
        markdown_text = f"{body}\n"
    else:
        markdown_text = (
            f"# {clean_title}\n\n"
            f"- **Tags**: {tags_line}\n\n"
            f"## 内容\n"
            f"{body}\n"
        )

    file_path.write_text(markdown_text, encoding="utf-8")
    return {
        "status": "saved",
        "filename": filename,
        "file_path": str(file_path),
        "title": clean_title,
        "tags": clean_tags,
        "markdown": markdown_text,
    }


def list_markdown_files() -> list[str]:
    """Lists all Markdown filenames in the knowledge base directory."""
    kb_dir = get_knowledge_base_dir()
    return sorted(p.name for p in kb_dir.glob("*.md") if p.is_file())


def search_markdown_knowledge(
    query: str,
    keywords: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Searches Markdown files in knowledge_base/ for relevant Generative AI notes.

    Args:
        query: User's natural language question.
        keywords: Additional search keywords extracted from the query.
        top_k: Maximum number of matching documents to return.

    Returns:
        List of matching document dicts with filename, score, and content.
    """
    kb_dir = get_knowledge_base_dir()
    md_files = sorted(p for p in kb_dir.glob("*.md") if p.is_file())
    if not md_files:
        return []

    raw_terms: list[str] = []
    if keywords:
        raw_terms.extend(k.strip().lower() for k in keywords if k and k.strip())

    # Extract tokens from query (ASCII words and CJK/katakana/kanji chunks)
    query_lower = (query or "").lower()
    raw_terms.extend(
        tok
        for tok in re.findall(
            r"[a-z0-9_\-\.]{2,}|[ァ-ヶー]{2,}|[一-龥]{2,}", query_lower
        )
        if tok not in {"について", "とは", "教えて", "ください", "ですか", "ますか", "what", "how", "the"}
    )

    # Deduplicate while preserving order
    search_terms = list(dict.fromkeys(t for t in raw_terms if t))

    scored_docs: list[dict[str, Any]] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        text_lower = text.lower()
        name_lower = path.stem.lower()
        score = 0
        matched_terms: list[str] = []
        for term in search_terms:
            if term in name_lower:
                score += 3
                matched_terms.append(term)
            elif term in text_lower:
                score += 2
                matched_terms.append(term)

        scored_docs.append(
            {
                "filename": path.name,
                "score": score,
                "matched_terms": matched_terms,
                "content": text,
            }
        )

    scored_docs.sort(key=lambda d: (-int(d["score"]), str(d["filename"])))
    positive_matches = [d for d in scored_docs if int(d["score"]) > 0]
    if positive_matches:
        return positive_matches[:top_k]

    # When the knowledge base is small, include up to top_k docs as fallback context
    return scored_docs[:top_k]
