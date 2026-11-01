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

"""Local Markdown storage, async background persistence, context budget management,
and guided error-handling tools for Smart Knowledge Agent.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

from google.adk.tools import FunctionTool
from pydantic import BaseModel, Field

from app.app_utils.observability import log_structured_event, redact_pii

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
MAX_DOC_CHARS_DEFAULT = 2000
MAX_TOTAL_CONTEXT_CHARS_DEFAULT = 6000

# Registry of in-flight background file persistence tasks to prevent GC and allow awaiting
_BACKGROUND_SAVE_TASKS: set[asyncio.Task[dict[str, Any]]] = set()


class GuidedToolError(BaseModel):
    """Structured error response providing actionable recovery instructions to the LLM."""

    status: Literal["error", "no_matches"] = Field(
        default="error",
        description="Error or non-fatal diagnostic status.",
    )
    error_code: str = Field(
        description="Machine-readable error identifier (e.g., EMPTY_CONTENT, INVALID_SLUG)."
    )
    message: str = Field(
        description="Human- and LLM-readable description of what went wrong."
    )
    recovery_instructions: str = Field(
        description=(
            "Step-by-step instructions telling the LLM how to adjust parameters "
            "or call an alternative tool to recover from this error."
        )
    )
    suggested_fix: dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested corrected parameter values for retrying the tool call.",
    )


class SaveKnowledgeOutput(BaseModel):
    """Structured output schema for knowledge persistence operations."""

    status: Literal["saved", "scheduled_async", "error"] = Field(
        description="Persistence outcome status."
    )
    filename: str = Field(default="", description="Saved Markdown filename.")
    file_path: str = Field(default="", description="Resolved path on disk.")
    title: str = Field(default="", description="Sanitized document title.")
    tags: list[str] = Field(default_factory=list, description="Applied topic tags.")
    markdown: str = Field(default="", description="Formatted Markdown text.")
    pii_redacted: bool = Field(
        default=False,
        description="True if sensitive PII was automatically redacted before saving.",
    )
    error_code: str | None = Field(
        default=None, description="Error code if status=='error'."
    )
    recovery_instructions: str | None = Field(
        default=None,
        description="LLM recovery instructions when status=='error'.",
    )


def build_guided_error(
    *,
    error_code: str,
    message: str,
    recovery_instructions: str,
    suggested_fix: dict[str, Any] | None = None,
    status: Literal["error", "no_matches"] = "error",
) -> dict[str, Any]:
    """Constructs a standardized dictionary with guided recovery instructions for the LLM."""
    err = GuidedToolError(
        status=status,
        error_code=error_code,
        message=message,
        recovery_instructions=recovery_instructions,
        suggested_fix=suggested_fix or {},
    )
    log_structured_event(
        event_type="guided_tool_error",
        message=message,
        severity="WARNING" if status == "error" else "INFO",
        error_code=error_code,
        recovery_instructions=recovery_instructions,
    )
    return err.model_dump()


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


def _normalize_markdown_body(body: str, clean_title: str) -> str:
    """Removes duplicate leading H1 headings from generated Markdown content."""
    text = (body or "").replace("\\n", "\n").strip()
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("# "):
        first_heading = lines[0].lstrip()[2:].strip()
        if (
            first_heading == clean_title
            or first_heading in clean_title
            or clean_title in first_heading
        ):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines.pop(0)
    # Demote any remaining top-level H1 headings inside body to H2 for clean hierarchy
    normalized_lines: list[str] = []
    for line in lines:
        if line.startswith("# "):
            normalized_lines.append("## " + line[2:])
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def save_markdown_knowledge(
    title: str,
    content: str,
    slug: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Saves Generative AI knowledge as a Markdown file in knowledge_base/ with
    PII redaction and guided error recovery.

    Args:
        title: Title of the knowledge note.
        content: Markdown body text to save.
        slug: Filename slug (without .md extension).
        tags: Optional list of Generative AI tags.

    Returns:
        Dictionary containing status ('saved' or 'error'), filename, file_path,
        title, tags, markdown, or guided recovery_instructions on error.
    """
    raw_content = (content or "").strip()
    if not raw_content:
        return build_guided_error(
            error_code="EMPTY_CONTENT",
            message="Cannot save an empty knowledge note: 'content' parameter was blank.",
            recovery_instructions=(
                "Extract and summarize the user's Generative AI knowledge into a non-empty "
                "Markdown string (at least 1-2 sentences or bullet points) and retry "
                "`save_markdown_knowledge(title=..., content=..., slug=..., tags=...)`."
            ),
            suggested_fix={
                "title": title or "生成AIナレッジ",
                "content": "## 概要\n（ここに生成AIに関する知見を記述してください）",
            },
        )

    kb_dir = get_knowledge_base_dir()
    clean_title = redact_pii((title or "生成AIナレッジ").strip())
    sanitized_body = redact_pii(raw_content)
    pii_was_redacted = (clean_title != (title or "生成AIナレッジ").strip()) or (
        sanitized_body != raw_content
    )

    safe_slug = sanitize_slug(slug, clean_title)
    filename = f"{safe_slug}.md"
    file_path = (kb_dir / filename).resolve()

    # Guided error handling instead of raising unhandled exceptions on path anomalies
    if kb_dir not in file_path.parents and file_path.parent != kb_dir:
        fallback_slug = sanitize_slug("", clean_title)
        return build_guided_error(
            error_code="INVALID_TARGET_PATH",
            message=f"Target path '{file_path}' resolved outside knowledge_base directory.",
            recovery_instructions=(
                f"Do not include directory separators or '..' in `slug`. "
                f"Retry `save_markdown_knowledge` with `slug='{fallback_slug}'`."
            ),
            suggested_fix={"slug": fallback_slug},
        )

    clean_tags = [
        redact_pii(t.strip())
        for t in (tags or ["Generative AI"])
        if t and isinstance(t, str) and t.strip()
    ]
    tags_line = ", ".join(clean_tags) if clean_tags else "Generative AI"

    normalized_body = _normalize_markdown_body(sanitized_body, clean_title)
    markdown_text = (
        f"# {clean_title}\n\n- **Tags**: {tags_line}\n\n## 内容\n{normalized_body}\n"
    )

    try:
        file_path.write_text(markdown_text, encoding="utf-8")
    except OSError as exc:
        return build_guided_error(
            error_code="STORAGE_WRITE_FAILED",
            message=f"Failed to write Markdown file '{filename}': {exc}",
            recovery_instructions=(
                "Verify that the knowledge_base directory is writable or retry with a "
                "shorter ASCII slug (under 32 characters). If disk persistence remains "
                "unavailable, present the formatted Markdown note directly in the chat response."
            ),
            suggested_fix={"slug": safe_slug[:32]},
        )

    output = SaveKnowledgeOutput(
        status="saved",
        filename=filename,
        file_path=str(file_path),
        title=clean_title,
        tags=clean_tags,
        markdown=markdown_text,
        pii_redacted=pii_was_redacted,
    )
    log_structured_event(
        event_type="knowledge_file_saved",
        message=f"Saved knowledge file '{filename}'.",
        filename=filename,
        title=clean_title,
        tags=clean_tags,
        pii_redacted=pii_was_redacted,
    )
    return output.model_dump()


async def save_markdown_knowledge_async(
    title: str,
    content: str,
    slug: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Asynchronously saves Generative AI knowledge off the main event loop thread."""
    return await asyncio.to_thread(
        save_markdown_knowledge,
        title=title,
        content=content,
        slug=slug,
        tags=tags,
    )


def schedule_background_knowledge_save(
    title: str,
    content: str,
    slug: str = "",
    tags: list[str] | None = None,
) -> asyncio.Task[dict[str, Any]]:
    """Schedules non-blocking background persistence of a Markdown knowledge note."""
    task = asyncio.create_task(
        save_markdown_knowledge_async(
            title=title,
            content=content,
            slug=slug,
            tags=tags,
        )
    )
    _BACKGROUND_SAVE_TASKS.add(task)

    def _on_done(done_task: asyncio.Task[dict[str, Any]]) -> None:
        _BACKGROUND_SAVE_TASKS.discard(done_task)
        try:
            res = done_task.result()
            log_structured_event(
                event_type="background_knowledge_save_completed",
                message=f"Background save finished with status '{res.get('status')}'.",
                filename=res.get("filename"),
                status=res.get("status"),
            )
        except Exception as exc:  # pragma: no cover
            log_structured_event(
                event_type="background_knowledge_save_failed",
                message=f"Background save raised error: {exc}",
                severity="ERROR",
            )

    task.add_done_callback(_on_done)
    return task


async def wait_for_background_saves(timeout: float = 5.0) -> None:
    """Waits for any pending background save tasks (useful in tests and graceful shutdown)."""
    if not _BACKGROUND_SAVE_TASKS:
        return
    await asyncio.wait(list(_BACKGROUND_SAVE_TASKS), timeout=timeout)


def list_markdown_files() -> list[str]:
    """Lists all Markdown filenames in the knowledge base directory."""
    kb_dir = get_knowledge_base_dir()
    return sorted(p.name for p in kb_dir.glob("*.md") if p.is_file())


async def list_markdown_files_async() -> list[str]:
    """Asynchronously lists all Markdown filenames in the knowledge base directory."""
    return await asyncio.to_thread(list_markdown_files)


def _truncate_for_context_budget(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncates document content to prevent context window bloat."""
    if len(text) <= max_chars:
        return text, False
    return (
        text[:max_chars].rstrip() + "\n...(コンテキスト予算保護のため以降を省略)...",
        True,
    )


def search_markdown_knowledge(
    query: str,
    keywords: list[str] | None = None,
    top_k: int = 5,
    max_chars_per_doc: int = MAX_DOC_CHARS_DEFAULT,
    max_total_chars: int = MAX_TOTAL_CONTEXT_CHARS_DEFAULT,
) -> list[dict[str, Any]]:
    """Searches Markdown files in knowledge_base/ for relevant Generative AI notes
    with context budget enforcement.

    Args:
        query: User's natural language question.
        keywords: Additional search keywords extracted from the query.
        top_k: Maximum number of matching documents to return.
        max_chars_per_doc: Maximum characters per document to prevent context bloat.
        max_total_chars: Maximum total characters across all returned documents.

    Returns:
        List of matching document dicts with filename, score, matched_terms, and content.
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
        if tok
        not in {
            "について",
            "とは",
            "教えて",
            "ください",
            "ですか",
            "ますか",
            "what",
            "how",
            "the",
        }
    )

    search_terms = list(dict.fromkeys(t for t in raw_terms if t))

    scored_docs: list[dict[str, Any]] = []
    for path in md_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        text_lower = text.lower()
        name_lower = path.stem.lower()
        first_lines = "\n".join(text.splitlines()[:4]).lower()
        score = 0
        matched_terms: list[str] = []
        for term in search_terms:
            if term in name_lower:
                score += 4
                matched_terms.append(term)
            elif term in first_lines:
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

    # Context bloat management: enforce per-doc and total character limits
    selected: list[dict[str, Any]] = []
    total_chars = 0
    for doc in positive_matches[: max(1, top_k)]:
        remaining_budget = max_total_chars - total_chars
        if remaining_budget <= 100:
            break
        doc_limit = min(max_chars_per_doc, remaining_budget)
        truncated_content, was_truncated = _truncate_for_context_budget(
            str(doc["content"]), doc_limit
        )
        total_chars += len(truncated_content)
        selected.append(
            {
                **doc,
                "content": truncated_content,
                "truncated_for_context_budget": was_truncated,
            }
        )

    return selected


async def search_markdown_knowledge_async(
    query: str,
    keywords: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Asynchronously searches Markdown files in knowledge_base/."""
    return await asyncio.to_thread(
        search_markdown_knowledge,
        query=query,
        keywords=keywords,
        top_k=top_k,
    )


def search_markdown_knowledge_guided(
    query: str,
    keywords: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Guided wrapper around search_markdown_knowledge that returns structured
    recovery instructions when a query is empty or yields no matching files.

    Args:
        query: Natural language search query about Generative AI.
        keywords: Optional list of extracted keywords (Japanese or English).
        top_k: Maximum number of documents to retrieve.

    Returns:
        Dictionary with 'status' ('success', 'no_matches', or 'error'),
        'matches', 'available_files', and 'recovery_instructions' when applicable.
    """
    clean_query = (query or "").strip()
    if not clean_query and not keywords:
        return build_guided_error(
            error_code="EMPTY_SEARCH_QUERY",
            message="Search query and keywords were both empty.",
            recovery_instructions=(
                "Provide a non-empty `query` string or a list of `keywords` (e.g., "
                "['ADK', 'Workflow', 'RAG', 'Gemini']) and retry `search_markdown_knowledge_guided`, "
                "or call `list_markdown_files` to inspect all stored filenames first."
            ),
            suggested_fix={
                "query": "Generative AI overview",
                "keywords": ["Gemini", "ADK", "RAG"],
            },
        )

    all_files = list_markdown_files()
    if not all_files:
        return build_guided_error(
            status="no_matches",
            error_code="KNOWLEDGE_BASE_EMPTY",
            message="The knowledge_base/ directory currently contains no Markdown files.",
            recovery_instructions=(
                "Inform the user that no local Markdown files are stored yet, answer their "
                "Generative AI question using general domain knowledge, and offer to save "
                "key takeaways using `save_markdown_knowledge`."
            ),
        )

    matches = search_markdown_knowledge(
        query=clean_query, keywords=keywords, top_k=top_k
    )
    if not matches:
        return {
            **build_guided_error(
                status="no_matches",
                error_code="NO_MATCHING_DOCUMENTS",
                message=f"No Markdown files matched query='{clean_query}' with keywords={keywords}.",
                recovery_instructions=(
                    f"Available files in knowledge_base/ are: {all_files}. "
                    "To recover: (1) Retry search using broader English/Japanese terms matching "
                    "those filenames, OR (2) Explicitly state to the user that the current local "
                    "knowledge base does not contain a note on this specific topic and answer "
                    "using general Generative AI knowledge."
                ),
                suggested_fix={"available_files": all_files},
            ),
            "matches": [],
            "available_files": all_files,
        }

    return {
        "status": "success",
        "matches": matches,
        "available_files": all_files,
        "total_matched": len(matches),
    }


def delete_markdown_knowledge(slug_or_filename: str) -> dict[str, Any]:
    """Deletes a Markdown knowledge file from knowledge_base/ with guided error handling.

    Args:
        slug_or_filename: The filename (e.g., 'few-shot-prompting.md') or slug to delete.

    Returns:
        Dictionary with status ('deleted' or 'error') and recovery_instructions on error.
    """
    raw = (slug_or_filename or "").strip()
    if not raw:
        return build_guided_error(
            error_code="EMPTY_FILENAME",
            message="No filename or slug was provided to delete_markdown_knowledge.",
            recovery_instructions=(
                "Call `list_markdown_files` first to view existing filenames, then pass "
                "an exact filename to `delete_markdown_knowledge`."
            ),
        )

    stem = raw[:-3] if raw.endswith(".md") else raw
    safe_stem = sanitize_slug(stem, stem)
    target = (get_knowledge_base_dir() / f"{safe_stem}.md").resolve()
    if not target.exists():
        available = list_markdown_files()
        return build_guided_error(
            error_code="FILE_NOT_FOUND",
            message=f"File '{safe_stem}.md' does not exist in knowledge_base/.",
            recovery_instructions=(
                f"Existing files are: {available}. Verify the filename from `list_markdown_files` "
                "and retry with an existing filename."
            ),
            suggested_fix={"available_files": available},
        )

    target.unlink()
    log_structured_event(
        event_type="knowledge_file_deleted",
        message=f"Deleted knowledge file '{target.name}'.",
        filename=target.name,
    )
    return {"status": "deleted", "filename": target.name}


def _requires_overwrite_confirmation(
    title: str = "",
    content: str = "",
    slug: str = "",
    tags: list[str] | None = None,
    **kwargs: Any,
) -> bool:
    """HITL predicate: requests confirmation when a tool call would overwrite an existing file."""
    safe_slug = sanitize_slug(slug, title or "生成AIナレッジ")
    target = get_knowledge_base_dir() / f"{safe_slug}.md"
    return target.exists() and os.environ.get(
        "ENABLE_HITL_CONFIRMATION", ""
    ).lower() in (
        "true",
        "1",
    )


# ADK FunctionTool wrappers with Human-in-the-Loop confirmation hooks
save_knowledge_tool = FunctionTool(
    save_markdown_knowledge,
    require_confirmation=_requires_overwrite_confirmation,
)
search_knowledge_tool = FunctionTool(search_markdown_knowledge_guided)
list_files_tool = FunctionTool(list_markdown_files)
delete_knowledge_tool = FunctionTool(
    delete_markdown_knowledge,
    require_confirmation=True,
)
