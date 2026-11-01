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
"""Unit tests for Smart Knowledge Agent tools, guided error handling,
async persistence, PII redaction, structured logging, and workflow architecture.
"""

# ruff: noqa: RUF001
import asyncio
from pathlib import Path

from google.adk.workflow import Workflow
from pytest import MonkeyPatch

from app.agent import (
    DECLINE_MESSAGE,
    FAST_MODEL,
    LITE_MODEL,
    PRO_MODEL,
    app,
    decline_non_genai,
    deep_rag_agent,
    rag_answer_agent,
    root_agent,
    save_knowledge_markdown,
)
from app.app_utils.observability import (
    log_intent_vs_outcome,
    redact_pii_with_report,
)
from app.tools import (
    delete_markdown_knowledge,
    list_markdown_files,
    sanitize_slug,
    save_markdown_knowledge,
    schedule_background_knowledge_save,
    search_markdown_knowledge,
    search_markdown_knowledge_guided,
    wait_for_background_saves,
)


def test_workflow_structure_and_app_configs() -> None:
    """Verifies Workflow, strategic model routing agents, compaction, HITL, and plugins."""
    assert isinstance(root_agent, Workflow)
    assert root_agent.name == "smart_knowledge_agent"
    assert LITE_MODEL != FAST_MODEL or FAST_MODEL != PRO_MODEL
    assert rag_answer_agent.name == "rag_answer_agent"
    assert deep_rag_agent.name == "deep_rag_agent"
    assert app.resumability_config is not None
    assert app.resumability_config.is_resumable is True
    assert app.events_compaction_config is not None
    assert app.context_cache_config is not None
    assert len(app.plugins) >= 3


def test_markdown_save_and_search(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Verifies saving and searching Markdown knowledge files in an isolated directory."""
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))

    result = save_markdown_knowledge(
        title="ADK Graph Workflow Test",
        content="Graph Workflow uses START, FunctionNode, and LlmAgent.",
        slug="adk-graph-test",
        tags=["ADK", "Workflow"],
    )
    assert result["status"] == "saved"
    assert result["filename"] == "adk-graph-test.md"
    assert (tmp_path / "adk-graph-test.md").exists()

    files = list_markdown_files()
    assert files == ["adk-graph-test.md"]

    matches = search_markdown_knowledge(
        query="FunctionNode とは",
        keywords=["ADK", "FunctionNode"],
    )
    assert len(matches) == 1
    assert matches[0]["filename"] == "adk-graph-test.md"
    assert int(matches[0]["score"]) > 0


def test_guided_error_handling_in_tools(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verifies tools return structured recovery_instructions instead of raising exceptions."""
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))

    # 1. Empty content returns guided recovery instructions
    empty_save = save_markdown_knowledge(
        title="Empty Note",
        content="   ",
        slug="empty-note",
    )
    assert empty_save["status"] == "error"
    assert empty_save["error_code"] == "EMPTY_CONTENT"
    assert "recovery_instructions" in empty_save
    assert "save_markdown_knowledge" in empty_save["recovery_instructions"]

    # 2. Empty search query returns guided recovery instructions
    empty_search = search_markdown_knowledge_guided(query="", keywords=[])
    assert empty_search["status"] == "error"
    assert empty_search["error_code"] == "EMPTY_SEARCH_QUERY"
    assert "recovery_instructions" in empty_search

    # 3. Search with no matching documents returns available_files and recovery instructions
    save_markdown_knowledge(
        title="Gemini Caching",
        content="Context caching reduces token cost.",
        slug="gemini-caching",
    )
    no_match = search_markdown_knowledge_guided(
        query="Quantum teleportation hardware",
        keywords=["quantum"],
    )
    assert no_match["status"] == "no_matches"
    assert no_match["error_code"] == "NO_MATCHING_DOCUMENTS"
    assert "gemini-caching.md" in no_match["available_files"]
    assert "recovery_instructions" in no_match

    # 4. Delete missing file returns guided recovery instructions
    missing_del = delete_markdown_knowledge("non-existent-file.md")
    assert missing_del["status"] == "error"
    assert missing_del["error_code"] == "FILE_NOT_FOUND"
    assert "recovery_instructions" in missing_del


def test_async_background_persistence_and_context_budget(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verifies async background file saving and context bloat truncation."""
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))

    async def _run() -> None:
        long_text = "Gemini ADK Workflow Context. " * 200
        task = schedule_background_knowledge_save(
            title="Async Background Note",
            content=long_text,
            slug="async-bg-note",
            tags=["ADK", "Async"],
        )
        await wait_for_background_saves(timeout=5.0)
        res = task.result()
        assert res["status"] == "saved"
        assert (tmp_path / "async-bg-note.md").exists()

        # Verify context bloat truncation
        matches = search_markdown_knowledge(
            query="Gemini ADK Workflow",
            max_chars_per_doc=300,
            max_total_chars=500,
        )
        assert len(matches) == 1
        assert matches[0]["truncated_for_context_budget"] is True
        assert len(matches[0]["content"]) <= 360

    asyncio.run(_run())


def test_pii_redaction_and_intent_outcome_logging(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verifies PII redaction in text/files and structured intent-vs-outcome logging."""
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))

    raw = "連絡先は user@example.com または 090-1234-5678、APIキーは sk-1234567890abcdefghijklmnop です。"
    sanitized, detected = redact_pii_with_report(raw)
    assert "user@example.com" not in sanitized
    assert "090-1234-5678" not in sanitized
    assert "sk-1234567890" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert "EMAIL" in detected

    saved = save_markdown_knowledge(
        title="PII Test user@example.com",
        content="담당者メール: admin@corp.co.jp でGeminiを運用。",
        slug="pii-test",
    )
    assert saved["pii_redacted"] is True
    assert "admin@corp.co.jp" not in saved["markdown"]
    assert "[REDACTED_EMAIL]" in saved["markdown"]

    log_entry = log_intent_vs_outcome(
        intent="query_knowledge",
        outcome="rag_context_retrieved",
        is_genai_related=True,
        routed_model=FAST_MODEL,
        complexity="simple",
        details={"email": "secret@example.com"},
    )
    assert log_entry["intent_outcome_aligned"] is True
    assert log_entry["details"]["email"] == "[REDACTED_EMAIL]"


def test_non_genai_guard_never_writes_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verifies that non-GenAI inputs never create files in knowledge_base/."""
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))

    event = save_knowledge_markdown(
        {
            "user_input": "美味しいパスタの茹で方は塩を1%入れることです。",
            "is_genai_related": False,
            "intent": "non_genai",
            "title": "Pasta Recipe",
            "slug": "pasta-recipe",
            "markdown_content": "Boil with 1% salt.",
        }
    )
    assert list_markdown_files() == []
    assert event.output == DECLINE_MESSAGE

    decline_event = decline_non_genai("今日の天気は？")
    assert list_markdown_files() == []
    assert decline_event.output == DECLINE_MESSAGE


def test_sanitize_slug() -> None:
    """Verifies slug sanitization prevents path traversal and invalid chars."""
    assert sanitize_slug("../../etc/passwd", "Test") == "etc-passwd"
    assert sanitize_slug("Gemini 2.5 & ADK!", "Test") == "gemini-2-5-adk"
    fallback = sanitize_slug("", "日本語タイトルのみ")
    assert fallback.startswith("genai-note-")
