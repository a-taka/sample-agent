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
"""Unit tests for Smart Knowledge Agent tools and workflow structure."""

from pathlib import Path
from pytest import MonkeyPatch

from google.adk.workflow import Workflow

from app.agent import (
    DECLINE_MESSAGE,
    decline_non_genai,
    root_agent,
    save_knowledge_markdown,
)
from app.tools import (
    list_markdown_files,
    sanitize_slug,
    save_markdown_knowledge,
    search_markdown_knowledge,
)


def test_workflow_structure() -> None:
    """Verifies that root_agent is an ADK 2.0 Workflow with expected name."""
    assert isinstance(root_agent, Workflow)
    assert root_agent.name == "smart_knowledge_agent"


def test_markdown_save_and_search(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
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
