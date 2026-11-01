# ruff: noqa
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

import asyncio
import re
from typing import Any, Literal

from google import genai
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models import Gemini
from google.adk.workflow import DEFAULT_ROUTE, Workflow
from google.genai import types
from pydantic import BaseModel, Field

from app.tools import (
    list_markdown_files,
    save_markdown_knowledge,
    search_markdown_knowledge,
)

MODEL = "gemini-3.8-flash"

DECLINE_MESSAGE = (
    "申し訳ありませんが、私は**生成AI（Generative AI）**に関するナレッジの蓄積と"
    "質問回答に特化したエージェント（Smart Knowledge Agent）です。\n"
    "生成AI以外のテーマについては、ナレッジファイル（Markdown）への保存および"
    "質問への回答を行っておりません。生成AI・LLM・RAG・AIエージェント（ADK等）に"
    "関するナレッジ共有やご質問がありましたらお気軽にお知らせください。"
)

_GENAI_KEYWORDS = (
    "生成ai",
    "generative ai",
    "genai",
    "llm",
    "大規模言語モデル",
    "gemini",
    "adk",
    "agent development kit",
    "workflow",
    "ワークフロー",
    "rag",
    "検索拡張生成",
    "プロンプト",
    "prompt",
    "エージェント",
    "ai agent",
    "ファインチューニング",
    "fine-tuning",
    "embedding",
    "埋め込み",
    "ハルシネーション",
    "hallucination",
    "vertex ai",
    "マルチモーダル",
    "multimodal",
    "function calling",
    "ガードレール",
    "guardrail",
    "mcp",
    "a2a",
)

_QUESTION_MARKERS = (
    "?",
    "？",
    "とは",
    "教えて",
    "どうすれば",
    "どうやって",
    "でしょうか",
    "ますか",
    "ですか",
    "何ですか",
    "違いは",
    "方法を",
    "what is",
    "how do",
    "how to",
    "can you explain",
    "why ",
)

_SAVE_MARKERS = (
    "ナレッジ",
    "メモ",
    "保存して",
    "記録して",
    "覚えて",
    "共有します",
    "知見",
    "ポイントです",
    "ことができます",
    "有効です",
    "特徴があり",
    "仕組みです",
    "save this",
    "note:",
    "knowledge:",
)


class KnowledgeClassification(BaseModel):
    """Structured classification output for incoming user messages."""

    is_genai_related: bool = Field(
        description=(
            "True ONLY if the user message is about Generative AI, LLMs, Gemini, "
            "prompt engineering, RAG, AI agents, ADK (Agent Development Kit), "
            "embeddings, multimodal AI, or AI safety/guardrails. "
            "False for any other topic (e.g., cooking, weather, travel, sports, "
            "general trivia, or non-AI topics)."
        )
    )
    intent: Literal["save_knowledge", "query_knowledge", "non_genai"] = Field(
        description=(
            "Must be 'non_genai' whenever is_genai_related is False. "
            "When is_genai_related is True: set to 'save_knowledge' if the user "
            "is providing/sharing knowledge, facts, tips, or notes that can be "
            "stored as a reusable Markdown document; set to 'query_knowledge' if "
            "the user is asking a question or searching for information."
        )
    )
    title: str = Field(
        default="",
        description="Concise document title when intent is 'save_knowledge'.",
    )
    slug: str = Field(
        default="",
        description=(
            "Lowercase ASCII hyphenated filename slug (e.g. 'adk-graph-workflow', "
            "'gemini-prompt-tips') when intent is 'save_knowledge'."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of 2-4 Generative AI topic tags (e.g. ['ADK', 'Workflow']).",
    )
    markdown_content: str = Field(
        default="",
        description=(
            "Structured Markdown content summarizing and organizing the user's "
            "Generative AI knowledge when intent is 'save_knowledge'."
        ),
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Important keywords (Japanese and English) for searching stored Markdown "
            "files when intent is 'query_knowledge'."
        ),
    )


def _extract_text(node_input: Any) -> str:
    """Extracts plain text from a string, dict, or Content input."""
    if isinstance(node_input, str):
        return node_input
    if isinstance(node_input, types.Content) and node_input.parts:
        return "".join(part.text for part in node_input.parts if part.text)
    if isinstance(node_input, dict):
        return str(node_input.get("user_input", "") or "")
    return str(node_input or "")


def _heuristic_classify(text: str) -> KnowledgeClassification:
    """Deterministic fallback classifier used if the LLM classification call fails."""
    lower = text.lower()
    is_genai = any(kw in lower for kw in _GENAI_KEYWORDS)
    if not is_genai:
        return KnowledgeClassification(
            is_genai_related=False,
            intent="non_genai",
        )

    has_question = any(q in lower for q in _QUESTION_MARKERS)
    has_save = any(s in lower for s in _SAVE_MARKERS)

    if has_question and not ("保存して" in lower or "記録して" in lower):
        tokens = re.findall(r"[a-zA-Z0-9_\-\.]{2,}|[ァ-ヶー]{2,}|[一-龥]{2,}", text)
        return KnowledgeClassification(
            is_genai_related=True,
            intent="query_knowledge",
            search_keywords=tokens[:6],
        )

    if has_save or not has_question:
        first_line = text.strip().splitlines()[0][:40].strip("。、. ")
        title = first_line or "生成AIナレッジノート"
        ascii_words = re.findall(r"[a-z0-9]+", lower)
        slug = "-".join(ascii_words[:4]) if ascii_words else ""
        return KnowledgeClassification(
            is_genai_related=True,
            intent="save_knowledge",
            title=title,
            slug=slug,
            tags=["Generative AI"],
            markdown_content=text.strip(),
        )

    return KnowledgeClassification(
        is_genai_related=True,
        intent="query_knowledge",
        search_keywords=[],
    )


_GENAI_CLIENT: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    """Returns a reusable genai.Client with HTTP timeout configured."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        _GENAI_CLIENT = genai.Client(
            http_options=types.HttpOptions(timeout=15000),
        )
    return _GENAI_CLIENT


async def classify_input(node_input: str) -> Event:
    """Classifies user input for topic (GenAI vs non-GenAI) and intent (save vs query)."""
    user_text = _extract_text(node_input).strip()
    if not user_text:
        empty_result = KnowledgeClassification(
            is_genai_related=False,
            intent="non_genai",
        )
        payload = {"user_input": user_text, **empty_result.model_dump()}
        return Event(
            output=payload,
            actions=EventActions(
                route="non_genai",
                state_delta={"last_classification": payload},
            ),
        )

    classification: KnowledgeClassification
    try:
        client = _get_genai_client()
        prompt = (
            "You are the input classifier and topic guardrail for 'Smart Knowledge Agent'.\n"
            "Analyze the user's message and classify it according to these strict rules:\n"
            "1. `is_genai_related`: Set to True ONLY if the topic is related to Generative AI "
            "(e.g., LLMs, Gemini, prompt engineering, RAG, AI agents, ADK / Agent Development Kit, "
            "Workflows for AI agents, embeddings, fine-tuning, multimodal AI, AI guardrails/safety, "
            "Vertex AI, MCP, A2A). Set to False for ANY other topic (cooking, weather, sports, "
            "general trivia, travel, non-AI questions, etc.).\n"
            "2. `intent`:\n"
            "   - If `is_genai_related` is False, you MUST set `intent` to 'non_genai' (even if "
            "the user asks to save a note or asks a question).\n"
            "   - If `is_genai_related` is True and the user is sharing/stating reusable knowledge, "
            "tips, technical explanations, or asking to save/record GenAI information, set `intent` "
            "to 'save_knowledge'. Also populate `title`, `slug` (ASCII lowercase hyphenated), "
            "`tags`, and `markdown_content` (well-formatted Japanese Markdown summarizing and "
            "preserving all key details from the user's input).\n"
            "   - If `is_genai_related` is True and the user is asking a question to be answered, "
            "set `intent` to 'query_knowledge' and populate `search_keywords` with relevant "
            "Japanese and English search terms.\n\n"
            f"User Input:\n{user_text}"
        )
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=KnowledgeClassification,
                ),
            ),
            timeout=15.0,
        )
        parsed = response.parsed
        if isinstance(parsed, KnowledgeClassification):
            classification = parsed
        else:
            classification = _heuristic_classify(user_text)
    except Exception:
        classification = _heuristic_classify(user_text)

    # Enforce strict guardrail invariants
    if not classification.is_genai_related or classification.intent == "non_genai":
        classification.is_genai_related = False
        classification.intent = "non_genai"
        route = "non_genai"
    elif classification.intent == "save_knowledge":
        route = "save_knowledge"
    else:
        classification.intent = "query_knowledge"
        route = "query_knowledge"

    payload = {
        "user_input": user_text,
        **classification.model_dump(),
    }
    return Event(
        output=payload,
        actions=EventActions(
            route=route,
            state_delta={"last_classification": payload},
        ),
    )


def save_knowledge_markdown(node_input: dict[str, Any]) -> Event:
    """Saves validated Generative AI knowledge as a Markdown file."""
    if not isinstance(node_input, dict) or not node_input.get("is_genai_related"):
        return decline_non_genai(node_input)

    user_input = str(node_input.get("user_input", "")).strip()
    title = str(node_input.get("title") or "生成AIナレッジ").strip()
    slug = str(node_input.get("slug") or "").strip()
    tags = node_input.get("tags") or ["Generative AI"]
    content = str(node_input.get("markdown_content") or user_input).strip()

    saved = save_markdown_knowledge(
        title=title,
        content=content,
        slug=slug,
        tags=tags if isinstance(tags, list) else ["Generative AI"],
    )

    tags_str = ", ".join(saved["tags"])
    message = (
        f"生成AIに関するナレッジを Markdown ファイルとして保存しました。\n\n"
        f"- **ファイル名**: `knowledge_base/{saved['filename']}`\n"
        f"- **タイトル**: {saved['title']}\n"
        f"- **タグ**: {tags_str}\n\n"
        f"### 保存された内容\n"
        f"{saved['markdown'].strip()}"
    )
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        ),
        output=message,
        actions=EventActions(
            state_delta={"last_saved_file": saved["filename"]},
        ),
    )


def retrieve_markdown_files(node_input: dict[str, Any]) -> Event:
    """Searches local Markdown files in knowledge_base/ and prepares context for RAG."""
    if not isinstance(node_input, dict):
        user_query = _extract_text(node_input)
        keywords: list[str] = []
    else:
        user_query = str(node_input.get("user_input", "")).strip()
        raw_kw = node_input.get("search_keywords") or []
        keywords = [str(k) for k in raw_kw] if isinstance(raw_kw, list) else []

    matched_docs = search_markdown_knowledge(
        query=user_query,
        keywords=keywords,
        top_k=5,
    )
    all_files = list_markdown_files()

    if matched_docs:
        docs_text_parts = []
        for doc in matched_docs:
            docs_text_parts.append(
                f"--- [File: knowledge_base/{doc['filename']}] ---\n"
                f"{doc['content'].strip()}\n"
            )
        retrieved_context = "\n".join(docs_text_parts)
    else:
        retrieved_context = "（保存済みの Markdown ナレッジファイルはまだありません。）"

    rag_prompt = (
        f"【ユーザーの質問（生成AI関連）】\n{user_query}\n\n"
        f"【検索されたローカル Markdown ナレッジファイル（全 {len(all_files)} 件中 {len(matched_docs)} 件取得）】\n"
        f"{retrieved_context}\n\n"
        "上記の Markdown ナレッジファイルの内容を優先的に活用（RAG）して、"
        "ユーザーの質問に日本語で分かりやすく回答してください。"
        "参照した Markdown ファイルがある場合は、ファイル名（例: `knowledge_base/xxx.md`）も明記してください。"
    )
    return Event(
        output=rag_prompt,
        actions=EventActions(
            state_delta={
                "retrieved_files": [str(d["filename"]) for d in matched_docs],
            }
        ),
    )


rag_answer_agent = LlmAgent(
    name="rag_answer_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description=(
        "Answers Generative AI questions using retrieved Markdown files from "
        "the local knowledge_base/ directory."
    ),
    instruction=(
        "あなたは生成AI（Generative AI）のナレッジ活用アシスタント『Smart Knowledge Agent』です。\n"
        "前段の検索ノードから渡されたローカル Markdown ファイル（`knowledge_base/*.md`）の検索結果をもとに、"
        "ユーザーの生成AIに関する質問へ正確かつ分かりやすく日本語で回答してください。\n\n"
        "回答時のルール:\n"
        "1. 検索された Markdown ファイルに関連情報が含まれている場合は、その内容を根拠として回答し、"
        "参照元ファイル名（例: `knowledge_base/adk-2-0-graph-workflow.md`）を明記してください。\n"
        "2. 検索された Markdown ファイルに直接の記載がない生成AIの質問については、"
        "「現在のナレッジベース（Markdownファイル）には該当する記載がありませんが、一般的な生成AIの知識として回答します」"
        "と一言添えた上で簡潔に回答してください。\n"
        "3. 生成AIと無関係な質問には絶対に回答しないでください。"
    ),
)


def decline_non_genai(node_input: Any) -> Event:
    """Guardrail node that refuses non-Generative-AI topics without saving any file."""
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=DECLINE_MESSAGE)],
        ),
        output=DECLINE_MESSAGE,
    )


root_agent = Workflow(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml:
    # telemetry reports this name as gen_ai.agent.name, and agents-cli reports the
    # manifest's value. Renaming the agent only here makes the two disagree, and
    # anything selecting traces by name stops finding this agent's.
    name="smart_knowledge_agent",
    description=(
        "Smart Knowledge Agent built with ADK 2.0 Graph Workflow. "
        "Evaluates whether user input is related to Generative AI, saves "
        "reusable Generative AI knowledge as Markdown files, answers "
        "Generative AI questions via file-based RAG, and strictly blocks "
        "non-Generative-AI topics from saving files or receiving answers."
    ),
    edges=[
        ("START", classify_input),
        (
            classify_input,
            {
                "save_knowledge": save_knowledge_markdown,
                "query_knowledge": retrieve_markdown_files,
                "non_genai": decline_non_genai,
                DEFAULT_ROUTE: decline_non_genai,
            },
        ),
        (retrieve_markdown_files, rag_answer_agent),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
