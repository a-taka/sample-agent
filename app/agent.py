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
import time
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai

load_dotenv()
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.memory import InMemoryMemoryService
from google.adk.models import Gemini
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.workflow import DEFAULT_ROUTE, RetryConfig, Workflow
from google.genai import types
from pydantic import BaseModel, Field

from app.app_utils.observability import (
    PolicyAndObservabilityPlugin,
    log_intent_vs_outcome,
    log_structured_event,
    redact_pii,
)
from app.tools import (
    delete_knowledge_tool,
    list_files_tool,
    list_markdown_files,
    list_markdown_files_async,
    save_knowledge_tool,
    save_markdown_knowledge,
    save_markdown_knowledge_async,
    schedule_background_knowledge_save,
    search_knowledge_tool,
    search_markdown_knowledge,
    search_markdown_knowledge_async,
    search_markdown_knowledge_guided,
)

# ---------------------------------------------------------------------------
# Strategic Model Routing Constants
# ---------------------------------------------------------------------------
# 1. LITE_MODEL: Low-latency model for fast classification, guardrail checks & compaction
LITE_MODEL = "gemini-3.5-flash-lite"
# 2. MODEL (FAST_MODEL): Default balanced model for standard RAG synthesis & tool execution
MODEL = "gemini-3.8-flash"
FAST_MODEL = MODEL
# 3. PRO_MODEL: High-reasoning model for complex architectural / multi-document synthesis
PRO_MODEL = "gemini-3.1-pro-preview"

RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5,
    initial_delay=1.0,
    max_delay=16.0,
    exp_base=2.0,
    jitter=0.5,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)

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

_COMPLEX_QUERY_MARKERS = (
    "比較",
    "アーキテクチャ",
    "設計",
    "トレードオフ",
    "違い",
    "組み合わせ",
    "最適化",
    "architecture",
    "trade-off",
    "compare",
    "design",
)

_HITL_CONFIRM_MARKERS = (
    "確認してから保存",
    "上書き確認",
    "confirm before saving",
    "ask before saving",
)

# Process-wide async memory service for background indexing
_MEMORY_SERVICE = InMemoryMemoryService()


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
    complexity: Literal["simple", "complex"] = Field(
        default="simple",
        description=(
            "Set to 'complex' when the user asks a multi-faceted architectural, "
            "comparative, or deep design question requiring advanced reasoning "
            "(routed to Pro model). Set to 'simple' for standard lookups or notes "
            "(routed to Flash model)."
        ),
    )
    require_hitl_confirmation: bool = Field(
        default=False,
        description=(
            "True ONLY if the user explicitly asks to confirm before saving/overwriting "
            "or if the request is destructive."
        ),
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
            "Generative AI knowledge when intent is 'save_knowledge' (do not repeat H1 title)."
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
    """Deterministic emergency classifier used only after guided LLM retries are exhausted."""
    lower = text.lower()
    is_genai = any(kw in lower for kw in _GENAI_KEYWORDS)
    if not is_genai:
        return KnowledgeClassification(
            is_genai_related=False,
            intent="non_genai",
            complexity="simple",
        )

    has_question = any(q in lower for q in _QUESTION_MARKERS)
    has_save = any(s in lower for s in _SAVE_MARKERS)
    is_complex = any(c in lower for c in _COMPLEX_QUERY_MARKERS) or len(text) > 120
    needs_hitl = any(h in lower for h in _HITL_CONFIRM_MARKERS)

    if has_question and not ("保存して" in lower or "記録して" in lower):
        tokens = re.findall(r"[a-zA-Z0-9_\-\.]{2,}|[ァ-ヶー]{2,}|[一-龥]{2,}", text)
        return KnowledgeClassification(
            is_genai_related=True,
            intent="query_knowledge",
            complexity="complex" if is_complex else "simple",
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
            complexity="simple",
            require_hitl_confirmation=needs_hitl,
            title=title,
            slug=slug,
            tags=["Generative AI"],
            markdown_content=text.strip(),
        )

    return KnowledgeClassification(
        is_genai_related=True,
        intent="query_knowledge",
        complexity="complex" if is_complex else "simple",
        search_keywords=[],
    )


_GENAI_CLIENT: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    """Returns a reusable genai.Client with HTTP timeout and 503 retry options configured."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        _GENAI_CLIENT = genai.Client(
            http_options=types.HttpOptions(
                timeout=20000,
                retry_options=RETRY_OPTIONS,
            ),
        )
    return _GENAI_CLIENT


# Dedicated Classifier Agent definition (uses LITE_MODEL for strategic routing)
classifier_agent = LlmAgent(
    name="classifier_agent",
    model=Gemini(
        model=LITE_MODEL,
        retry_options=RETRY_OPTIONS,
    ),
    description=(
        "Fast front-door classifier and guardrail agent using Gemini Flash-Lite "
        "to classify domain relevance, intent, and task complexity."
    ),
    instruction=(
        "Classify incoming user messages for Generative AI relevance, intent "
        "(save_knowledge, query_knowledge, non_genai), and query complexity (simple vs complex)."
    ),
    output_schema=KnowledgeClassification,
    output_key="last_classification",
)


async def _classify_with_guided_recovery(
    base_prompt: str,
    user_text: str,
) -> tuple[KnowledgeClassification, str]:
    """Executes classification with strategic model fallback (LITE_MODEL -> FAST_MODEL)
    and guided error-recovery feedback if schema validation or fields are incomplete.
    """
    models_to_try = [LITE_MODEL, FAST_MODEL]
    recovery_feedback = ""

    for attempt_idx, candidate_model in enumerate(models_to_try):
        full_prompt = (
            base_prompt
            if not recovery_feedback
            else f"{base_prompt}\n\n{recovery_feedback}"
        )
        try:
            client = _get_genai_client()
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=candidate_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=KnowledgeClassification,
                    ),
                ),
                timeout=18.0,
            )
            parsed = response.parsed
            if isinstance(parsed, KnowledgeClassification):
                # Validate completeness for save_knowledge; if missing content, issue guided retry
                if (
                    parsed.is_genai_related
                    and parsed.intent == "save_knowledge"
                    and not parsed.markdown_content.strip()
                    and attempt_idx == 0
                ):
                    recovery_feedback = (
                        "RECOVERY INSTRUCTION: Your previous classification set "
                        "`intent='save_knowledge'` but left `markdown_content` empty. "
                        "Populate `title`, `slug` (ASCII hyphenated), `tags`, and "
                        "`markdown_content` with a well-structured Japanese Markdown summary."
                    )
                    continue
                return parsed, candidate_model

            recovery_feedback = (
                "RECOVERY INSTRUCTION: Previous response did not match KnowledgeClassification "
                "JSON schema. Return valid JSON with `is_genai_related`, `intent`, and `complexity`."
            )
        except Exception as exc:
            log_structured_event(
                event_type="classifier_model_fallback",
                message=f"Classifier model '{candidate_model}' failed ({exc}); routing to fallback.",
                severity="WARNING",
                failed_model=candidate_model,
                attempt=attempt_idx + 1,
            )
            recovery_feedback = (
                f"RECOVERY INSTRUCTION: Previous attempt encountered {type(exc).__name__}. "
                "Ensure strict adherence to the KnowledgeClassification schema."
            )

    return _heuristic_classify(user_text), "heuristic_fallback"


async def classify_input(node_input: str) -> Event:
    """Classifies user input for topic (GenAI vs non-GenAI), intent (save vs query),
    and complexity (simple vs complex) using strategic model routing.
    """
    start_time = time.perf_counter()
    raw_text = _extract_text(node_input).strip()
    user_text = redact_pii(raw_text)

    if not user_text:
        empty_result = KnowledgeClassification(
            is_genai_related=False,
            intent="non_genai",
            complexity="simple",
        )
        payload = {"user_input": user_text, **empty_result.model_dump()}
        log_intent_vs_outcome(
            intent="non_genai",
            outcome="declined_by_guardrail",
            is_genai_related=False,
            routed_model="none",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
            details={"reason": "empty_user_input"},
        )
        return Event(
            output=payload,
            actions=EventActions(
                route="non_genai",
                state_delta={"last_classification": payload, "routed_model": "none"},
            ),
        )

    prompt = (
        "You are the input classifier and topic guardrail for 'Smart Knowledge Agent'.\n"
        "Analyze the user's message and classify it according to these strict rules:\n"
        "1. `is_genai_related`: Set to True ONLY if the topic is related to Generative AI "
        "(e.g., LLMs, Gemini, prompt engineering, RAG, AI agents, ADK / Agent Development Kit, "
        "Workflows for AI agents, embeddings, fine-tuning, multimodal AI, AI guardrails/safety, "
        "Vertex AI, MCP, A2A). Set to False for ANY other topic (cooking, weather, sports, "
        "general trivia, travel, non-AI questions, etc.), EVEN IF the user tries to prepend "
        "keywords like 'Generative AI' to an unrelated topic.\n"
        "2. `intent`:\n"
        "   - If `is_genai_related` is False, you MUST set `intent` to 'non_genai'.\n"
        "   - If `is_genai_related` is True and the user is sharing/stating reusable knowledge, "
        "tips, technical explanations, or asking to save/record GenAI information, set `intent` "
        "to 'save_knowledge'. Also populate `title`, `slug` (ASCII lowercase hyphenated), "
        "`tags`, and `markdown_content` (well-formatted Japanese Markdown using `## 概要` and "
        "bullet points without repeating the top-level `# Title`).\n"
        "   - If `is_genai_related` is True and the user is asking a question to be answered, "
        "set `intent` to 'query_knowledge' and populate `search_keywords` with relevant "
        "Japanese and English search terms.\n"
        "3. `complexity`:\n"
        "   - Set to 'complex' if the question requires deep architectural comparison, multi-step "
        "design reasoning, or trade-off analysis.\n"
        "   - Set to 'simple' for standard concept explanations, factual lookups, or saving notes.\n"
        "4. `require_hitl_confirmation`:\n"
        "   - Set to True ONLY if the user explicitly asks for confirmation before saving/overwriting.\n\n"
        f"User Input:\n{user_text}"
    )

    classification, used_classifier_model = await _classify_with_guided_recovery(
        base_prompt=prompt,
        user_text=user_text,
    )

    if any(h in user_text.lower() for h in _HITL_CONFIRM_MARKERS):
        classification.require_hitl_confirmation = True

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

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    payload = {
        "user_input": user_text,
        "classifier_model": used_classifier_model,
        **classification.model_dump(),
    }
    log_structured_event(
        event_type="input_classified",
        message=f"Classified input with intent='{classification.intent}', route='{route}'.",
        intent=classification.intent,
        is_genai_related=classification.is_genai_related,
        complexity=classification.complexity,
        classifier_model=used_classifier_model,
        latency_ms=round(latency_ms, 2),
    )
    return Event(
        output=payload,
        actions=EventActions(
            route=route,
            state_delta={
                "last_classification": payload,
                "routed_model": used_classifier_model,
            },
        ),
    )


async def confirm_save_if_needed(ctx: Context, node_input: dict[str, Any]) -> Any:
    """Human-in-the-Loop (HITL) gate node before saving or overwriting knowledge files.
    Emits `RequestInput` when explicit human confirmation is required and resumes via
    `ctx.resume_inputs`.
    """
    if not isinstance(node_input, dict) or not node_input.get("is_genai_related"):
        yield Event(output=node_input, actions=EventActions(route="cancel"))
        return

    requires_hitl = bool(node_input.get("require_hitl_confirmation", False))
    if requires_hitl:
        resume_inputs = ctx.resume_inputs or {}
        if "confirm_knowledge_save" not in resume_inputs:
            title = str(node_input.get("title") or "生成AIナレッジ")
            slug = str(node_input.get("slug") or "genai-note")
            log_intent_vs_outcome(
                intent="save_knowledge",
                outcome="hitl_confirmation_requested",
                is_genai_related=True,
                routed_model=LITE_MODEL,
                details={"title": title, "slug": slug},
            )
            yield RequestInput(
                interrupt_id="confirm_knowledge_save",
                message=(
                    f"【保存確認 (Human-in-the-Loop)】\n"
                    f"タイトル「{title}」（ファイル名: `{slug}.md`）としてナレッジベースに保存してよろしいですか？\n"
                    f"承認する場合は `yes`、キャンセルする場合は `no` と入力してください。"
                ),
            )
            return

        user_decision = (
            str(resume_inputs.get("confirm_knowledge_save", "")).strip().lower()
        )
        if user_decision in {"no", "cancel", "false", "いいえ", "キャンセル"}:
            yield Event(
                output=node_input,
                actions=EventActions(route="cancel"),
            )
            return

    yield Event(output=node_input, actions=EventActions(route="proceed"))


def handle_hitl_cancel(node_input: Any) -> Event:
    """Handles user-cancelled Human-in-the-Loop knowledge save operations."""
    cancel_msg = "ユーザーの確認（Human-in-the-Loop）により、ナレッジファイルへの保存をキャンセルしました。"
    log_intent_vs_outcome(
        intent="save_knowledge",
        outcome="hitl_cancelled_by_user",
        is_genai_related=True,
        routed_model=LITE_MODEL,
    )
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=cancel_msg)],
        ),
        output=cancel_msg,
    )


def save_knowledge_markdown(node_input: dict[str, Any]) -> Event:
    """Saves validated Generative AI knowledge as a Markdown file and schedules
    asynchronous background persistence and memory indexing.
    """
    start_time = time.perf_counter()
    if not isinstance(node_input, dict) or not node_input.get("is_genai_related"):
        return decline_non_genai(node_input)

    user_input = redact_pii(str(node_input.get("user_input", "")).strip())
    title = redact_pii(str(node_input.get("title") or "生成AIナレッジ").strip())
    slug = str(node_input.get("slug") or "").strip()
    tags = node_input.get("tags") or ["Generative AI"]
    content = redact_pii(str(node_input.get("markdown_content") or user_input).strip())

    # If an event loop is running, also schedule async background verification/indexing
    saved = save_markdown_knowledge(
        title=title,
        content=content,
        slug=slug,
        tags=tags if isinstance(tags, list) else ["Generative AI"],
    )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running() and saved.get("status") == "saved":
            schedule_background_knowledge_save(
                title=title,
                content=content,
                slug=slug,
                tags=tags if isinstance(tags, list) else ["Generative AI"],
            )
    except RuntimeError:
        pass

    if saved.get("status") == "error":
        # Guided recovery fallback if primary save reported a structured error
        fallback_fix = saved.get("suggested_fix") or {}
        saved = save_markdown_knowledge(
            title=fallback_fix.get("title", title or "生成AIナレッジ"),
            content=fallback_fix.get(
                "content", content or user_input or "生成AIに関する知見"
            ),
            slug=fallback_fix.get("slug", ""),
            tags=tags if isinstance(tags, list) else ["Generative AI"],
        )

    tags_str = ", ".join(saved.get("tags", ["Generative AI"]))
    message = (
        f"生成AIに関するナレッジを Markdown ファイルとして保存しました。\n\n"
        f"- **ファイル名**: `knowledge_base/{saved['filename']}`\n"
        f"- **タイトル**: {saved['title']}\n"
        f"- **タグ**: {tags_str}\n\n"
        f"### 保存された内容\n"
        f"{saved['markdown'].strip()}"
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    log_intent_vs_outcome(
        intent="save_knowledge",
        outcome="knowledge_saved_async",
        is_genai_related=True,
        routed_model=str(node_input.get("classifier_model") or LITE_MODEL),
        latency_ms=latency_ms,
        details={"filename": saved["filename"], "tags": saved.get("tags", [])},
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


async def retrieve_markdown_files(node_input: dict[str, Any]) -> Event:
    """Asynchronously searches local Markdown files in knowledge_base/, enforces
    context budget limits, and routes to the appropriate RAG model based on complexity.
    """
    start_time = time.perf_counter()
    if not isinstance(node_input, dict):
        user_query = redact_pii(_extract_text(node_input))
        keywords: list[str] = []
        complexity: Literal["simple", "complex"] = "simple"
    else:
        user_query = redact_pii(str(node_input.get("user_input", "")).strip())
        raw_kw = node_input.get("search_keywords") or []
        keywords = [str(k) for k in raw_kw] if isinstance(raw_kw, list) else []
        raw_comp = str(node_input.get("complexity", "simple"))
        complexity = "complex" if raw_comp == "complex" else "simple"

    matched_docs = await search_markdown_knowledge_async(
        query=user_query,
        keywords=keywords,
        top_k=5,
    )
    all_files = await list_markdown_files_async()

    if matched_docs:
        docs_text_parts = []
        for doc in matched_docs:
            docs_text_parts.append(
                f"--- [File: knowledge_base/{doc['filename']}] ---\n"
                f"{doc['content'].strip()}\n"
            )
        retrieved_context = "\n".join(docs_text_parts)
        outcome_label = "rag_context_retrieved"
    else:
        available_list = ", ".join(f"`knowledge_base/{f}`" for f in all_files) or "なし"
        retrieved_context = (
            "（該当する記載があるローカル Markdown ナレッジファイルは見つかりませんでした。\n"
            f"現在保存されているファイル一覧: {available_list}）"
        )
        outcome_label = "rag_no_matches_fallback"

    target_route = "complex_query" if complexity == "complex" else "simple_query"
    target_model = PRO_MODEL if complexity == "complex" else FAST_MODEL

    rag_prompt = (
        f"【ユーザーの質問（生成AI関連）】\n{user_query}\n\n"
        f"【検索されたローカル Markdown ナレッジファイル（全 {len(all_files)} 件中 {len(matched_docs)} 件一致）】\n"
        f"{retrieved_context}\n\n"
        "上記の Markdown ナレッジファイルの内容を優先的に活用（RAG）して、"
        "ユーザーの質問に日本語で分かりやすく回答してください。\n"
        "参照した Markdown ファイルがある場合は、ファイル名（例: `knowledge_base/xxx.md`）も明記してください。\n"
        "一致件数が 0 件の場合は、「現在のナレッジベース（Markdownファイル）には該当する記載がありませんが、"
        "一般的な生成AIの知識として回答します」と明記した上で回答してください。"
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    log_intent_vs_outcome(
        intent="query_knowledge",
        outcome=outcome_label,
        is_genai_related=True,
        routed_model=target_model,
        complexity=complexity,
        latency_ms=latency_ms,
        details={
            "matched_files": [str(d["filename"]) for d in matched_docs],
            "total_files": len(all_files),
            "target_route": target_route,
        },
    )

    return Event(
        output=rag_prompt,
        actions=EventActions(
            route=target_route,
            state_delta={
                "retrieved_files": [str(d["filename"]) for d in matched_docs],
                "routed_model": target_model,
                "query_complexity": complexity,
            },
        ),
    )


_RAG_INSTRUCTION = (
    "あなたは生成AI（Generative AI）のナレッジ活用アシスタント『Smart Knowledge Agent』です。\n"
    "前段の検索ノードから渡されたローカル Markdown ファイル（`knowledge_base/*.md`）の検索結果をもとに、"
    "ユーザーの生成AIに関する質問へ正確かつ分かりやすく日本語で回答してください。\n\n"
    "回答時のルール:\n"
    "1. 検索された Markdown ファイルに関連情報が含まれている場合は、その内容を根拠として回答し、"
    "参照元ファイル名（例: `knowledge_base/adk-2-0-graph-workflow.md`）を明記してください。\n"
    "2. 検索された Markdown ファイルに直接の記載がない生成AIの質問については、"
    "「現在のナレッジベース（Markdownファイル）には該当する記載がありませんが、一般的な生成AIの知識として回答します」"
    "と一言添えた上で簡潔に回答してください。\n"
    "3. ツール実行結果に `recovery_instructions` が含まれる場合は、そのリカバリ手順に従ってパラメータを補正して再試行してください。\n"
    "4. 生成AIと無関係な質問には絶対に回答しないでください。"
)

# Fast RAG Agent (Gemini 3.8 Flash) for standard queries
rag_answer_agent = LlmAgent(
    name="rag_answer_agent",
    model=Gemini(
        model=FAST_MODEL,
        retry_options=RETRY_OPTIONS,
    ),
    description=(
        "Answers standard Generative AI questions rapidly using Gemini Flash and "
        "retrieved Markdown files from the local knowledge_base/ directory."
    ),
    instruction=_RAG_INSTRUCTION,
    tools=[search_knowledge_tool, list_files_tool, save_knowledge_tool],
)

# Deep Reasoning RAG Agent (Gemini 3.1 Pro) for complex architectural/comparative queries
deep_rag_agent = LlmAgent(
    name="deep_rag_agent",
    model=Gemini(
        model=PRO_MODEL,
        retry_options=RETRY_OPTIONS,
    ),
    description=(
        "Answers complex multi-concept, comparative, or architectural Generative AI "
        "questions using Gemini Pro with deep reasoning over retrieved Markdown files."
    ),
    instruction=(
        _RAG_INSTRUCTION
        + "\n5. 複雑な設計・比較・アーキテクチャに関する質問のため、構成要素・仕組み・利点・トレードオフを構造化して丁寧に解説してください。"
    ),
    tools=[
        search_knowledge_tool,
        list_files_tool,
        save_knowledge_tool,
        delete_knowledge_tool,
    ],
)


def decline_non_genai(node_input: Any) -> Event:
    """Guardrail node that refuses non-Generative-AI topics without saving any file."""
    log_intent_vs_outcome(
        intent="non_genai",
        outcome="declined_by_guardrail",
        is_genai_related=False,
        routed_model=LITE_MODEL,
        details={"input_preview": redact_pii(_extract_text(node_input)[:80])},
    )
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
        "Uses strategic model routing (Flash-Lite classifier, Flash RAG agent, "
        "and Pro deep-reasoning RAG agent), Human-in-the-Loop confirmation hooks, "
        "async background Markdown persistence, context compaction, PII redaction, "
        "and structured Intent-vs-Outcome JSON telemetry."
    ),
    retry_config=RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        max_delay=10.0,
        backoff_factor=2.0,
    ),
    edges=[
        ("START", classify_input),
        (
            classify_input,
            {
                "save_knowledge": confirm_save_if_needed,
                "query_knowledge": retrieve_markdown_files,
                "non_genai": decline_non_genai,
                DEFAULT_ROUTE: decline_non_genai,
            },
        ),
        (
            confirm_save_if_needed,
            {
                "proceed": save_knowledge_markdown,
                "cancel": handle_hitl_cancel,
                DEFAULT_ROUTE: save_knowledge_markdown,
            },
        ),
        (
            retrieve_markdown_files,
            {
                "simple_query": rag_answer_agent,
                "complex_query": deep_rag_agent,
                DEFAULT_ROUTE: rag_answer_agent,
            },
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        token_threshold=16000,
        event_retention_size=5,
        overlap_size=2,
        compaction_interval=5,
        summarizer=LlmEventSummarizer(
            llm=Gemini(
                model=LITE_MODEL,
                retry_options=RETRY_OPTIONS,
            )
        ),
    ),
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=1800,
        cache_intervals=10,
    ),
    plugins=[
        PolicyAndObservabilityPlugin(),
        ContextFilterPlugin(num_invocations_to_keep=10),
        ReflectAndRetryToolPlugin(
            max_retries=3,
            throw_exception_if_retry_exceeded=False,
        ),
    ],
)
