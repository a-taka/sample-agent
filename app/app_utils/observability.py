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

"""Structured JSON logging, PII redaction, Intent-vs-Outcome telemetry, and ADK Policy Plugin."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import trace

# ---------------------------------------------------------------------------
# 1. PII Redaction Engine
# ---------------------------------------------------------------------------

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "SECRET_KEY",
        re.compile(
            r"\b(?:AIza[0-9A-Za-z\-_]{35}|ya29\.[0-9A-Za-z\-_]+|sk-[0-9A-Za-z]{20,}|Bearer\s+[0-9A-Za-z\-._~+/]+=*)\b"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "[REDACTED_CREDIT_CARD]",
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?\(?\d{2,4}\)?[-\s]\d{2,4}[-\s]\d{3,4}(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        "SSN_OR_ID",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}-\d{4}-\d{4}\b"),
        "[REDACTED_ID_NUMBER]",
    ),
)


def redact_pii_with_report(text: str) -> tuple[str, list[str]]:
    """Redacts sensitive PII from text and returns (sanitized_text, detected_pii_types)."""
    if not text or not isinstance(text, str):
        return text, []

    sanitized = text
    detected: list[str] = []
    for pii_type, pattern, replacement in _PII_PATTERNS:
        if pattern.search(sanitized):
            detected.append(pii_type)
            sanitized = pattern.sub(replacement, sanitized)
    return sanitized, detected


def redact_pii(text: str) -> str:
    """Redacts sensitive PII patterns (email, phone, credit card, secrets, IDs) from text."""
    sanitized, _ = redact_pii_with_report(text)
    return sanitized


def redact_pii_in_obj(value: Any) -> Any:
    """Recursively redacts PII across strings, dicts, lists, and tuples."""
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {str(k): redact_pii_in_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_pii_in_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_pii_in_obj(item) for item in value)
    return value


def redact_pii_in_content(
    content: types.Content | None,
) -> tuple[types.Content | None, list[str]]:
    """Returns a copy of types.Content with PII redacted from all text parts."""
    if content is None or not content.parts:
        return content, []

    detected_all: list[str] = []
    new_parts: list[types.Part] = []
    modified = False

    for part in content.parts:
        if part.text:
            clean_text, detected = redact_pii_with_report(part.text)
            if detected:
                modified = True
                detected_all.extend(detected)
                new_parts.append(types.Part.from_text(text=clean_text))
            else:
                new_parts.append(part)
        else:
            new_parts.append(part)

    if not modified:
        return content, []

    unique_detected = list(dict.fromkeys(detected_all))
    return types.Content(role=content.role, parts=new_parts), unique_detected


# ---------------------------------------------------------------------------
# 2. Structured JSON Logging & OpenTelemetry Trace Correlation
# ---------------------------------------------------------------------------


def _get_trace_context() -> dict[str, str]:
    """Extracts current OpenTelemetry trace_id and span_id if active."""
    span = trace.get_current_span()
    if span is None:
        return {}
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }


class StructuredJsonFormatter(logging.Formatter):
    """Formats Python log records as single-line JSON objects with PII redaction."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
        }
        trace_ctx = _get_trace_context()
        if trace_ctx:
            payload.update(trace_ctx)

        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict):
            payload.update(redact_pii_in_obj(structured_data))

        if record.exc_info:
            payload["exception"] = redact_pii(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_structured_logger(name: str = "smart_knowledge_agent") -> logging.Logger:
    """Returns a singleton structured JSON logger configured for stdout."""
    logger = logging.getLogger(name)
    if not getattr(logger, "_is_structured_configured", False):
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.handlers = [handler]
        logger.propagate = False
        logger._is_structured_configured = True  # type: ignore[attr-defined]
    return logger


def log_structured_event(
    event_type: str,
    message: str,
    *,
    severity: str = "INFO",
    **fields: Any,
) -> dict[str, Any]:
    """Emits a structured JSON log entry with automatic PII redaction and OTel correlation."""
    logger = get_structured_logger()
    sanitized_fields = redact_pii_in_obj(fields)
    structured_payload = {
        "event_type": event_type,
        **sanitized_fields,
    }
    level = getattr(logging, severity.upper(), logging.INFO)
    logger.log(
        level,
        redact_pii(message),
        extra={"structured_data": structured_payload},
    )
    return structured_payload


def log_intent_vs_outcome(
    *,
    intent: str,
    outcome: str,
    is_genai_related: bool,
    routed_model: str = "",
    complexity: str = "simple",
    latency_ms: float | None = None,
    session_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicitly logs classified user intent vs. execution outcome to JSON logs and OTel spans."""
    expected_outcomes_by_intent: dict[str, set[str]] = {
        "save_knowledge": {
            "knowledge_saved",
            "knowledge_saved_async",
            "hitl_confirmation_requested",
            "hitl_cancelled_by_user",
        },
        "query_knowledge": {
            "rag_context_retrieved",
            "rag_answered_fast",
            "rag_answered_deep",
            "rag_no_matches_fallback",
        },
        "non_genai": {
            "declined_by_guardrail",
        },
    }
    aligned = outcome in expected_outcomes_by_intent.get(intent, set())

    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute("agent.intent", intent)
        span.set_attribute("agent.outcome", outcome)
        span.set_attribute("agent.intent_outcome_aligned", aligned)
        span.set_attribute("agent.is_genai_related", is_genai_related)
        if routed_model:
            span.set_attribute("agent.routed_model", routed_model)
        if complexity:
            span.set_attribute("agent.complexity", complexity)

    record: dict[str, Any] = {
        "intent": intent,
        "outcome": outcome,
        "intent_outcome_aligned": aligned,
        "is_genai_related": is_genai_related,
        "routed_model": routed_model,
        "complexity": complexity,
    }
    if latency_ms is not None:
        record["latency_ms"] = round(latency_ms, 2)
    if session_id:
        record["session_id"] = session_id
    if details:
        record["details"] = redact_pii_in_obj(details)

    return log_structured_event(
        event_type="intent_vs_outcome",
        message=f"Intent '{intent}' resulted in outcome '{outcome}' (aligned={aligned})",
        **record,
    )


# ---------------------------------------------------------------------------
# 3. ADK Policy, Context Guardrail, and Observability Plugin
# ---------------------------------------------------------------------------


class PolicyAndObservabilityPlugin(BasePlugin):
    """Cross-cutting ADK plugin enforcing PII redaction, context bloat limits,
    guided tool error recovery, and structured JSON telemetry.
    """

    def __init__(
        self,
        name: str = "policy_and_observability_plugin",
        max_message_chars: int = 12000,
        max_history_contents: int = 16,
    ) -> None:
        super().__init__(name=name)
        self.max_message_chars = max_message_chars
        self.max_history_contents = max_history_contents
        self._agent_start_times: dict[str, float] = {}

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Sanitizes PII from incoming user messages before agent processing."""
        redacted_content, pii_types = redact_pii_in_content(user_message)
        session_id = getattr(invocation_context.session, "id", None)
        log_structured_event(
            event_type="user_message_received",
            message="Received user message at workflow entry.",
            session_id=session_id,
            pii_redacted=bool(pii_types),
            pii_types_detected=pii_types,
        )
        if pii_types:
            return redacted_content
        return None

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Records start time and initializes session state safely."""
        self._agent_start_times[agent.name] = time.perf_counter()
        log_structured_event(
            event_type="agent_execution_started",
            message=f"Agent '{agent.name}' started.",
            agent_name=agent.name,
        )
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Logs agent execution latency and records intent vs. outcome summary."""
        started = self._agent_start_times.pop(agent.name, None)
        latency_ms = (time.perf_counter() - started) * 1000.0 if started else None
        state = callback_context.state
        last_cls = state.get("last_classification") if state else None
        routed_model = state.get("routed_model", "") if state else ""

        log_structured_event(
            event_type="agent_execution_completed",
            message=f"Agent '{agent.name}' completed.",
            agent_name=agent.name,
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
            routed_model=routed_model,
            classification_intent=last_cls.get("intent")
            if isinstance(last_cls, dict)
            else None,
        )
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Enforces PII redaction and prevents context bloat before LLM invocation."""
        if llm_request.contents:
            # 1. Context bloat management: cap history depth
            if len(llm_request.contents) > self.max_history_contents:
                llm_request.contents = [
                    llm_request.contents[0],
                    *llm_request.contents[-(self.max_history_contents - 1) :],
                ]

            # 2. Redact PII and truncate oversized individual parts
            sanitized_contents: list[types.Content] = []
            for content in llm_request.contents:
                clean_content, _ = redact_pii_in_content(content)
                if clean_content and clean_content.parts:
                    trimmed_parts: list[types.Part] = []
                    for part in clean_content.parts:
                        if part.text and len(part.text) > self.max_message_chars:
                            trimmed_text = (
                                part.text[: self.max_message_chars]
                                + "\n...[TRUNCATED_FOR_CONTEXT_BUDGET]..."
                            )
                            trimmed_parts.append(
                                types.Part.from_text(text=trimmed_text)
                            )
                        else:
                            trimmed_parts.append(part)
                    sanitized_contents.append(
                        types.Content(role=clean_content.role, parts=trimmed_parts)
                    )
                elif clean_content:
                    sanitized_contents.append(clean_content)
            llm_request.contents = sanitized_contents

        log_structured_event(
            event_type="llm_request_dispatched",
            message=f"Dispatching LLM request to model '{llm_request.model}'.",
            agent_name=callback_context.agent_name,
            model=str(llm_request.model or ""),
            num_contents=len(llm_request.contents) if llm_request.contents else 0,
        )
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Redacts any PII in model output before returning to user or downstream nodes."""
        if llm_response.content:
            clean_content, pii_types = redact_pii_in_content(llm_response.content)
            if pii_types and clean_content:
                llm_response.content = clean_content

        log_structured_event(
            event_type="llm_response_received",
            message=f"Received LLM response for agent '{callback_context.agent_name}'.",
            agent_name=callback_context.agent_name,
        )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        """Logs structured model error details for 503/rate-limit diagnostics."""
        log_structured_event(
            event_type="llm_error",
            message=f"Model call failed in '{callback_context.agent_name}': {error}",
            severity="ERROR",
            agent_name=callback_context.agent_name,
            model=str(llm_request.model or ""),
            error_type=type(error).__name__,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Redacts PII in tool arguments and logs tool invocation."""
        for key, val in list(tool_args.items()):
            if isinstance(val, str):
                tool_args[key] = redact_pii(val)
        log_structured_event(
            event_type="tool_invocation_started",
            message=f"Invoking tool '{tool.name}'.",
            tool_name=tool.name,
            tool_args=tool_args,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Logs tool execution outcome (success, warning, or guided error)."""
        status = (
            result.get("status", "success") if isinstance(result, dict) else "success"
        )
        log_structured_event(
            event_type="tool_invocation_completed",
            message=f"Tool '{tool.name}' completed with status '{status}'.",
            tool_name=tool.name,
            status=status,
            error_code=result.get("error_code") if isinstance(result, dict) else None,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Converts unhandled tool exceptions into descriptive recovery instructions for the LLM."""
        log_structured_event(
            event_type="tool_execution_error",
            message=f"Tool '{tool.name}' raised {type(error).__name__}: {error}",
            severity="WARNING",
            tool_name=tool.name,
            error_type=type(error).__name__,
        )
        return {
            "status": "error",
            "error_code": f"TOOL_{type(error).__name__.upper()}",
            "message": f"Tool '{tool.name}' encountered an error: {error}",
            "recovery_instructions": (
                f"Inspect the arguments passed to `{tool.name}` ({list(tool_args.keys())}). "
                "Ensure all required string parameters are non-empty, slugs contain only "
                "ASCII alphanumeric characters and hyphens, and retry the tool call. "
                "If searching failed, call `list_markdown_files` first to inspect available files."
            ),
            "provided_arguments": redact_pii_in_obj(tool_args),
        }
