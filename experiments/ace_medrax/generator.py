"""MedRAX agent wrapped as an ACE Generator.

ACE's built-in Generator is a single text-only LLM call. Here each "generation" is a full
MedRAX run: GPT-4o sees the X-ray(s), calls the imaging tools, and answers. The class keeps the
interface of ``experiments.ace.core.Generator`` (``.model`` and ``.generate(...)``) so ACE's
training loop and evaluation helpers can use it unchanged.
"""

import base64
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from medrax.agent import Agent
from medrax.utils import add_playbook_to_prompt

from experiments.ace.playbook_utils import parse_playbook_line

# Same wording as experiments/benchmark_medrax.ipynb, so the no-playbook baseline matches the
# paper's MedRAX setup as closely as possible.
TASK_INSTRUCTIONS = (
    "Answer this question correctly using chain of thought reasoning and carefully evaluating "
    "choices. Solve using your own vision and reasoning and then use tools to complement your "
    "reasoning. Trust your own judgement over any tools."
)

ANSWER_FORMAT_INSTRUCTIONS = (
    "When you are done, end your response with a JSON object on its own line, in exactly this "
    'format:\n{"final_answer": "<one letter, A-F>", "bullet_ids": ["<IDs of playbook entries '
    'that helped, e.g. err-00003; empty list if none>"]}'
)

FOLLOWUP_PROMPT = (
    "If you had to choose the best option, respond with only the JSON object "
    '{"final_answer": "<one letter, A-F>", "bullet_ids": [...]} and nothing else.'
)

_FINAL_JSON_RE = re.compile(r"\{[^{}]*\"final_answer\"[^{}]*\}", re.DOTALL)


class SerializedToolAgent(Agent):
    """MedRAX Agent whose tool executions are serialized by a shared lock.

    ACE evaluates samples in a thread pool. The LLM calls can safely run concurrently, but the
    tools share GPU memory and some (e.g. segmentation, via matplotlib) are not thread-safe, so
    only one tool batch runs at a time.
    """

    def __init__(self, *args, tool_lock: Optional[threading.Lock] = None, **kwargs):
        self._tool_lock = tool_lock or threading.Lock()
        super().__init__(*args, **kwargs)

    def execute_tools(self, state):
        with self._tool_lock:
            return super().execute_tools(state)


def encode_image_as_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if limit and len(text) > limit:
        return text[:limit] + f"... [truncated {len(text) - limit} chars]"
    return text


def format_trace(messages: List[Any], max_tool_output_chars: int = 1500) -> str:
    """Render an agent run as plain text for the ACE Reflector (which cannot see images)."""
    lines = []
    for msg in messages:
        if isinstance(msg, (SystemMessage, HumanMessage)):
            continue
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if content.strip():
                lines.append(f"ASSISTANT: {content.strip()}")
            for call in msg.tool_calls or []:
                lines.append(f"TOOL CALL {call['name']}({json.dumps(call.get('args', {}))})")
        elif isinstance(msg, ToolMessage):
            lines.append(
                f"TOOL RESULT {msg.name}: {_truncate(msg.content, max_tool_output_chars)}"
            )
    return "\n".join(lines)


def parse_final_json(text: str) -> Optional[Dict[str, Any]]:
    """Return the last {"final_answer": ...} object in ``text``, or None."""
    for candidate in reversed(_FINAL_JSON_RE.findall(text or "")):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "final_answer" in parsed:
            return parsed
    return None


def playbook_bullet_ids(playbook: str) -> set:
    ids = set()
    for line in (playbook or "").split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            ids.add(parsed["id"])
    return ids


class MedRAXGenerator:
    """Drop-in replacement for ACE's Generator that runs the MedRAX agent.

    Args:
        agent: A MedRAX Agent (ideally SerializedToolAgent) built with ``system_prompt=""``;
            the system prompt (base prompt + playbook) is sent per call so concurrent calls with
            different playbooks cannot interfere.
        base_system_prompt: MedRAX's MEDICAL_ASSISTANT prompt.
        model_name: Name recorded in ACE's run_config.json.
        max_agent_steps: LangGraph recursion limit (each LLM turn and each tool turn is a step).
        max_tool_output_chars: Truncation for tool outputs in the trace shown to the Reflector.
        send_images_to_llm: Attach the images to the LLM message (GPT-4o vision) in addition to
            listing their paths for the tools.
        run_log_path: Optional JSONL file; every agent run is appended with its full trace.
    """

    def __init__(
        self,
        agent: Agent,
        base_system_prompt: str,
        model_name: str,
        max_agent_steps: int = 30,
        max_tool_output_chars: int = 1500,
        send_images_to_llm: bool = True,
        run_log_path: Optional[str] = None,
    ):
        self.agent = agent
        self.base_system_prompt = base_system_prompt
        self.model = model_name
        self.max_agent_steps = max_agent_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.send_images_to_llm = send_images_to_llm
        self.run_log_path = run_log_path
        self._log_lock = threading.Lock()

    # -- prompt construction ------------------------------------------------

    def build_messages(
        self, question: str, playbook: str, image_paths: List[str], reflection: str
    ) -> List[Any]:
        system = SystemMessage(content=add_playbook_to_prompt(self.base_system_prompt, playbook))

        text = f"{TASK_INSTRUCTIONS}\n{question}\n\nImages (local paths, for tools):\n"
        text += "\n".join(f"- {p}" for p in image_paths)
        if reflection and reflection.strip() and reflection.strip() != "(empty)":
            text += (
                "\n\nA reviewer analyzed your previous attempt at this question. Use their "
                f"diagnosis to avoid repeating the mistake:\n{reflection}"
            )
        text += f"\n\n{ANSWER_FORMAT_INSTRUCTIONS}"

        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        if self.send_images_to_llm:
            for path in image_paths:
                content.append(
                    {"type": "image_url", "image_url": {"url": encode_image_as_data_url(path)}}
                )
        return [system, HumanMessage(content=content)]

    # -- ACE Generator interface -------------------------------------------

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Run MedRAX on one question.

        ``context`` is the JSON string produced by ChestAgentBenchProcessor (image paths, IDs).
        Returns ``(response_text, bullet_ids, call_info)`` like ACE's Generator. The response is
        a readable trace ending in a ``{"final_answer": ...}`` line, which ACE's
        ``extract_answer`` picks up.
        """
        meta = json.loads(context) if context else {}
        image_paths = meta.get("image_paths", [])
        start = time.time()
        record: Dict[str, Any] = {
            "call_id": call_id,
            "run_id": uuid.uuid4().hex[:12],
            "case_id": meta.get("case_id"),
            "question_id": meta.get("question_id"),
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "used_reflection": bool(reflection and reflection.strip() != "(empty)"),
            "used_followup": False,
            "error": None,
        }

        final_answer, bullet_ids, trace, messages = "", [], "", []
        try:
            messages = self.build_messages(question, playbook, image_paths, reflection)
            run_config = {"recursion_limit": self.max_agent_steps}
            state = self.agent.workflow.invoke({"messages": messages}, run_config)
            messages = state["messages"]
            parsed = parse_final_json(messages[-1].content if messages else "")

            if parsed is None:
                # Same fallback as the original benchmark notebook: ask for the choice.
                record["used_followup"] = True
                state = self.agent.workflow.invoke(
                    {"messages": messages + [HumanMessage(content=FOLLOWUP_PROMPT)]}, run_config
                )
                messages = state["messages"]
                parsed = parse_final_json(messages[-1].content if messages else "")

            trace = format_trace(messages, self.max_tool_output_chars)
            if parsed is not None:
                final_answer = str(parsed.get("final_answer", "")).strip()
                valid_ids = playbook_bullet_ids(playbook)
                bullet_ids = [b for b in parsed.get("bullet_ids", []) or [] if b in valid_ids]
        except Exception as e:  # noqa: BLE001 - keep the experiment running, but record it
            record["error"] = f"{type(e).__name__}: {e}"
            trace = (
                format_trace(messages, self.max_tool_output_chars) if messages else ""
            ) + (
                f"\nAGENT ERROR (infrastructure failure, not a reasoning mistake): "
                f"{record['error']}"
            )
            print(f"[MedRAXGenerator] {call_id} failed: {record['error']}")

        duration = time.time() - start
        tool_calls = [
            {"name": c["name"], "args": c.get("args", {})}
            for m in messages
            if isinstance(m, AIMessage)
            for c in (m.tool_calls or [])
        ]
        record.update(
            {
                "duration": round(duration, 2),
                "final_answer": final_answer,
                "bullet_ids": bullet_ids,
                "tools_used": [c["name"] for c in tool_calls],
                "trace": trace,
            }
        )
        self._append_log(record)

        response = (
            f"{trace}\n\nFINAL: "
            + json.dumps({"final_answer": final_answer, "bullet_ids": bullet_ids})
        )
        call_info = {
            "role": "generator",
            "call_id": call_id,
            "model": self.model,
            "total_time": duration,
            "num_tool_calls": len(tool_calls),
            "error": record["error"],
        }
        return response, bullet_ids, call_info

    def _append_log(self, record: Dict[str, Any]) -> None:
        if not self.run_log_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.run_log_path)), exist_ok=True)
        with self._log_lock, open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
