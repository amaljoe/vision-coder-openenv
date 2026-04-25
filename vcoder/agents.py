"""Developer and Critic agents for VisionCoder OpenEnv.

All agent logic (prompts, tool-call handling, TODO-list critique, episode loop)
lives here so inference.py remains a thin driver script.

Usage:
    from vcoder.agents import run_episode, AgentConfig

    config = AgentConfig(api_key=..., api_base=..., model=...)
    result = run_episode(env_client, config, difficulty="hard", session=obs, dbg=dbg)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags (read from env at import time, same as inference.py)
# ---------------------------------------------------------------------------
ONE_SHOT = bool(os.environ.get("ONE_SHOT", ""))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DEVELOPER_SYSTEM = (
    "You are a UI-to-code expert. Given a reference screenshot of a web page, "
    "generate complete HTML with inline CSS that reproduces the layout as accurately as possible.\n\n"
    "Output ONLY the raw HTML code starting with <!DOCTYPE html>. "
    "No explanations, no markdown fences, no tool calls — just the HTML."
)

CRITIC_SYSTEM = (
    "You are a precise UI reviewer. You will be shown a reference screenshot and the current "
    "rendered HTML output.\n\n"
    "You maintain a TODO list of issues to fix. On each review:\n"
    "1. Mark previously listed items [✓] if they are now fixed in the current render.\n"
    "2. Keep previously listed items [ ] if still not fixed.\n"
    "3. Add new items with [+] for newly discovered issues.\n\n"
    "Always respond in this exact format:\n\n"
    "TODO LIST:\n"
    "[ ] or [✓] or [+] <specific issue>\n"
    "...\n\n"
    "STATUS: DONE (if all items are [✓]) or OPEN (<N> items remaining)\n\n"
    "Rules:\n"
    "- Be concrete: name specific elements, colors, text, sizes, positions.\n"
    "- Always list at least 1 item.\n"
    "- DONE only when every item is [✓] — meaning the render matches the reference.\n"
    "- A blank, white, or unstyled page is never DONE."
)

FALLBACK_HTML = "<!DOCTYPE html><html><head></head><body><p>Generation failed.</p></body></html>"

# ---------------------------------------------------------------------------
# One-shot example messages
# ---------------------------------------------------------------------------

_DEV_ONESHOT_EXAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: sans-serif; background: #f5f5f5; margin: 0; }
  .card { background: #fff; border-radius: 8px; padding: 24px; max-width: 400px; margin: 40px auto; box-shadow: 0 2px 8px rgba(0,0,0,.1); }
  h2 { margin: 0 0 8px; color: #1a1a2e; }
  p  { margin: 0; color: #555; font-size: 14px; }
  .btn { margin-top: 16px; display: inline-block; padding: 10px 20px; background: #4f46e5; color: #fff; border-radius: 6px; text-decoration: none; font-size: 14px; }
</style>
</head>
<body>
  <div class="card">
    <h2>Welcome back!</h2>
    <p>Your dashboard is ready. Explore your latest metrics below.</p>
    <a class="btn" href="#">Go to Dashboard</a>
  </div>
</body>
</html>"""

_DEV_ONESHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "[Example task] Reference screenshot: a card UI with title 'Welcome back!', "
            "subtitle text, and an indigo 'Go to Dashboard' button on a light grey background."
        ),
    },
    {
        "role": "assistant",
        "content": _DEV_ONESHOT_EXAMPLE_HTML,
    },
]

_CRITIC_ONESHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "[Example 1] Reference: blue navigation bar, white logo centred, three menu items. "
            "Current render: plain white page, no navigation bar at all."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "TODO LIST:\n"
            "[+] Add full-width blue (#1e40af) navigation bar at the top\n"
            "[+] Add white logo centred in the nav bar\n"
            "[+] Add three menu items (e.g. Home, About, Contact) in the nav\n\n"
            "STATUS: OPEN (3 items remaining)"
        ),
    },
    {
        "role": "user",
        "content": (
            "[Example 2, step 2 of same episode] Previous TODO shown above. "
            "Current render now shows: blue nav bar and logo present; still missing the three menu items."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "TODO LIST:\n"
            "[✓] Add full-width blue (#1e40af) navigation bar at the top\n"
            "[✓] Add white logo centred in the nav bar\n"
            "[ ] Add three menu items (Home, About, Contact) in the nav\n\n"
            "STATUS: OPEN (1 item remaining)"
        ),
    },
    {
        "role": "user",
        "content": (
            "[Example 3, step 3 of same episode] All three menu items now present."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "TODO LIST:\n"
            "[✓] Add full-width blue (#1e40af) navigation bar at the top\n"
            "[✓] Add white logo centred in the nav bar\n"
            "[✓] Add three menu items (Home, About, Contact) in the nav\n\n"
            "STATUS: DONE"
        ),
    },
]

# ---------------------------------------------------------------------------
# TODO list tracker
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    text: str
    done: bool = False


@dataclass
class TodoList:
    items: List[TodoItem] = field(default_factory=list)

    def all_done(self) -> bool:
        return bool(self.items) and all(item.done for item in self.items)

    def pending_count(self) -> int:
        return sum(1 for item in self.items if not item.done)

    def format_for_critic(self) -> str:
        """Previous TODO list passed to Critic so it can update it."""
        if not self.items:
            return "(No previous TODO list — this is the first review.)"
        lines = ["Previous TODO list (update based on what you see in the current render):"]
        for item in self.items:
            marker = "[✓]" if item.done else "[ ]"
            lines.append(f"{marker} {item.text}")
        return "\n".join(lines)

    def format_for_developer(self) -> str:
        """Pending items formatted as actionable critique for the Developer."""
        pending = [item for item in self.items if not item.done]
        if not pending:
            return ""
        lines = ["Fix the following issues (Critic feedback):"]
        for item in pending:
            lines.append(f"- {item.text}")
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> "TodoList":
        """Parse a TODO list from Critic output text."""
        result = cls()
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("[✓]"):
                result.items.append(TodoItem(text=line[3:].strip(), done=True))
            elif line.startswith("[ ]"):
                result.items.append(TodoItem(text=line[3:].strip(), done=False))
            elif line.startswith("[+]"):
                result.items.append(TodoItem(text=line[3:].strip(), done=False))
        return result

    @classmethod
    def merge(cls, prev: "TodoList", updated: "TodoList") -> "TodoList":
        """Merge updated list back — preserving any prev items the Critic forgot to include.

        Items from updated take precedence; prev items not in updated are appended as pending
        only if they were not yet done (Critic may omit resolved items).
        """
        result = cls(items=list(updated.items))
        updated_texts = {item.text.lower() for item in updated.items}
        for prev_item in prev.items:
            if prev_item.text.lower() not in updated_texts and not prev_item.done:
                # Re-add pending items the Critic forgot about
                result.items.append(prev_item)
        return result


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _looks_like_html(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("<!doctype") or t.startswith("<html")


def _parse_qwen_xml_tool_call(content: str) -> Optional[Tuple[str, dict]]:
    """Fallback parser for Qwen3's XML tool call format when vllm hermes parser misses it."""
    if "<tool_call>" not in content:
        return None
    fn_m = re.search(r"<function=(\w+)>", content)
    if not fn_m:
        return None
    func_name = fn_m.group(1)
    args = {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"<parameter=(\w+)>(.*?)(?:</parameter>|\Z)", content, re.DOTALL)
    }
    return (func_name, args) if args else None


def _clean_html_output(content: str) -> str:
    """Strip residual <tool_call> wrapper or markdown fences from model output."""
    parsed = _parse_qwen_xml_tool_call(content)
    if parsed:
        _, args = parsed
        if "html" in args:
            return args["html"]
    fence = re.match(r"```(?:html)?\s*(.*?)\s*```", content, re.DOTALL)
    if fence:
        return fence.group(1)
    return content


# ---------------------------------------------------------------------------
# Developer agent
# ---------------------------------------------------------------------------

def developer_turn(
    client: OpenAI,
    env_client,  # unused — kept for signature compatibility
    model: str,
    ref_b64: str,
    current_html: str,
    todo: Optional[TodoList] = None,
    dbg=None,
) -> str:
    """Developer generates HTML from the reference screenshot in a single LLM call.

    No tools — rendering is the environment's responsibility after step().
    On subsequent steps the Critic's TODO list is included so the Developer
    knows exactly what to fix.
    """
    if dbg:
        dbg.log_developer_input(current_html, todo.format_for_developer() if todo else None)

    messages = [{"role": "system", "content": DEVELOPER_SYSTEM}]
    if ONE_SHOT:
        messages.extend(_DEV_ONESHOT_MESSAGES)

    user_content: list = [
        {"type": "text", "text": "Reference screenshot (reproduce this UI):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]

    if current_html and todo and todo.items:
        user_content.append({
            "type": "text",
            "text": (
                f"\n\nYour previous HTML:\n```html\n{current_html[:3000]}\n```\n\n"
                f"{todo.format_for_developer()}\n\n"
                "Output the revised HTML only."
            ),
        })
    else:
        user_content.append({
            "type": "text",
            "text": "\n\nGenerate complete HTML with inline CSS. Output the HTML only.",
        })

    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )
    content = response.choices[0].message.content or ""
    html_out = _clean_html_output(content)
    if not _looks_like_html(html_out):
        html_out = FALLBACK_HTML
    if dbg:
        dbg.log_developer_output(html_out)
    return html_out


# ---------------------------------------------------------------------------
# Critic agent
# ---------------------------------------------------------------------------

def critic_turn(
    client: OpenAI,
    model: str,
    ref_b64: str,
    render_curr_b64: str,
    prev_todo: Optional[TodoList],
    render_prev_b64: Optional[str] = None,
    dbg=None,
) -> Tuple[str, TodoList]:
    """Critic reviews current render vs reference and returns (raw_text, updated TodoList).

    The TodoList is maintained across steps: items are marked done/pending/new
    based on what the Critic observes. Episode ends programmatically when all_done().
    """
    if dbg:
        prev_critique_text = prev_todo.format_for_developer() if prev_todo else None
        dbg.log_critic_input(ref_b64, render_prev_b64, prev_critique_text, render_curr_b64)

    critic_messages = [{"role": "system", "content": CRITIC_SYSTEM}]
    if ONE_SHOT:
        critic_messages.extend(_CRITIC_ONESHOT_MESSAGES)

    content: list = [
        {"type": "text", "text": "Reference screenshot:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]

    if render_prev_b64 and prev_todo:
        content += [
            {"type": "text", "text": "Previous render (before this step's revision):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_prev_b64}"}},
        ]

    content += [
        {"type": "text", "text": "Current render:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_curr_b64}"}},
        {
            "type": "text",
            "text": (
                f"\n{prev_todo.format_for_critic() if prev_todo else '(First review — create the initial TODO list.)'}\n\n"
                "Update the TODO list based on what you see. "
                "Mark fixed items [✓], keep unresolved items [ ], add new issues with [+]. "
                "Output STATUS: DONE only when every item is [✓]."
            ),
        },
    ]

    critic_messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=critic_messages,
        max_tokens=1024,
        temperature=0.1,
    )
    critique_text = response.choices[0].message.content or ""

    updated_todo = TodoList.parse(critique_text)
    if prev_todo:
        updated_todo = TodoList.merge(prev_todo, updated_todo)

    if dbg:
        dbg.log_critic_output(critique_text, updated_todo)

    return critique_text, updated_todo


# ---------------------------------------------------------------------------
# Episode config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    api_key: str
    api_base: str
    model: str
    max_steps: int = 5
    done_reward_threshold: float = 0.75


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    html: str
    reward: float
    done: bool
    critique: str
    todo: Optional[TodoList]
    render_full_b64: Optional[str]
    sub_rewards: Optional[dict]
    error: Optional[str] = None


def run_episode(
    env_client,
    config: AgentConfig,
    session_id: str,
    ref_b64: str,
    dbg=None,
    on_step=None,  # optional callback(StepResult) → None, called immediately after env step
) -> List[StepResult]:
    """Run one full episode (Developer↔Critic loop) and return per-step results.

    Terminates when:
    - max_steps reached (env done=True)
    - TodoList.all_done() AND reward >= done_reward_threshold

    on_step is called right after each env.step() so callers can log [STEP] before
    [CRITIC] appears — keeping stdout in chronological order.
    """
    client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    current_html = ""
    todo: Optional[TodoList] = None
    render_prev: Optional[str] = None
    results: List[StepResult] = []

    for step_i in range(config.max_steps):
        error: Optional[str] = None

        # Developer turn
        try:
            current_html = developer_turn(
                client, env_client, config.model,
                ref_b64, current_html, todo, dbg,
            )
        except Exception as exc:
            error = str(exc)[:120]
            current_html = FALLBACK_HTML

        # Step the environment
        step_resp = env_client.post(
            "/step",
            json={"html": current_html, "session_id": session_id},
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = float(result.get("reward", 0.0))
        env_done = bool(result.get("done", False))
        render_full = result.get("render_full")
        sub_rewards = result.get("metadata", {}).get("rewards")

        if dbg:
            dbg.log_step_result(reward, env_done, render_full, sub_rewards)

        step_n = step_i + 1

        sr = StepResult(
            step=step_n,
            html=current_html,
            reward=reward,
            done=env_done,
            critique="",
            todo=todo,
            render_full_b64=render_full,
            sub_rewards=sub_rewards,
            error=error,
        )

        # Notify caller immediately so [STEP] prints before [CRITIC]
        if on_step:
            on_step(sr)

        # Critic turn (skip on final env step)
        if not env_done:
            try:
                critique_text, todo = critic_turn(
                    client, config.model,
                    ref_b64, render_full,
                    prev_todo=todo,
                    render_prev_b64=render_prev,
                    dbg=dbg,
                )
                sr.critique = critique_text
                sr.todo = todo
                preview = critique_text.replace("\n", " ")[:200]
                print(f"[CRITIC] step={step_n} reward={reward:.2f} → {preview}", flush=True)
            except Exception as exc:
                logger.warning("Critic failed: %s", exc)
                todo = None

        results.append(sr)
        render_prev = render_full

        # Termination checks
        if env_done:
            break
        if todo is not None and todo.all_done() and reward >= config.done_reward_threshold:
            print(f"[CRITIC] All TODO items resolved at step={step_n} reward={reward:.2f} — stopping.", flush=True)
            break

    return results
