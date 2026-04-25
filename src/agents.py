"""Developer and Critic agents for VisionCoder OpenEnv.

All agent logic (tool-call handling, TODO-list critique, episode loop)
lives here. Prompts are in openenv.prompts.

Usage:
    from openenv.agents import run_episode, AgentConfig

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

from openenv.prompts import (
    DEVELOPER_SYSTEM,
    FIRST_CRITIC_SYSTEM,
    SUBSEQUENT_CRITIC_SYSTEM,
    FALLBACK_HTML,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# TODO list tracker
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclass
class TodoItem:
    text: str       # full item text including "PRIORITY | DIMENSION — description"
    done: bool = False
    priority: str = "MEDIUM"  # HIGH / MEDIUM / LOW


@dataclass
class TodoList:
    items: List[TodoItem] = field(default_factory=list)

    def all_done(self) -> bool:
        return bool(self.items) and all(item.done for item in self.items)

    def pending_count(self) -> int:
        return sum(1 for item in self.items if not item.done)

    def format_for_critic(self) -> str:
        """Previous TODO list passed to Critic — includes priority tag for exact copying."""
        if not self.items:
            return "(No previous TODO list — this is the first review.)"
        lines = ["Previous TODO list (copy with EXACT priority and text, update only the markers):"]
        for item in self.items:
            marker = "[✓]" if item.done else "[ ]"
            lines.append(f"{marker} {item.priority} | {item.text}")
        return "\n".join(lines)

    def format_for_developer(self) -> str:
        """Pending items sorted by priority, formatted as actionable critique."""
        _NOISE_PHRASES = (
            "matches the reference", "which matches",
            "is present and correct", "is correct", "matches reference",
        )
        pending = [
            item for item in self.items
            if not item.done
            and not any(p in item.text.lower() for p in _NOISE_PHRASES)
        ]
        if not pending:
            return (
                "The Critic found no remaining issues. Look carefully at the reference "
                "screenshot for fine details (spacing, colors, missing elements) and refine."
            )
        # Sort by priority: HIGH first, cap at 8 so Developer gets focused feedback
        pending.sort(key=lambda it: _PRIORITY_ORDER.get(it.priority, 1))
        pending = pending[:8]
        lines = ["Fix these issues in priority order (Critic feedback):"]
        for item in pending:
            lines.append(f"- [{item.priority}] {item.text}")
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> "TodoList":
        """Parse a TODO list from Critic output text.

        Expected item format: [✓/[ ]/[+]] PRIORITY | DIMENSION — description
        Priority tag (HIGH/MEDIUM/LOW) is optional — defaults to MEDIUM if absent.

        [+] items are always kept pending (can't resolve same step they're discovered).
        Duplicate and truncated items are dropped.
        """
        _TRUNCATION_ENDINGS = (
            " in", " on", " at", " to", " of", " for", " and", " the",
            " a", " an", " with", " by", " from", " as", " or", " but",
        )
        _VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}

        result = cls()
        seen: set = set()
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("[✓]"):
                item_text = line[3:].strip()
                done = True
            elif line.startswith("[ ]"):
                item_text = line[3:].strip()
                done = False
            elif line.startswith("[+]"):
                item_text = line[3:].strip()
                done = False
            else:
                continue
            if len(item_text) < 10:
                continue
            if any(item_text.lower().endswith(e) for e in _TRUNCATION_ENDINGS):
                continue
            # Extract priority if present: "HIGH | LAYOUT — ..."
            priority = "MEDIUM"
            parts = item_text.split("|", 1)
            if len(parts) == 2:
                candidate = parts[0].strip().upper()
                if candidate in _VALID_PRIORITIES:
                    priority = candidate
                    item_text = parts[1].strip()
            key = item_text.lower()[:60]
            if key not in seen:
                seen.add(key)
                result.items.append(TodoItem(text=item_text, done=done, priority=priority))
        return result

    @classmethod
    def merge(cls, prev: "TodoList", updated: "TodoList") -> "TodoList":
        """Merge updated list back — re-adds any pending prev items the Critic forgot.

        Uses 40-char prefix matching so paraphrased items count as the same issue.
        Resolved prev items (done=True) are never re-added.
        New [+] items introduced in this step are capped at 3 by priority so the
        list doesn't balloon when the model ignores the per-step limit.
        """
        prev_prefixes = {item.text.lower()[:40] for item in prev.items}

        # Separate carried items (also in prev) from genuinely new [+] items
        carried: list = []
        new_items: list = []
        for item in updated.items:
            if item.text.lower()[:40] in prev_prefixes:
                carried.append(item)
            else:
                new_items.append(item)

        # Keep at most 3 new items (highest priority first)
        new_items.sort(key=lambda it: _PRIORITY_ORDER.get(it.priority, 1))
        new_items = new_items[:3]

        result = cls(items=carried + new_items)
        updated_prefixes = {item.text.lower()[:40] for item in result.items}

        # Re-add any pending prev items the Critic dropped entirely
        for prev_item in prev.items:
            if prev_item.done:
                continue
            if prev_item.text.lower()[:40] not in updated_prefixes:
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
    current_html: str = "",
    dbg=None,
) -> Tuple[str, TodoList]:
    """Critic reviews current render vs reference and returns (raw_text, updated TodoList).

    Receives the Developer's HTML source so it can write selector-specific CSS fixes
    instead of abstract visual observations.
    """
    is_first = prev_todo is None

    if dbg:
        prev_critique_text = prev_todo.format_for_developer() if prev_todo else None
        dbg.log_critic_input(ref_b64, render_prev_b64, prev_critique_text, render_curr_b64)

    system = FIRST_CRITIC_SYSTEM if is_first else SUBSEQUENT_CRITIC_SYSTEM

    critic_messages = [{"role": "system", "content": system}]

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
    ]

    if current_html:
        content.append({
            "type": "text",
            "text": (
                f"\nDeveloper's current HTML source (use exact selectors in your FIX instructions):\n"
                f"```html\n{current_html[:3000]}\n```"
            ),
        })

    if is_first:
        content.append({
            "type": "text",
            "text": (
                "\nThis is the first review. Perform a comprehensive visual audit covering "
                "LAYOUT, STRUCTURE, COLOR, TYPOGRAPHY, SPACING, and TEXT dimensions. "
                "Output your initial TODO LIST with [+] items only. "
                "Each item MUST include a → FIX: instruction with exact CSS."
            ),
        })
    else:
        content.append({
            "type": "text",
            "text": (
                f"\n{prev_todo.format_for_critic()}\n\n"
                "Update the TODO list based on what you see in the CURRENT RENDER and HTML. "
                "Mark fixed items [✓], keep unresolved items [ ] (update FIX selector if HTML changed), "
                "add new issues with [+]. Each item must have a → FIX: instruction. "
                "Stop after the last item — no STATUS or summary line."
            ),
        })

    critic_messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=critic_messages,
        max_tokens=2048,
        temperature=0.1,
    )
    critique_text = response.choices[0].message.content or ""

    updated_todo = TodoList.parse(critique_text)
    if prev_todo:
        if updated_todo.all_done():
            # Critic explicitly marked every visible item [✓] — trust that signal.
            # Skipping merge avoids re-adding items the Critic intentionally resolved.
            pass
        else:
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
    - Critic marks all TODO items resolved
    - No reward improvement for 2 consecutive steps (plateau)

    Monotonic reward guarantee: Developer always receives the best-seen HTML as
    its base, so regressions don't compound. If a step produces lower reward the
    Developer retries from the best-known state on the next step.
    """
    client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    current_html = ""
    best_html = ""
    best_reward = 0.0
    no_improve_streak = 0
    _MAX_NO_IMPROVE = 2

    todo: Optional[TodoList] = None
    render_prev: Optional[str] = None
    results: List[StepResult] = []

    for step_i in range(config.max_steps):
        # Guard: Critic resolved everything
        if todo is not None and todo.pending_count() == 0:
            break
        # Guard: plateau — no improvement for N consecutive steps
        if no_improve_streak >= _MAX_NO_IMPROVE:
            print(
                f"[CRITIC] No improvement for {_MAX_NO_IMPROVE} consecutive steps "
                f"(best={best_reward:.3f}) — stopping early.",
                flush=True,
            )
            break

        error: Optional[str] = None

        # Developer always starts from the best-seen HTML to avoid compounding regressions
        try:
            current_html = developer_turn(
                client, env_client, config.model,
                ref_b64, best_html, todo, dbg,
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

        # Monotonic tracking — update best only on genuine improvement
        if reward > best_reward:
            best_reward = reward
            best_html = current_html
            no_improve_streak = 0
        else:
            no_improve_streak += 1

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
                    current_html=current_html,
                    dbg=dbg,
                )
                sr.critique = critique_text
                sr.todo = todo
                preview = critique_text.replace("\n", " ")[:200]
                print(
                    f"[CRITIC] step={step_n} reward={reward:.3f} best={best_reward:.3f} → {preview}",
                    flush=True,
                )
            except Exception as exc:
                logger.warning("Critic failed: %s", exc)
                todo = None

        results.append(sr)
        render_prev = render_full

        if env_done:
            break
        if todo is not None and todo.pending_count() == 0:
            print(
                f"[CRITIC] All items resolved at step={step_n} reward={reward:.3f} — stopping.",
                flush=True,
            )
            break

    return results


# ---------------------------------------------------------------------------
# Approach B: Long-horizon Developer (no Critic, sees full history)
# ---------------------------------------------------------------------------

def developer_turn_long_horizon(
    client: OpenAI,
    model: str,
    ref_b64: str,
    history: List[Tuple[str, str]],  # list of (render_full_b64, html)
    dbg=None,
) -> str:
    """Developer with full history: reference + all previous renders + all previous HTML."""
    messages = [{"role": "system", "content": DEVELOPER_SYSTEM}]

    user_content: list = [
        {"type": "text", "text": "Reference screenshot (reproduce this UI):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]

    if history:
        for i, (render_b64, prev_html) in enumerate(history, 1):
            user_content.append({
                "type": "text",
                "text": f"\n\nStep {i} render:",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{render_b64}"},
            })
            user_content.append({
                "type": "text",
                "text": f"Step {i} HTML:\n```html\n{prev_html[:2000]}\n```",
            })
        user_content.append({
            "type": "text",
            "text": (
                "\n\nAll your previous attempts are shown above. "
                "Generate improved HTML that better matches the reference. "
                "Output the HTML only."
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
    return html_out if _looks_like_html(html_out) else FALLBACK_HTML


def run_episode_long_dev(
    env_client,
    config: AgentConfig,
    session_id: str,
    ref_b64: str,
    dbg=None,
    on_step=None,
) -> List[StepResult]:
    """Approach B: Long-horizon Developer only — full history, no Critic."""
    client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    current_html = ""
    history: List[Tuple[str, str]] = []
    results: List[StepResult] = []

    for step_i in range(config.max_steps):
        error: Optional[str] = None
        try:
            current_html = developer_turn_long_horizon(
                client, config.model, ref_b64, history, dbg
            )
        except Exception as exc:
            error = str(exc)[:120]
            current_html = FALLBACK_HTML

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

        step_n = step_i + 1
        sr = StepResult(
            step=step_n,
            html=current_html,
            reward=reward,
            done=env_done,
            critique="",
            todo=None,
            render_full_b64=render_full,
            sub_rewards=sub_rewards,
            error=error,
        )
        if on_step:
            on_step(sr)

        if render_full:
            history.append((render_full, current_html))

        results.append(sr)
        if env_done:
            break

    return results


# ---------------------------------------------------------------------------
# Approach C: Short-horizon Developer (no Critic, sees only last render)
# ---------------------------------------------------------------------------

def developer_turn_short_horizon(
    client: OpenAI,
    model: str,
    ref_b64: str,
    prev_render_b64: Optional[str],
    prev_html: Optional[str],
    dbg=None,
) -> str:
    """Developer with short horizon: reference + only last render + only last HTML."""
    messages = [{"role": "system", "content": DEVELOPER_SYSTEM}]

    user_content: list = [
        {"type": "text", "text": "Reference screenshot (reproduce this UI):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]

    if prev_render_b64 and prev_html:
        user_content += [
            {"type": "text", "text": "\n\nYour previous render:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_render_b64}"}},
            {
                "type": "text",
                "text": (
                    f"\n\nYour previous HTML:\n```html\n{prev_html[:3000]}\n```\n\n"
                    "Compare the renders and output improved HTML only."
                ),
            },
        ]
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
    return html_out if _looks_like_html(html_out) else FALLBACK_HTML


def run_episode_short_dev(
    env_client,
    config: AgentConfig,
    session_id: str,
    ref_b64: str,
    dbg=None,
    on_step=None,
) -> List[StepResult]:
    """Approach C: Short-horizon Developer only — sees only last render each step, no Critic."""
    client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    current_html = ""
    prev_render: Optional[str] = None
    results: List[StepResult] = []

    for step_i in range(config.max_steps):
        error: Optional[str] = None
        try:
            current_html = developer_turn_short_horizon(
                client, config.model, ref_b64,
                prev_render,
                current_html if step_i > 0 else None,
                dbg,
            )
        except Exception as exc:
            error = str(exc)[:120]
            current_html = FALLBACK_HTML

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

        step_n = step_i + 1
        sr = StepResult(
            step=step_n,
            html=current_html,
            reward=reward,
            done=env_done,
            critique="",
            todo=None,
            render_full_b64=render_full,
            sub_rewards=sub_rewards,
            error=error,
        )
        if on_step:
            on_step(sr)

        prev_render = render_full  # only keep the latest render
        results.append(sr)
        if env_done:
            break

    return results


# ---------------------------------------------------------------------------
# Approach D: Long-horizon Developer (low-res renders) + simple free-form Critic
# ---------------------------------------------------------------------------

_SIMPLE_CRITIC_SYSTEM = (
    "You are a UI reviewer. You will be shown a reference screenshot and a current render "
    "of HTML that is meant to reproduce it.\n\n"
    "Describe what needs to change the most to make the render match the reference. "
    "Be concise and specific — mention exact colors, sizes, or elements where helpful. "
    "You can write a short paragraph or a bullet list. No structured format required."
)

_SIMPLE_DEV_SYSTEM = (
    "You are a UI-to-code expert. Given a reference screenshot of a web page, "
    "generate complete HTML with inline CSS that reproduces the layout as accurately as possible.\n\n"
    "Critical layout rules:\n"
    "- Always use `* { box-sizing: border-box; margin: 0; padding: 0; }` reset.\n"
    "- Page and all top-level sections must be full-width: `width: 100%; min-height: 100vh`.\n"
    "- Never center-constrain the overall page — only constrain inner content containers if the reference does.\n"
    "- Match background colors, section colors, and typography as precisely as possible.\n\n"
    "Output ONLY the raw HTML code starting with <!DOCTYPE html>. "
    "No explanations, no markdown fences — just the HTML."
)


def _simple_critic_turn(
    client: OpenAI,
    model: str,
    ref_b64: str,
    render_full_b64: str,
) -> str:
    """Simple free-form critic: compare ref vs render, say what needs to change most."""
    messages = [{"role": "system", "content": _SIMPLE_CRITIC_SYSTEM}]
    messages.append({"role": "user", "content": [
        {"type": "text", "text": "Reference:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
        {"type": "text", "text": "Current render:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_full_b64}"}},
        {"type": "text", "text": "What needs to change the most?"},
    ]})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _developer_turn_d(
    client: OpenAI,
    model: str,
    ref_b64: str,
    history: List[Tuple[str, str]],  # (render_low_b64, html)
    critique: Optional[str],
) -> str:
    """Approach D developer: full-res ref + all previous low-res renders + all HTML + critic feedback."""
    messages = [{"role": "system", "content": _SIMPLE_DEV_SYSTEM}]

    user_content: list = [
        {"type": "text", "text": "Reference screenshot (reproduce this UI):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]

    if history:
        for i, (render_low_b64, prev_html) in enumerate(history, 1):
            user_content.append({"type": "text", "text": f"\n\nStep {i} render (low-res preview):"})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_low_b64}"}})
            user_content.append({"type": "text", "text": f"Step {i} HTML:\n```html\n{prev_html[:2000]}\n```"})

    if critique:
        user_content.append({
            "type": "text",
            "text": f"\n\nReviewer feedback on your last render:\n{critique}\n\nGenerate improved HTML addressing this feedback. Output the HTML only.",
        })
    elif history:
        user_content.append({
            "type": "text",
            "text": "\n\nGenerate improved HTML that better matches the reference. Output the HTML only.",
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
    return html_out if _looks_like_html(html_out) else FALLBACK_HTML


def run_episode_d(
    env_client,
    config: AgentConfig,
    session_id: str,
    ref_b64: str,
    dbg=None,
    on_step=None,
) -> List[StepResult]:
    """Approach D: long-horizon dev (low-res renders) + simple free-form critic."""
    client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    current_html = ""
    history: List[Tuple[str, str]] = []  # (render_low_b64, html)
    critique: Optional[str] = None
    results: List[StepResult] = []

    for step_i in range(config.max_steps):
        error: Optional[str] = None
        try:
            current_html = _developer_turn_d(
                client, config.model, ref_b64, history, critique
            )
        except Exception as exc:
            error = str(exc)[:120]
            current_html = FALLBACK_HTML

        step_resp = env_client.post(
            "/step",
            json={"html": current_html, "session_id": session_id},
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = float(result.get("reward", 0.0))
        env_done = bool(result.get("done", False))
        render_full = result.get("render_full")
        render_low = result.get("render_low")
        sub_rewards = result.get("metadata", {}).get("rewards")

        step_n = step_i + 1
        sr = StepResult(
            step=step_n,
            html=current_html,
            reward=reward,
            done=env_done,
            critique=critique or "",
            todo=None,
            render_full_b64=render_full,
            sub_rewards=sub_rewards,
            error=error,
        )
        if on_step:
            on_step(sr)

        if render_low:
            history.append((render_low, current_html))

        # Critic turn (skip on final env step)
        if not env_done and render_full:
            try:
                critique = _simple_critic_turn(client, config.model, ref_b64, render_full)
                preview = critique.replace("\n", " ")[:200]
                print(f"[CRITIC-D] step={step_n} reward={reward:.2f} → {preview}", flush=True)
            except Exception as exc:
                logger.warning("Critic-D failed: %s", exc)
                critique = None

        results.append(sr)
        if env_done:
            break

    return results
