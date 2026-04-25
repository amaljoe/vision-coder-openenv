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
    "Critical layout rules:\n"
    "- Always use `* { box-sizing: border-box; margin: 0; padding: 0; }` reset.\n"
    "- Page and all top-level sections must be full-width: `width: 100%; min-height: 100vh`.\n"
    "- Never center-constrain the overall page — only constrain inner content containers if the reference does.\n"
    "- Match background colors, section colors, and typography as precisely as possible.\n\n"
    "Output ONLY the raw HTML code starting with <!DOCTYPE html>. "
    "No explanations, no markdown fences, no tool calls — just the HTML."
)

FIRST_CRITIC_SYSTEM = (
    "You are a precise UI reviewer performing an initial visual audit.\n\n"
    "You will be shown a reference screenshot and the current rendered HTML output.\n"
    "Systematically audit ALL of the following dimensions:\n\n"
    "1. LAYOUT — Does the page fill the full viewport width? Is the column/grid structure correct?\n"
    "   Is any content cropped or cut off at the edges? Are sections the right width/height?\n\n"
    "2. STRUCTURE — Are all major sections present? (header/nav, hero, content areas, cards, sidebar, footer)\n"
    "   Are sections in the correct visual order and proportion?\n\n"
    "3. COLOR — Background color, text colors, button colors, section backgrounds, borders.\n"
    "   Estimate hex codes from the image (e.g. 'should be dark navy ~#0f172a, currently white #ffffff').\n\n"
    "4. TYPOGRAPHY — Font size (headings vs body), font weight (bold/normal), font family if visually distinctive.\n\n"
    "5. SPACING — Padding inside sections, gaps between elements, overall visual density.\n\n"
    "6. TEXT — Only flag if a visible heading or label is clearly wrong or absent.\n\n"
    "Respond in this EXACT format:\n\n"
    "TODO LIST:\n"
    "[+] DIMENSION — specific issue with reference vs render comparison\n"
    "[+] ...\n\n"
    "STATUS: OPEN (<N> items remaining)\n\n"
    "Rules:\n"
    "- Use [+] for every item (this is the first review — no history).\n"
    "- Each item MUST start with a DIMENSION prefix: LAYOUT / STRUCTURE / COLOR / TYPOGRAPHY / SPACING / TEXT.\n"
    "- Be quantitative: 'content too narrow ~400px, reference is full-width ~1200px' not just 'too narrow'.\n"
    "- For colors: always include estimated hex — 'background should be dark navy ~#0f172a, render is white #ffffff'.\n"
    "- A blank or nearly-blank white page must generate many items. Never output STATUS: DONE on a first review.\n"
    "- Do NOT flag items that appear correct in the render."
)

SUBSEQUENT_CRITIC_SYSTEM = (
    "You are a precise UI reviewer updating a running TODO list.\n\n"
    "You will see the reference screenshot, the previous render, and the current render.\n"
    "Update the TODO list based on what you observe in the CURRENT RENDER:\n\n"
    "1. Look carefully at the CURRENT RENDER image before deciding anything.\n"
    "2. Mark [✓] ONLY if you can clearly see the issue is resolved in the current render.\n"
    "3. Keep [ ] if the issue is still visibly wrong.\n"
    "4. Add [+] for newly spotted differences — use the same DIMENSION prefix format.\n\n"
    "Respond in this EXACT format:\n\n"
    "TODO LIST:\n"
    "[ ] or [✓] or [+] DIMENSION — issue description\n"
    "...\n\n"
    "STATUS: DONE (if all items are [✓]) or OPEN (<N> items remaining)\n\n"
    "Rules:\n"
    "- Copy EVERY item from the previous list, updating only the marker.\n"
    "- Do NOT mark [✓] unless you can clearly see the fix in the current render.\n"
    "- Do NOT output STATUS: DONE unless every single item is [✓].\n"
    "- A blank or mostly-white page is never DONE.\n"
    "- [+] items added now are always OPEN — they can only be [✓] in a future step."
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
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: sans-serif; background: #0f172a; color: #f1f5f9; }
  nav { display: flex; align-items: center; justify-content: space-between;
        padding: 16px 40px; background: #1e293b; width: 100%; }
  nav .logo { font-weight: 700; font-size: 20px; color: #fff; }
  nav a { color: #cbd5e1; text-decoration: none; margin-left: 24px; font-size: 14px; }
  nav .btn { padding: 8px 18px; background: #22c55e; color: #fff; border-radius: 6px;
             font-size: 14px; margin-left: 32px; }
  .hero { text-align: center; padding: 80px 40px; }
  .hero h1 { font-size: 48px; font-weight: 800; margin-bottom: 16px; }
  .hero p  { font-size: 18px; color: #94a3b8; margin-bottom: 32px; }
  .hero .cta { display: inline-block; padding: 14px 32px; background: #22c55e; color: #fff;
               border-radius: 8px; text-decoration: none; font-size: 16px; font-weight: 600; }
  .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
              padding: 60px 40px; background: #1e293b; width: 100%; }
  .card { background: #0f172a; border-radius: 10px; padding: 24px; }
  .card h3 { font-size: 18px; margin-bottom: 8px; }
  .card p  { font-size: 14px; color: #94a3b8; }
  footer { text-align: center; padding: 32px 40px; color: #64748b; font-size: 14px; }
</style>
</head>
<body>
  <nav>
    <span class="logo">SiteName</span>
    <div>
      <a href="#">Features</a>
      <a href="#">Pricing</a>
      <a href="#">Docs</a>
      <a class="btn" href="#">Get Started</a>
    </div>
  </nav>
  <section class="hero">
    <h1>Build faster, ship better</h1>
    <p>The platform teams trust to deliver results.</p>
    <a class="cta" href="#">Start free trial</a>
  </section>
  <section class="features">
    <div class="card"><h3>Feature One</h3><p>Short description of this feature and its benefits.</p></div>
    <div class="card"><h3>Feature Two</h3><p>Short description of this feature and its benefits.</p></div>
    <div class="card"><h3>Feature Three</h3><p>Short description of this feature and its benefits.</p></div>
  </section>
  <footer>&copy; 2024 SiteName. All rights reserved.</footer>
</body>
</html>"""

_DEV_ONESHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "[Example task] Reference screenshot: a full-width dark-themed SaaS landing page. "
            "Dark navy background (#0f172a). Full-width top nav with logo on the left, "
            "nav links (Features, Pricing, Docs) and a green 'Get Started' button on the right. "
            "Large centered hero section with big white heading, subtitle in muted grey, and a green CTA button. "
            "Below: a dark grey (#1e293b) full-width section with three feature cards in a row. "
            "Footer with copyright text."
        ),
    },
    {
        "role": "assistant",
        "content": _DEV_ONESHOT_EXAMPLE_HTML,
    },
]

_FIRST_CRITIC_ONESHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "[FORMAT EXAMPLE — generic placeholders, do NOT copy these specifics]\n"
            "Reference: a full-width SaaS landing page. Dark navy background. Full-width header nav "
            "with logo on left, nav links, and 'Get Started' button on right. Large hero section with "
            "white headline and bright green CTA button. Below: a three-column feature cards section "
            "with icon + title + body per card. Footer with links at the bottom.\n"
            "Current render: a narrow centered white page (~500px wide). Only plain unstyled text, "
            "no sections, no colors, no layout structure.\n"
            "(This is the first review — no previous TODO list.)"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "TODO LIST:\n"
            "[+] LAYOUT — page content is narrow (~500px centered); reference is full-width (~1200px, 100vw)\n"
            "[+] LAYOUT — content is cut off on the right edge; reference shows no horizontal overflow\n"
            "[+] STRUCTURE — header/navigation bar is absent; reference has a full-width nav with logo and CTA\n"
            "[+] STRUCTURE — hero section is absent; reference has a large banner with headline and CTA button\n"
            "[+] STRUCTURE — three-column feature cards section is missing; reference shows icon+title+body cards\n"
            "[+] STRUCTURE — footer section is absent from the render\n"
            "[+] COLOR — page background is white #ffffff; reference shows dark navy ~#0f172a\n"
            "[+] COLOR — CTA button should be bright green ~#22c55e; no button visible in render\n"
            "[+] TYPOGRAPHY — body text appears ~10px; reference shows ~16px normal weight\n"
            "[+] SPACING — no vertical padding between sections; reference shows ~80px top/bottom per section\n\n"
            "STATUS: OPEN (10 items remaining)"
        ),
    },
]

_SUBSEQUENT_CRITIC_ONESHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "[FORMAT EXAMPLE step 2 — generic placeholders, do NOT copy these specifics]\n"
            "Previous TODO list:\n"
            "[+] LAYOUT — page content is narrow (~500px centered); reference is full-width (~1200px, 100vw)\n"
            "[+] LAYOUT — content is cut off on the right edge\n"
            "[+] STRUCTURE — header/navigation bar is absent\n"
            "[+] STRUCTURE — hero section is absent\n"
            "[+] STRUCTURE — three-column feature cards section is missing\n"
            "[+] STRUCTURE — footer section is absent\n"
            "[+] COLOR — page background is white #ffffff; reference shows dark navy ~#0f172a\n"
            "[+] COLOR — CTA button should be bright green ~#22c55e\n"
            "[+] TYPOGRAPHY — body text appears ~10px; reference shows ~16px\n"
            "[+] SPACING — no vertical padding between sections; reference shows ~80px per section\n\n"
            "Current render: page is now full-width with a dark background. Header nav is present. "
            "Hero section present with CTA button. Feature cards section is missing. Footer missing. "
            "Text size looks correct. Section padding is present. Nav color is dark grey (~#374151) "
            "instead of dark navy."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "TODO LIST:\n"
            "[✓] LAYOUT — page content is narrow (~500px centered); reference is full-width (~1200px, 100vw)\n"
            "[ ] LAYOUT — content is cut off on the right edge\n"
            "[✓] STRUCTURE — header/navigation bar is absent\n"
            "[✓] STRUCTURE — hero section is absent\n"
            "[ ] STRUCTURE — three-column feature cards section is missing\n"
            "[ ] STRUCTURE — footer section is absent\n"
            "[✓] COLOR — page background is white #ffffff; reference shows dark navy ~#0f172a\n"
            "[✓] COLOR — CTA button should be bright green ~#22c55e\n"
            "[✓] TYPOGRAPHY — body text appears ~10px; reference shows ~16px\n"
            "[✓] SPACING — no vertical padding between sections; reference shows ~80px per section\n"
            "[+] COLOR — nav bar is dark grey ~#374151; reference shows dark navy ~#0f172a\n\n"
            "STATUS: OPEN (4 items remaining)"
        ),
    },
    {
        "role": "user",
        "content": (
            "[FORMAT EXAMPLE step 3] All 4 remaining items from step 2 are now fixed in the current render."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "TODO LIST:\n"
            "[✓] LAYOUT — page content is narrow (~500px centered); reference is full-width (~1200px, 100vw)\n"
            "[✓] LAYOUT — content is cut off on the right edge\n"
            "[✓] STRUCTURE — header/navigation bar is absent\n"
            "[✓] STRUCTURE — hero section is absent\n"
            "[✓] STRUCTURE — three-column feature cards section is missing\n"
            "[✓] STRUCTURE — footer section is absent\n"
            "[✓] COLOR — page background is white #ffffff; reference shows dark navy ~#0f172a\n"
            "[✓] COLOR — CTA button should be bright green ~#22c55e\n"
            "[✓] TYPOGRAPHY — body text appears ~10px; reference shows ~16px\n"
            "[✓] SPACING — no vertical padding between sections; reference shows ~80px per section\n"
            "[✓] COLOR — nav bar is dark grey ~#374151; reference shows dark navy ~#0f172a\n\n"
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
            return (
                "The Critic found no remaining issues from its checklist. "
                "Look carefully at the reference screenshot for any fine details "
                "(spacing, colors, text content, missing elements) and refine your HTML."
            )
        lines = ["Fix the following issues (Critic feedback):"]
        for item in pending:
            lines.append(f"- {item.text}")
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> "TodoList":
        """Parse a TODO list from Critic output text.

        [+] items (newly discovered) are always kept pending — they can't be
        resolved the same step they were first identified.
        Duplicate items (same text) are silently dropped.
        """
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
            key = item_text.lower()
            if key not in seen:
                seen.add(key)
                result.items.append(TodoItem(text=item_text, done=done))
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
    is_first = prev_todo is None

    if dbg:
        prev_critique_text = prev_todo.format_for_developer() if prev_todo else None
        dbg.log_critic_input(ref_b64, render_prev_b64, prev_critique_text, render_curr_b64)

    system = FIRST_CRITIC_SYSTEM if is_first else SUBSEQUENT_CRITIC_SYSTEM
    oneshot = _FIRST_CRITIC_ONESHOT_MESSAGES if is_first else _SUBSEQUENT_CRITIC_ONESHOT_MESSAGES

    critic_messages = [{"role": "system", "content": system}]
    if ONE_SHOT:
        critic_messages.extend(oneshot)

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

    if is_first:
        content.append({
            "type": "text",
            "text": (
                "\nThis is the first review. Perform a comprehensive visual audit covering "
                "LAYOUT, STRUCTURE, COLOR, TYPOGRAPHY, SPACING, and TEXT dimensions. "
                "Output your initial TODO LIST with [+] items only."
            ),
        })
    else:
        content.append({
            "type": "text",
            "text": (
                f"\n{prev_todo.format_for_critic()}\n\n"
                "Update the TODO list based on what you see in the CURRENT RENDER. "
                "Mark fixed items [✓], keep unresolved items [ ], add new issues with [+]. "
                "Output STATUS: DONE only when every item is [✓]."
            ),
        })

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
    - TodoList.all_done() (Critic verified all items resolved)

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
        if todo is not None and todo.all_done():
            print(f"[CRITIC] All TODO items resolved at step={step_n} reward={reward:.2f} — stopping.", flush=True)
            break

    return results
