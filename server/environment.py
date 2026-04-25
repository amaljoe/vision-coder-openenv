"""VisionCoder OpenEnv Environment — multi-step, session-aware."""
from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from PIL import Image

from openenv.models import Action, Observation, RenderRequest, RenderResponse, State
from vcoder.data.dataset import load_websight_dataset
from vcoder.rewards.color_rewards import color_reward
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.position_rewards import position_reward
from vcoder.rewards.ssim_reward import ssim_reward
from vcoder.rewards.structural_rewards import structural_similarity_reward
from vcoder.rewards.text_block_rewards import text_block_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.rewards import extract_html
from vcoder.rewards.visual_rewards import _render_html, clip_visual_reward

MAX_STEPS = 5  # max developer turns per episode

REWARD_WEIGHTS = {
    "format":     0.5,   # was 1.0 — saturates to 1.0 after early training; reduce weight
    "validity":   0.5,   # was 1.0 — saturates quickly; reduce weight
    "structural": 0.5,   # unchanged — inflated by inline-style refs
    "text_block": 3.0,   # unchanged — most discriminative, blank/wrong layout → 0
    "position":   1.0,   # unchanged
    "color":      1.5,   # was 1.0 — increased for near-perfect sensitivity
    "clip":       2.5,   # was 2.0 — most continuous signal at top, increase
    "ssim":       1.5,   # new — pixel-level SSIM, fills variance gap in 0.7-0.97 zone
}
_WEIGHT_SUM = sum(REWARD_WEIGHTS.values())  # 11.0

LOW_RES = (320, 240)   # developer self-check render
FULL_RES = (640, 480)  # critic + reward computation render

DIFFICULTY_PROMPTS = {
    "easy": (
        "You are a UI-to-code assistant. Given a screenshot of a simple website, "
        "generate complete HTML with inline CSS. Output only raw HTML."
    ),
    "medium": (
        "You are a UI-to-code assistant. Given a screenshot of a website with navigation "
        "and multiple sections, generate complete HTML with inline CSS. Output only raw HTML."
    ),
    "hard": (
        "You are a UI-to-code assistant. Given a screenshot of a complex website with forms, "
        "tables, and rich layout, generate complete HTML with inline CSS. Output only raw HTML."
    ),
}
_DEFAULT_PROMPT = DIFFICULTY_PROMPTS["medium"]


def _image_to_b64(image: Image.Image, size: Optional[tuple] = None) -> str:
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class _Session:
    episode_id: str
    session_id: str
    difficulty: str
    sample: dict
    ref_image: Image.Image
    step_count: int = 0
    sample_index: int = 0


class VisionCoderEnvironment:
    """Multi-step, session-aware OpenEnv environment for screenshot-to-HTML generation.

    Each reset() creates an independent session identified by session_id.
    step() accepts session_id in the Action and allows up to MAX_STEPS turns
    per episode before returning done=True.

    step() returns render_low and render_full (base64 PNG) alongside the reward
    so the Developer agent can inspect its render without an extra /render call.
    """

    def __init__(self, max_samples: int = 2000):
        self._max_samples = max_samples
        self._datasets: Dict[str, list] = {}
        self._dataset_indices: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "mixed": 0}
        self._sessions: Dict[str, _Session] = {}
        self._last_session_id: Optional[str] = None  # backward-compat fallback

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _get_dataset(self, difficulty: str) -> list:
        key = difficulty if difficulty in ("easy", "medium", "hard") else "mixed"
        if key not in self._datasets:
            self._datasets[key] = load_websight_dataset(
                max_samples=self._max_samples,
                difficulty=key if key != "mixed" else None,
            )
        return self._datasets[key]

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "mixed") -> Observation:
        """Start a new episode. Returns session_id and the reference screenshot."""
        dataset = self._get_dataset(difficulty)
        key = difficulty if difficulty in ("easy", "medium", "hard") else "mixed"

        idx = self._dataset_indices[key]
        sample = dataset[idx]
        self._dataset_indices[key] = (idx + 1) % len(dataset)

        session_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())

        ref_image = _render_html(sample["solution"])
        if ref_image is None:
            ref_image = Image.new("RGB", FULL_RES, color=(255, 255, 255))

        session = _Session(
            episode_id=episode_id,
            session_id=session_id,
            difficulty=difficulty,
            sample={**sample, "image": ref_image},
            ref_image=ref_image,
            sample_index=idx,
        )
        self._sessions[session_id] = session
        self._last_session_id = session_id

        return Observation(
            done=False,
            session_id=session_id,
            screenshot_b64=_image_to_b64(ref_image),
            prompt=DIFFICULTY_PROMPTS.get(difficulty, _DEFAULT_PROMPT),
            metadata={
                "episode_id": episode_id,
                "session_id": session_id,
                "sample_index": idx,
                "difficulty": difficulty,
                "max_steps": MAX_STEPS,
            },
        )

    def step(self, action: Action) -> Observation:
        """Score submitted HTML and return reward + rendered images.

        Uses action.session_id to look up the episode. Falls back to the most
        recently created session when session_id is omitted (single-client compat).

        Returns done=True when step_count reaches MAX_STEPS.
        """
        session_id = action.session_id or self._last_session_id
        if session_id is None or session_id not in self._sessions:
            raise RuntimeError("No active session. Call reset() first.")

        session = self._sessions[session_id]
        session.step_count += 1
        done = session.step_count >= MAX_STEPS

        completions = [[{"content": action.html}]]
        images = [session.ref_image]
        solutions = [session.sample["solution"]]

        fmt   = format_reward(completions)[0]
        val   = html_validity_reward(completions)[0]
        struct = structural_similarity_reward(completions, solution=solutions)[0]
        tb    = text_block_reward(completions, solution=solutions)[0]
        pos   = position_reward(completions, solution=solutions)[0]

        ref_w, ref_h = session.ref_image.size
        pred_render = _render_html(extract_html(action.html), width=ref_w, height=ref_h)
        if pred_render is None:
            pred_render = Image.new("RGB", (ref_w, ref_h), color=(255, 255, 255))
        pred_renders = [pred_render]

        col  = color_reward(completions, image=images, pred_image=pred_renders)[0]
        clip = clip_visual_reward(completions, image=images, pred_image=pred_renders)[0]
        ssim = ssim_reward(completions, image=images, pred_image=pred_renders)[0]

        raw_total = (
            REWARD_WEIGHTS["format"]     * fmt
            + REWARD_WEIGHTS["validity"]   * val
            + REWARD_WEIGHTS["structural"] * struct
            + REWARD_WEIGHTS["text_block"] * tb
            + REWARD_WEIGHTS["position"]   * pos
            + REWARD_WEIGHTS["color"]      * col
            + REWARD_WEIGHTS["clip"]       * clip
            + REWARD_WEIGHTS["ssim"]       * ssim
        )
        total = raw_total / _WEIGHT_SUM

        return Observation(
            done=done,
            reward=total,
            session_id=session_id,
            render_low=_image_to_b64(pred_render, size=LOW_RES),
            render_full=_image_to_b64(pred_render),
            metadata={
                "episode_id": session.episode_id,
                "session_id": session_id,
                "step_count": session.step_count,
                "difficulty": session.difficulty,
                "max_steps": MAX_STEPS,
                "rewards": {
                    "format": fmt,
                    "validity": val,
                    "structural": struct,
                    "text_block": tb,
                    "position": pos,
                    "color": col,
                    "clip": clip,
                    "ssim": ssim,
                    "total": total,
                },
            },
        )

    def render(self, request: RenderRequest) -> RenderResponse:
        """Render HTML to images without computing rewards.

        Used by the Developer agent's render() tool call to self-check
        mid-generation without consuming an episode step.
        """
        image = _render_html(extract_html(request.html))
        if image is None:
            image = Image.new("RGB", FULL_RES, color=(255, 255, 255))
        return RenderResponse(
            image_b64=_image_to_b64(image),
            image_low_b64=_image_to_b64(image, size=LOW_RES),
        )

    @property
    def state(self) -> State:
        """Return metadata for the most recently created session."""
        if self._last_session_id and self._last_session_id in self._sessions:
            s = self._sessions[self._last_session_id]
            return State(
                episode_id=s.episode_id,
                session_id=s.session_id,
                step_count=s.step_count,
                sample_index=s.sample_index,
                max_steps=MAX_STEPS,
            )
        return State()
