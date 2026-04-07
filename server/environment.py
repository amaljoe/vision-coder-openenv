"""VisionCoder OpenEnv Environment implementation.

Wraps the VisionCoder reward pipeline into the standard OpenEnv interface:
  - reset() → Observation  (loads next WebSight screenshot)
  - step(action) → Observation  (scores submitted HTML)
  - state → State  (current episode metadata)
"""
from __future__ import annotations

import base64
import io
import uuid
from typing import Optional

from PIL import Image

from openenv.models import Action, Observation, State
from vcoder.data.dataset import load_websight_dataset
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.structural_rewards import structural_similarity_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.rewards.visual_rewards import clip_visual_reward, _render_html

PROMPT = (
    "You are a UI-to-code assistant. Given a screenshot of a website, generate the "
    "complete HTML code with inline CSS that reproduces the visual layout. "
    "Output only the raw HTML — no markdown fencing."
)

# Reward signal weights (must sum to the normalisation denominator below)
REWARD_WEIGHTS = {
    "format": 1.0,
    "validity": 1.0,
    "structural": 1.0,
    "clip": 3.0,
}
_WEIGHT_SUM = sum(REWARD_WEIGHTS.values())  # 6.0 — used to normalise to [0, 1]

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


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class VisionCoderEnvironment:
    """OpenEnv-compatible environment for screenshot-to-HTML generation.

    Each episode presents an agent with a UI screenshot. The agent submits
    HTML code via step(), which is rendered and scored against the reference
    using four reward signals (format, validity, structural, CLIP visual).

    The composite reward is normalised to [0.0, 1.0].
    """

    def __init__(self, max_samples: int = 2000):
        self._max_samples = max_samples
        self._datasets: dict[str, list] = {}
        self._dataset_indices: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "mixed": 0}
        self._current_sample: Optional[dict] = None
        self._state = State()
        self._current_difficulty: str = "mixed"

    def _get_dataset(self, difficulty: str) -> list:
        key = difficulty if difficulty in ("easy", "medium", "hard") else "mixed"
        if key not in self._datasets:
            diff_arg = key if key != "mixed" else None
            self._datasets[key] = load_websight_dataset(
                max_samples=self._max_samples,
                difficulty=diff_arg,
            )
        return self._datasets[key]

    def reset(self, difficulty: str = "mixed") -> Observation:
        """Start a new episode by sampling the next WebSight screenshot.

        Args:
            difficulty: Task difficulty — "easy", "medium", "hard", or "mixed".

        Returns:
            Observation with the target screenshot encoded as base64 PNG.
        """
        self._current_difficulty = difficulty
        dataset = self._get_dataset(difficulty)
        key = difficulty if difficulty in ("easy", "medium", "hard") else "mixed"

        idx = self._dataset_indices[key]
        self._current_sample = dataset[idx]
        self._dataset_indices[key] = (idx + 1) % len(dataset)

        self._state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            sample_index=idx,
        )

        # Render reference HTML live to get the screenshot
        ref_image = _render_html(self._current_sample["solution"])
        if ref_image is None:
            # Fallback: blank white image
            ref_image = Image.new("RGB", (640, 480), color=(255, 255, 255))
        self._current_sample["image"] = ref_image

        prompt = DIFFICULTY_PROMPTS.get(difficulty, PROMPT)

        return Observation(
            done=False,
            reward=None,
            screenshot_b64=_image_to_b64(ref_image),
            prompt=prompt,
            metadata={
                "episode_id": self._state.episode_id,
                "sample_index": self._state.sample_index,
                "difficulty": difficulty,
            },
        )

    def step(self, action: Action) -> Observation:
        """Score the agent's submitted HTML against the reference screenshot.

        Computes four reward signals and returns their weighted sum normalised
        to [0.0, 1.0]:
          - format      (weight 1): markdown fencing + html/doctype tags
          - validity    (weight 1): parseability + structure + tag diversity
          - structural  (weight 1): DOM tag-sequence + CSS-class overlap
          - clip        (weight 3): CLIP image-image similarity after rendering

        Returns:
            Observation with done=True and reward in [0.0, 1.0].
        """
        if self._current_sample is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1

        # Pass raw action directly — extract_html handles fences/think blocks
        completions = [[{"content": action.html}]]
        images = [self._current_sample["image"]]
        solutions = [self._current_sample["solution"]]

        fmt = format_reward(completions)[0]
        val = html_validity_reward(completions)[0]
        struct = structural_similarity_reward(completions, solution=solutions)[0]
        clip = clip_visual_reward(completions, image=images)[0]

        raw_total = (
            REWARD_WEIGHTS["format"] * fmt
            + REWARD_WEIGHTS["validity"] * val
            + REWARD_WEIGHTS["structural"] * struct
            + REWARD_WEIGHTS["clip"] * clip
        )
        # Normalise to [0, 1]
        total = raw_total / _WEIGHT_SUM

        return Observation(
            done=True,
            reward=total,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "difficulty": self._current_difficulty,
                "rewards": {
                    "format": fmt,
                    "validity": val,
                    "structural": struct,
                    "clip": clip,
                    "total": total,
                },
            },
        )

    @property
    def state(self) -> State:
        return self._state
