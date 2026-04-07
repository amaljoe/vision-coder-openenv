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
from vcoder.data.websight import load_websight_dataset
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.structural_rewards import structural_similarity_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.rewards.visual_rewards import clip_visual_reward

PROMPT = (
    "You are a UI-to-code assistant. Given a screenshot of a website, generate the "
    "complete HTML code with inline CSS that reproduces the visual layout. "
    "Output only the raw HTML — no markdown fencing."
)

# Reward signal weights (matches training configuration)
REWARD_WEIGHTS = {
    "format": 1.0,
    "validity": 1.0,
    "structural": 1.0,
    "clip": 3.0,
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
    """

    def __init__(self, max_samples: int = 2000):
        self._max_samples = max_samples
        self._dataset = None
        self._dataset_index: int = 0
        self._current_sample: Optional[dict] = None
        self._state = State()

    def _ensure_dataset(self) -> None:
        if self._dataset is None:
            self._dataset = load_websight_dataset(max_samples=self._max_samples)

    def reset(self) -> Observation:
        """Start a new episode by sampling the next WebSight screenshot.

        Returns an Observation with the target screenshot encoded as base64 PNG.
        """
        self._ensure_dataset()
        self._state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            sample_index=self._dataset_index,
        )
        self._current_sample = self._dataset[self._dataset_index]
        self._dataset_index = (self._dataset_index + 1) % len(self._dataset)

        return Observation(
            done=False,
            reward=None,
            screenshot_b64=_image_to_b64(self._current_sample["image"]),
            prompt=PROMPT,
            metadata={
                "episode_id": self._state.episode_id,
                "sample_index": self._state.sample_index,
            },
        )

    def step(self, action: Action) -> Observation:
        """Score the agent's submitted HTML against the reference screenshot.

        Computes four reward signals and returns their weighted sum:
          - format   (1×): markdown fencing + html/doctype tags
          - validity (1×): parseability + structure + tag diversity
          - structural (1×): DOM tag-sequence + CSS-class overlap
          - clip     (3×): CLIP image-image similarity after rendering

        Returns an Observation with done=True and the composite reward.
        """
        if self._current_sample is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1

        # Wrap in fenced block so reward functions can extract and score properly
        completion_text = f"```html\n{action.html}\n```"
        completions = [[{"content": completion_text}]]
        images = [self._current_sample["image"]]
        solutions = [self._current_sample["solution"]]

        fmt = format_reward(completions)[0]
        val = html_validity_reward(completions)[0]
        struct = structural_similarity_reward(completions, solution=solutions)[0]
        clip = clip_visual_reward(completions, image=images)[0]

        total = (
            REWARD_WEIGHTS["format"] * fmt
            + REWARD_WEIGHTS["validity"] * val
            + REWARD_WEIGHTS["structural"] * struct
            + REWARD_WEIGHTS["clip"] * clip
        )

        return Observation(
            done=True,
            reward=total,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
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
