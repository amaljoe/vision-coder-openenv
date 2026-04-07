"""OpenEnv data models for the VisionCoder environment.

Defines the Action, Observation, and State types used by both
the server-side Environment and the client-side HTTPEnvClient.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """An action submitted by the agent: raw HTML code to evaluate."""

    html: str


class Observation(BaseModel):
    """The environment's response after reset() or step().

    On reset(): screenshot_b64 is populated, reward is None, done is False.
    On step():  reward is populated, done is True, screenshot_b64 is None.
    """

    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Base64-encoded PNG of the target screenshot (present after reset)
    screenshot_b64: Optional[str] = None
    # Instruction shown to the agent
    prompt: str = ""


class State(BaseModel):
    """Episode metadata tracked by the environment."""

    episode_id: str = ""
    step_count: int = 0
    sample_index: int = 0
