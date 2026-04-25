"""OpenEnv data models for the VisionCoder environment."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """An action submitted by the agent: raw HTML code to evaluate."""

    html: str
    session_id: Optional[str] = None  # required for multi-step episodes


class Observation(BaseModel):
    """The environment's response after reset() or step().

    On reset(): screenshot_b64 and session_id are populated, reward is None, done is False.
    On step():  reward, render_low, render_full are populated; done is True at max steps.
    """

    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    screenshot_b64: Optional[str] = None   # target screenshot (after reset)
    render_low: Optional[str] = None       # low-res render of submitted HTML (after step)
    render_full: Optional[str] = None      # full-res render of submitted HTML (after step)
    session_id: Optional[str] = None       # session identifier (after reset)
    prompt: str = ""


class State(BaseModel):
    """Episode metadata tracked by the environment."""

    episode_id: str = ""
    session_id: str = ""
    step_count: int = 0
    sample_index: int = 0
    max_steps: int = 0


class RenderRequest(BaseModel):
    """Request body for the /render endpoint."""

    html: str


class RenderResponse(BaseModel):
    """Response from the /render endpoint."""

    image_b64: str        # full-res PNG (base64)
    image_low_b64: str    # low-res PNG (base64)
