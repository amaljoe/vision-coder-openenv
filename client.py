"""HTTPEnvClient for the VisionCoder OpenEnv environment."""
from __future__ import annotations

import base64
import io
from typing import Optional

import httpx
from PIL import Image

from openenv.models import Action, Observation, RenderRequest, RenderResponse, State


class VisionCoderClient:
    """Synchronous HTTP client for the VisionCoder OpenEnv server."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "mixed") -> Observation:
        """Start a new episode. Returns Observation with session_id and reference screenshot."""
        resp = self._client.post("/reset", params={"difficulty": difficulty})
        resp.raise_for_status()
        return Observation(**resp.json())

    def step(self, action: Action) -> Observation:
        """Submit HTML. Returns reward, render_low, render_full, done."""
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return Observation(**resp.json())

    def render(self, html: str) -> RenderResponse:
        """Render HTML to images without scoring (Developer tool call)."""
        resp = self._client.post("/render", json=RenderRequest(html=html).model_dump())
        resp.raise_for_status()
        return RenderResponse(**resp.json())

    def state(self) -> State:
        """Return current episode metadata from the server."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return State(**resp.json())

    def close(self) -> None:
        """Signal session end and release HTTP resources."""
        try:
            self._client.delete("/close")
        finally:
            self._client.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def decode_image(self, b64: Optional[str]) -> Optional[Image.Image]:
        """Decode a base64 PNG string into a PIL Image."""
        if b64 is None:
            return None
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def decode_screenshot(self, obs: Observation) -> Optional[Image.Image]:
        """Decode the reference screenshot from a reset() Observation."""
        return self.decode_image(obs.screenshot_b64)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> VisionCoderClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
