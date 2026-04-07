"""HTTPEnvClient for the VisionCoder OpenEnv environment.

Provides a synchronous client for interacting with the VisionCoder
FastAPI server via the standard OpenEnv HTTP interface.

Usage:
    from openenv.client import VisionCoderClient

    with VisionCoderClient("http://localhost:8080") as client:
        obs = client.reset()
        # decode screenshot, run model inference ...
        action = Action(html="<html>...</html>")
        result = client.step(action)
        print(f"Reward: {result.reward}")
        print(f"Breakdown: {result.metadata['rewards']}")
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import httpx
from PIL import Image

from openenv.models import Action, Observation, State


class VisionCoderClient:
    """Synchronous HTTP client for the VisionCoder OpenEnv server.

    Implements the standard OpenEnv interface: reset(), step(), state(), close().
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 120.0):
        """
        Args:
            base_url: URL of the running VisionCoder OpenEnv server.
            timeout:  Request timeout in seconds (rendering HTML can be slow).
        """
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a new episode. Returns an Observation with the target screenshot."""
        resp = self._client.post("/reset")
        resp.raise_for_status()
        return Observation(**resp.json())

    def step(self, action: Action) -> Observation:
        """Submit HTML code for evaluation. Returns reward and done=True."""
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return Observation(**resp.json())

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

    def decode_screenshot(self, obs: Observation) -> Optional[Image.Image]:
        """Decode the base64 PNG screenshot from an Observation into a PIL Image."""
        if obs.screenshot_b64 is None:
            return None
        raw = base64.b64decode(obs.screenshot_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> VisionCoderClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
