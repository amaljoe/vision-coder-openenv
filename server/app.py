"""FastAPI server exposing the VisionCoderEnvironment via the OpenEnv HTTP API.

Endpoints:
  POST /reset       → Observation  (start new episode; ?difficulty=easy|medium|hard|mixed)
  POST /step        → Observation  (submit HTML action)
  GET  /state       → State        (current episode metadata)
  DELETE /close     → 204          (signal end of session)
  GET  /health      → 200          (liveness probe for HF Spaces)

Run with:
  uvicorn openenv.server.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from openenv.models import Action, Observation, State
from openenv.server.environment import VisionCoderEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VisionCoder OpenEnv",
    description="Screenshot-to-HTML reinforcement learning environment (OpenEnv-compatible)",
    version="0.1.0",
)

_env = VisionCoderEnvironment()


@app.get("/health")
def health():
    """Liveness probe — returns 200 OK."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(
    difficulty: str = Query(
        default="mixed",
        description="Task difficulty: easy | medium | hard | mixed",
    )
) -> Observation:
    """Start a new episode. Returns an Observation containing the target screenshot."""
    if difficulty not in ("easy", "medium", "hard", "mixed"):
        raise HTTPException(status_code=422, detail=f"Invalid difficulty: {difficulty!r}")
    try:
        obs = _env.reset(difficulty=difficulty)
        logger.info(
            "Episode %s started — difficulty=%s sample=%d",
            obs.metadata.get("episode_id"),
            difficulty,
            obs.metadata.get("sample_index", 0),
        )
        return obs
    except Exception as exc:
        logger.exception("reset() failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=Observation)
def step(action: Action) -> Observation:
    """Submit HTML code. Returns an Observation with reward in [0.0, 1.0] and done=True."""
    try:
        obs = _env.step(action)
        rewards = obs.metadata.get("rewards", {})
        logger.info(
            "Episode %s step %d — total=%.4f (fmt=%.2f val=%.2f struct=%.2f clip=%.2f)",
            obs.metadata.get("episode_id"),
            obs.metadata.get("step_count", 1),
            rewards.get("total", 0.0),
            rewards.get("format", 0.0),
            rewards.get("validity", 0.0),
            rewards.get("structural", 0.0),
            rewards.get("clip", 0.0),
        )
        return obs
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("step() failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=State)
def state() -> State:
    """Return current episode metadata."""
    return _env.state


@app.delete("/close", status_code=204)
def close() -> None:
    """Signal end of session (no-op for single-instance server)."""
    logger.info("Session closed by client.")


def main() -> None:
    """Entry point for the `server` console script."""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("openenv.server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()


def main():
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
