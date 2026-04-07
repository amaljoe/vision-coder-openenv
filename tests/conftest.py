"""Pytest configuration: add repo root to sys.path so vcoder/ and server/ are importable."""
import sys
from pathlib import Path

# Insert repo root so that `vcoder`, `server`, etc. resolve correctly
# without going through the root __init__.py (which needs openenv.client).
repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
