"""Root conftest: prevent pytest from importing the root __init__.py as a test module.

The root __init__.py maps to the `openenv` package alias (see pyproject.toml) and
requires openenv.client which is not available in all environments. Tests live in
tests/ and are self-contained — they add the repo root to sys.path via
tests/conftest.py.
"""
import sys
from pathlib import Path

# Make vcoder/ and server/ importable without going through the root __init__.py
repo_root = str(Path(__file__).parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Tell pytest to skip the root __init__.py
collect_ignore = ["__init__.py"]
