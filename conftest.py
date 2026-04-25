"""Root conftest: add src/ to sys.path and exclude root __init__.py from test collection."""
import sys
from pathlib import Path

_root = Path(__file__).parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

repo_root = str(_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

collect_ignore = ["__init__.py"]
