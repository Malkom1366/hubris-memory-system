"""daemon_discovery.py - Dynamic daemon spec discovery for HuBrIS.

Scans for daemon_*.py files in the HuBrIS directory.  Daemons registered in
manifest.json appear first (in manifest order) with their curated metadata.
Any daemon_*.py file not listed in the manifest is auto-discovered and appended
in alphabetical order with default metadata derived from the module docstring.

daemon_base.py is excluded - it is a shared base class, not a runnable daemon.
"""

import ast
import json
from pathlib import Path

from log import get_logger

_log = get_logger("hubris.daemon_discovery")

_DAEMON_DIR = Path(__file__).parent
_MANIFEST_PATH = _DAEMON_DIR / "manifest.json"


def _extract_description(path: Path) -> str:
    """Extract a short description from the first line of the module docstring.

    Docstrings in daemon files follow the convention:
        daemon_X.py - HuBrIS <short description>.

    Returns the part after " - ", or an empty string on any failure.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        if docstring:
            first_line = docstring.splitlines()[0].strip()
            if " - " in first_line:
                return first_line.split(" - ", 1)[1]
    except Exception as exc:
        _log.debug("daemon_discovery: could not extract docstring from %s: %s", path.name, exc)
    return ""


def discover_daemon_specs() -> list[dict]:
    """Return the full list of daemon specs, merged from manifest + file discovery.

    Manifest-registered daemons appear first in their declared order with their
    curated descriptions, restart policies, and health-check intervals.  Any
    daemon_*.py file present on disk but absent from the manifest is appended in
    alphabetical order with:
      - description extracted from the module docstring
      - restart: "always"
      - health_check_interval_s: 5

    This means adding a new daemon_*.py file is sufficient to make it appear in
    the startup dialog and be managed by the supervisor without editing the
    manifest.
    """
    # Load manifest entries (ordered)
    try:
        with open(_MANIFEST_PATH, encoding="utf-8") as fh:
            manifest_list: list[dict] = json.load(fh)["daemons"]
        manifest_names: set[str] = {s["name"] for s in manifest_list}
    except Exception as exc:
        _log.warning("daemon_discovery: could not read manifest: %s", exc)
        manifest_list = []
        manifest_names = set()

    specs: list[dict] = list(manifest_list)

    # Append any daemon_*.py files not yet in the manifest
    for path in sorted(_DAEMON_DIR.glob("daemon_*.py")):
        module = path.stem          # e.g. "daemon_embed"
        name = module[len("daemon_"):]  # e.g. "embed"
        if name == "base":
            continue
        if name not in manifest_names:
            _log.info(
                "daemon_discovery: auto-discovered %s (not in manifest)", module
            )
            specs.append({
                "name": name,
                "module": module,
                "description": _extract_description(path),
                "restart": "always",
                "health_check_interval_s": 5,
            })

    specs.sort(key=lambda s: s["name"])
    return specs
