"""
Auto-discovery and factory for front-end session adapters.

Discovers adapter classes by scanning for frontend_adapter_*.py modules in the
same directory. A class qualifies if it has an ADAPTER_NAME class attribute.
The ADAPTER_NAME value is the canonical name used in config.

Per-adapter config lives in cfg["adapters"][name]. Reserved keys:
  "module" - explicit dotted module path (overrides discovery)
  "class"  - explicit class name within the module

All other keys in the per-adapter config dict are passed as **kwargs to the
adapter constructor.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path


def discover_frontend_adapters(search_dir: Path | None = None) -> dict[str, type]:
    """
    Scan search_dir for frontend_adapter_*.py files and return a mapping
    of adapter name -> adapter class.

    Uses the ADAPTER_NAME class attribute as the canonical name. Falls back to
    the filename suffix (the part after the last underscore, without .py) if no
    ADAPTER_NAME is defined on any class in the module.
    """
    if search_dir is None:
        search_dir = Path(__file__).parent

    found: dict[str, type] = {}
    for path in sorted(search_dir.glob("frontend_adapter_*.py")):
        stem = path.stem  # e.g. "frontend_adapter_continue"
        try:
            module = importlib.import_module(stem)
        except Exception:
            continue
        for obj in vars(module).values():
            if not inspect.isclass(obj):
                continue
            if obj.__module__ != module.__name__:
                # Skip classes re-exported from other modules.
                continue
            adapter_name = getattr(obj, "ADAPTER_NAME", None)
            if adapter_name is None:
                # Fall back: frontend_adapter_continue -> "continue"
                parts = stem.split("_")
                adapter_name = parts[-1] if parts else stem
            found[adapter_name] = obj
    return found


def build_active_frontend_adapters(cfg: dict) -> list:
    """
    Build and return all adapters listed in cfg["active_adapters"].

    For each name in active_adapters:
    - Looks up the class via discovery, or via explicit module/class config keys.
    - Instantiates with **kwargs from cfg["adapters"][name] (minus reserved
      keys "module" and "class").

    Raises ValueError for any name not resolvable via discovery or explicit config.
    """
    active_names: list[str] = cfg.get("active_adapters", ["continue"])
    adapters_cfg: dict[str, dict] = cfg.get("adapters", {})

    discovered = discover_frontend_adapters()
    result = []
    for name in active_names:
        per_cfg = dict(adapters_cfg.get(name, {}))

        if "module" in per_cfg:
            module = importlib.import_module(per_cfg.pop("module"))
            cls_name = per_cfg.pop("class", None)
            if cls_name:
                cls = getattr(module, cls_name)
            else:
                cls = None
                for obj in vars(module).values():
                    if inspect.isclass(obj) and getattr(obj, "ADAPTER_NAME", None) == name:
                        cls = obj
                        break
                if cls is None:
                    raise ValueError(
                        f"No class with ADAPTER_NAME={name!r} found in module "
                        f"{module.__name__!r}."
                    )
        else:
            per_cfg.pop("class", None)  # strip stray 'class' key
            if name not in discovered:
                available = sorted(discovered)
                raise ValueError(
                    f"Unknown frontend adapter: {name!r}. "
                    f"Available: {available!r}. "
                    f"Add frontend_adapter_{name}.py or use explicit "
                    f"'module'/'class' keys in cfg['adapters'][{name!r}]."
                )
            cls = discovered[name]

        result.append(cls(**per_cfg))
    return result
