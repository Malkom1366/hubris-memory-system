"""Startup configuration dialog for HuBrIS.

Provides show(cfg) -> dict | None | bool.

- Returns a new config dict if the user clicked Save & Load.
- Returns False if the user clicked Save (saved to disk) or Shut Down.
- Returns None if tkinter is unavailable (headless guard).
- Safe to call in headless environments: logs a warning and returns None.

The dialog is tabbed:
  Backend   - adapter, endpoint, API key, timeout, model selection
  Daemons   - enable/disable individual background daemons
  Frontend  - active session adapters and their paths
  Sessions  - blacklist pre-existing sessions on first run
  Advanced  - workspace ID, token limits, polling, recency weights
"""

import json
from pathlib import Path

from daemons import discover_daemon_specs
from log import get_logger

_log = get_logger("hubris.config_ui")

# Check for sqlite_vec at import time so the embed model section can be
# conditionally rendered without an additional import inside show().
_SQLITE_VEC_AVAILABLE: bool
try:
    import sqlite_vec as _  # noqa: F401
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


def show(cfg: dict) -> dict | None:
    """
    Display the startup configuration dialog and return the updated config.

    Returns the new config dict when the user clicks Save & Load.
    Returns False when the user clicks Save (config written to disk, no launch),
    Shut Down, or closes the window with X.
    Returns None immediately (without opening a window) if tkinter is not
    available in the current environment.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        _log.warning("config_ui: tkinter not available - skipping startup dialog")
        return None

    import db as _db
    from backend_adapters import _REGISTRY

    result: dict | None = None
    _shutdown: list[bool] = [False]  # mutable cell for nested function assignment

    # Last model list fetched from the backend. Shared across dropdowns.
    _models: list[str] = []

    # ----------------------------------------------------------------
    # Root window
    # ----------------------------------------------------------------
    root = tk.Tk()
    root.title("HuBrIS Startup Configuration")
    root.geometry("740x600")
    root.resizable(True, True)
    root.minsize(580, 480)

    # ----------------------------------------------------------------
    # Notebook
    # ----------------------------------------------------------------
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

    # ================================================================
    # TAB 1: Backend
    # ================================================================
    tab_backend = ttk.Frame(notebook, padding=10)
    notebook.add(tab_backend, text="Backend")
    tab_backend.columnconfigure(1, weight=1)

    # Adapter selection
    adapter_names = sorted(_REGISTRY.keys())
    adapter_var = tk.StringVar(value=cfg.get("backend_adapter", "ollama"))

    ttk.Label(tab_backend, text="Backend Adapter:").grid(
        row=0, column=0, sticky="w", pady=4
    )
    adapter_combo = ttk.Combobox(
        tab_backend,
        textvariable=adapter_var,
        values=adapter_names,
        state="readonly",
        width=20,
    )
    adapter_combo.grid(row=0, column=1, sticky="w", padx=8, pady=4)

    # Connection fields (pre-populated from the currently active adapter's block)
    _bcfg_init = cfg.get("backends", {}).get(adapter_var.get(), {})
    endpoint_var = tk.StringVar(
        value=_bcfg_init.get("endpoint", "http://localhost:11434")
    )
    api_key_var = tk.StringVar(value=_bcfg_init.get("api_key", ""))
    timeout_var = tk.StringVar(value=str(_bcfg_init.get("timeout", 600)))

    ttk.Label(tab_backend, text="Endpoint URL:").grid(
        row=1, column=0, sticky="w", pady=4
    )
    ttk.Entry(tab_backend, textvariable=endpoint_var, width=45).grid(
        row=1, column=1, columnspan=2, sticky="ew", padx=8, pady=4
    )

    ttk.Label(tab_backend, text="API Key:").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Entry(tab_backend, textvariable=api_key_var, width=45, show="*").grid(
        row=2, column=1, columnspan=2, sticky="ew", padx=8, pady=4
    )

    ttk.Label(tab_backend, text="Timeout (s):").grid(
        row=3, column=0, sticky="w", pady=4
    )
    ttk.Entry(tab_backend, textvariable=timeout_var, width=10).grid(
        row=3, column=1, sticky="w", padx=8, pady=4
    )

    ttk.Separator(tab_backend, orient="horizontal").grid(
        row=4, column=0, columnspan=3, sticky="ew", pady=8
    )

    # Model dropdowns
    meta_model_var = tk.StringVar(value=cfg.get("meta_model", ""))
    subagent_model_var = tk.StringVar(value=cfg.get("subagent_model", ""))
    embed_model_var = tk.StringVar(value=cfg.get("embed_model", ""))

    ttk.Label(tab_backend, text="Meta Model:").grid(
        row=5, column=0, sticky="w", pady=4
    )
    meta_combo = ttk.Combobox(
        tab_backend, textvariable=meta_model_var, values=[], width=35
    )
    meta_combo.grid(row=5, column=1, columnspan=2, sticky="ew", padx=8, pady=4)

    ttk.Label(tab_backend, text="Subagent Model:").grid(
        row=6, column=0, sticky="w", pady=4
    )
    subagent_combo = ttk.Combobox(
        tab_backend, textvariable=subagent_model_var, values=[], width=35
    )
    subagent_combo.grid(row=6, column=1, columnspan=2, sticky="ew", padx=8, pady=4)

    ttk.Separator(tab_backend, orient="horizontal").grid(
        row=7, column=0, columnspan=3, sticky="ew", pady=4
    )

    # Embed model section - shown when sqlite_vec is available, greyed when not
    if _SQLITE_VEC_AVAILABLE:
        ttk.Label(tab_backend, text="Embed Model:").grid(
            row=8, column=0, sticky="w", pady=4
        )
        embed_combo: ttk.Combobox | None = ttk.Combobox(
            tab_backend, textvariable=embed_model_var, values=[], width=35
        )
        embed_combo.grid(row=8, column=1, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Label(
            tab_backend,
            text="(embedding model - used for vector recall with sqlite_vec)",
            foreground="gray",
        ).grid(row=9, column=1, columnspan=2, sticky="w", padx=8)
    else:
        ttk.Label(
            tab_backend, text="Embed Model:", foreground="gray"
        ).grid(row=8, column=0, sticky="w", pady=4)
        ttk.Entry(
            tab_backend, textvariable=embed_model_var, state="disabled", width=35
        ).grid(row=8, column=1, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Label(
            tab_backend,
            text="Install sqlite_vec to enable embedding",
            foreground="gray",
        ).grid(row=9, column=1, columnspan=2, sticky="w", padx=8)
        embed_combo = None

    # Load models button and status
    fetch_btn = ttk.Button(tab_backend, text="Load Models from Endpoint")
    fetch_btn.grid(row=10, column=1, sticky="w", padx=8, pady=8)
    fetch_status = ttk.Label(tab_backend, text="")
    fetch_status.grid(row=10, column=2, sticky="w", padx=4)

    def _update_model_dropdowns(models: list[str]) -> None:
        nonlocal _models
        _models = models
        meta_combo["values"] = models
        subagent_combo["values"] = models
        if embed_combo is not None:
            embed_combo["values"] = models

    def _fetch_models_blocking() -> None:
        """Show a blocking loading dialog, call list_models(), then populate dropdowns."""
        name = adapter_var.get()
        cls = _REGISTRY.get(name)
        if cls is None:
            return

        loading = tk.Toplevel(root)
        loading.title("Loading")
        loading.geometry("240x80")
        loading.transient(root)
        loading.grab_set()
        loading.resizable(False, False)
        ttk.Label(loading, text="Loading models from endpoint...", padding=20).pack(
            expand=True
        )
        loading.update()

        try:
            timeout_val = float(timeout_var.get() or 600)
        except ValueError:
            timeout_val = 600.0

        adapter_instance = cls(
            endpoint=endpoint_var.get().strip(),
            api_key=api_key_var.get().strip(),
            timeout=timeout_val,
        )
        models = adapter_instance.list_models()
        loading.destroy()
        _update_model_dropdowns(models)

        if models:
            fetch_status.config(text=f"{len(models)} models loaded")
        else:
            fetch_status.config(text="No models found - check endpoint")

    def _on_adapter_changed(event=None) -> None:
        # Refresh connection fields to the new adapter's config block.
        new_bcfg = cfg.get("backends", {}).get(adapter_var.get(), {})
        endpoint_var.set(new_bcfg.get("endpoint", "http://localhost:11434"))
        api_key_var.set(new_bcfg.get("api_key", ""))
        timeout_var.set(str(new_bcfg.get("timeout", 600)))
        _fetch_models_blocking()

    adapter_combo.bind("<<ComboboxSelected>>", _on_adapter_changed)
    fetch_btn.config(command=_fetch_models_blocking)

    # ================================================================
    # TAB 2: Daemons
    # ================================================================
    tab_daemons = ttk.Frame(notebook, padding=10)
    notebook.add(tab_daemons, text="Daemons")

    ttk.Label(
        tab_daemons,
        text="Uncheck daemons to disable them at startup.",
        foreground="gray",
    ).pack(anchor="w", pady=(0, 8))

    daemon_specs = discover_daemon_specs()
    disabled_set = set(cfg.get("disabled_daemons", []))
    daemon_vars: dict[str, tk.BooleanVar] = {}

    for spec in daemon_specs:
        name = spec["name"]
        desc = spec.get("description", "")
        bvar = tk.BooleanVar(value=name not in disabled_set)
        daemon_vars[name] = bvar
        row_frame = ttk.Frame(tab_daemons)
        row_frame.pack(fill="x", pady=3)
        ttk.Checkbutton(row_frame, variable=bvar, text=name, width=14).pack(
            side="left"
        )
        ttk.Label(
            row_frame,
            text=desc,
            foreground="gray",
            wraplength=500,
            justify="left",
        ).pack(side="left", padx=6)

    # ================================================================
    # TAB 3: Frontend
    # ================================================================
    tab_frontend = ttk.Frame(notebook, padding=10)
    notebook.add(tab_frontend, text="Frontend")

    ttk.Label(
        tab_frontend,
        text="Session sources - HuBrIS reads conversation files from these adapters.",
    ).pack(anchor="w", pady=(0, 8))

    _all_adapter_names = ["continue", "ghcp"]
    _active_adapters = cfg.get("active_adapters", ["continue"])
    frontend_vars: dict[str, tk.BooleanVar] = {}

    # Continue
    cont_frame = ttk.LabelFrame(tab_frontend, text="Continue", padding=8)
    cont_frame.pack(fill="x", pady=4)
    frontend_vars["continue"] = tk.BooleanVar(value="continue" in _active_adapters)
    ttk.Checkbutton(
        cont_frame, variable=frontend_vars["continue"], text="Enable Continue adapter"
    ).pack(anchor="w")
    sessions_dir_var = tk.StringVar(
        value=cfg.get("adapters", {}).get("continue", {}).get("sessions_dir", "")
    )
    ttk.Label(cont_frame, text="Sessions directory (blank = auto):").pack(
        anchor="w", pady=(4, 0)
    )
    ttk.Entry(cont_frame, textvariable=sessions_dir_var, width=55).pack(
        fill="x", pady=2
    )

    # GitHub Copilot Chat
    ghcp_frame = ttk.LabelFrame(tab_frontend, text="GitHub Copilot Chat", padding=8)
    ghcp_frame.pack(fill="x", pady=4)
    frontend_vars["ghcp"] = tk.BooleanVar(value="ghcp" in _active_adapters)
    ttk.Checkbutton(
        ghcp_frame,
        variable=frontend_vars["ghcp"],
        text="Enable GitHub Copilot Chat adapter",
    ).pack(anchor="w")
    ws_storage_var = tk.StringVar(
        value=cfg.get("adapters", {}).get("ghcp", {}).get("workspace_storage_dir", "")
    )
    ttk.Label(ghcp_frame, text="workspaceStorage directory (blank = auto):").pack(
        anchor="w", pady=(4, 0)
    )
    ttk.Entry(ghcp_frame, textvariable=ws_storage_var, width=55).pack(
        fill="x", pady=2
    )
    injection_limit_var = tk.StringVar(
        value=str(cfg.get("ghcp_adapter", {}).get("injection_token_limit", 600))
    )
    ttk.Label(ghcp_frame, text="Context injection token limit:").pack(
        anchor="w", pady=(4, 0)
    )
    ttk.Entry(ghcp_frame, textvariable=injection_limit_var, width=10).pack(
        anchor="w", pady=2
    )

    # ================================================================
    # TAB 4: Sessions (select sessions to include in memory indexing)
    # ================================================================
    tab_sessions = ttk.Frame(notebook, padding=10)
    notebook.add(tab_sessions, text="Sessions")

    ttk.Label(
        tab_sessions,
        text="Check sessions to include in memory indexing.",
    ).pack(anchor="w", pady=(0, 2))
    ttk.Label(
        tab_sessions,
        text="Only checked sessions will be processed. Unchecked sessions are never indexed.",
        foreground="gray",
    ).pack(anchor="w", pady=(0, 8))

    sessions_outer = ttk.Frame(tab_sessions)
    sessions_outer.pack(fill="both", expand=True)

    sessions_canvas = tk.Canvas(sessions_outer, highlightthickness=0)
    sessions_scrollbar = ttk.Scrollbar(
        sessions_outer, orient="vertical", command=sessions_canvas.yview
    )
    sessions_inner = ttk.Frame(sessions_canvas)
    sessions_inner.bind(
        "<Configure>",
        lambda e: sessions_canvas.configure(
            scrollregion=sessions_canvas.bbox("all")
        ),
    )
    sessions_canvas.create_window((0, 0), window=sessions_inner, anchor="nw")
    sessions_canvas.configure(yscrollcommand=sessions_scrollbar.set)
    sessions_canvas.pack(side="left", fill="both", expand=True)
    sessions_scrollbar.pack(side="right", fill="y")

    sessions_status = ttk.Label(
        tab_sessions,
        text="Switch to this tab to load sessions",
        foreground="gray",
    )
    sessions_status.pack(anchor="w", pady=4)

    _session_vars: dict[str, tk.BooleanVar] = {}
    _sessions_ever_opened = [False]

    _ADAPTER_DISPLAY: dict[str, str] = {"continue": "Continue", "ghcp": "GHCP"}

    def _reload_sessions() -> None:
        # Preserve checked state across reloads (user-toggled + DB whitelist).
        prev_checked = {sid for sid, bvar in _session_vars.items() if bvar.get()}
        _session_vars.clear()
        for widget in sessions_inner.winfo_children():
            widget.destroy()
        sessions_status.config(text="Loading...")
        root.update()
        try:
            from registry import discover_frontend_adapters

            discovered = discover_frontend_adapters()
            all_sessions: list[tuple[str, str, float]] = []  # (sid, display_label, mtime)
            seen_sids: set[str] = set()
            for adapter_name, adapter_cls in discovered.items():
                # Only include adapters that are currently enabled in the Frontend tab.
                if not frontend_vars.get(adapter_name, tk.BooleanVar(value=True)).get():
                    continue
                host_label = _ADAPTER_DISPLAY.get(adapter_name, adapter_name)
                try:
                    adapter_inst = adapter_cls()
                    for sid in adapter_inst.list_historical_sessions():
                        if sid in seen_sids:
                            continue
                        seen_sids.add(sid)
                        title: str | None = None
                        try:
                            if hasattr(adapter_inst, "get_session_title"):
                                title = adapter_inst.get_session_title(sid)
                        except Exception:
                            pass
                        mtime: float = 0.0
                        try:
                            mtime = adapter_inst.get_mtime(sid)
                        except Exception:
                            pass
                        if title:
                            label = f"{title}  ({sid[:8]})  [{host_label}]"
                        else:
                            label = f"{sid}  [{host_label}]"
                        all_sessions.append((sid, label, mtime))
                except Exception:
                    pass

            # Combine DB whitelist with any in-session checked state.
            try:
                already_whitelisted = _db.load_session_whitelist()
            except Exception:
                already_whitelisted = set()

            all_sorted = sorted(all_sessions, key=lambda x: x[2], reverse=True)
            if not all_sorted:
                sessions_status.config(text="No sessions found.")
                return

            sessions_status.config(
                text=f"{len(all_sorted)} sessions found - check any to watch."
            )
            for sid, label, _mtime in all_sorted:
                bvar = tk.BooleanVar(
                    value=(sid in already_whitelisted or sid in prev_checked)
                )
                _session_vars[sid] = bvar
                ttk.Checkbutton(
                    sessions_inner, variable=bvar, text=label
                ).pack(anchor="w", pady=1)

        except Exception as exc:
            sessions_status.config(text=f"Error loading sessions: {exc}")

    def _on_adapter_toggle(*_args) -> None:
        # Only reload the session list if the Sessions tab has been opened at
        # least once - no point rebuilding widgets the user has never seen.
        if _sessions_ever_opened[0]:
            _reload_sessions()

    for _fvar in frontend_vars.values():
        _fvar.trace_add("write", _on_adapter_toggle)

    def _on_tab_changed(event=None) -> None:
        try:
            current = notebook.tab(notebook.select(), "text")
            if current == "Sessions":
                _sessions_ever_opened[0] = True
                _reload_sessions()
        except Exception:
            pass

    notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)

    # ================================================================
    # TAB 5: Advanced
    # ================================================================
    tab_advanced = ttk.Frame(notebook, padding=10)
    notebook.add(tab_advanced, text="Advanced")
    tab_advanced.columnconfigure(1, weight=1)

    _adv_fields: list[tuple[str, str, type]] = [
        ("workspace_id", "Workspace ID", str),
        ("poll_interval", "Poll interval (s)", float),
        ("max_active_tokens", "Max active tokens", int),
        ("max_recall_tokens", "Max recall tokens", int),
        ("split_subject_threshold", "Split subject threshold", int),
        ("recency_alpha", "Recency alpha", float),
        ("recency_lambda", "Recency lambda", float),
    ]
    adv_vars: dict[str, tk.StringVar] = {}
    for row_idx, (key, label, _typ) in enumerate(_adv_fields):
        adv_vars[key] = tk.StringVar(value=str(cfg.get(key, "")))
        ttk.Label(tab_advanced, text=label + ":").grid(
            row=row_idx, column=0, sticky="w", pady=4
        )
        ttk.Entry(tab_advanced, textvariable=adv_vars[key], width=25).grid(
            row=row_idx, column=1, sticky="w", padx=8, pady=4
        )

    # ================================================================
    # Bottom button bar
    # ================================================================
    btn_frame = ttk.Frame(root, padding=(10, 6, 10, 10))
    btn_frame.pack(fill="x", side="bottom")

    validation_label = ttk.Label(btn_frame, text="", foreground="red")
    validation_label.pack(side="left")

    def _collect_cfg() -> dict | None:
        """Validate and assemble a new config dict from the current UI state."""
        new_cfg = dict(cfg)

        # Backend
        adapter_name = adapter_var.get()
        new_cfg["backend_adapter"] = adapter_name
        new_cfg["meta_model"] = meta_model_var.get().strip()
        new_cfg["subagent_model"] = subagent_model_var.get().strip()
        new_cfg["embed_model"] = embed_model_var.get().strip()

        backends = dict(cfg.get("backends", {}))
        bcfg_entry = dict(backends.get(adapter_name, {}))
        bcfg_entry["endpoint"] = endpoint_var.get().strip()
        bcfg_entry["api_key"] = api_key_var.get().strip()
        try:
            bcfg_entry["timeout"] = float(timeout_var.get() or 600)
        except ValueError:
            validation_label.config(text="Timeout must be a number.")
            return None
        backends[adapter_name] = bcfg_entry
        new_cfg["backends"] = backends

        # Daemons
        new_cfg["disabled_daemons"] = [
            name for name, bvar in daemon_vars.items() if not bvar.get()
        ]

        # Frontend
        new_cfg["active_adapters"] = [
            name for name in _all_adapter_names if frontend_vars[name].get()
        ]
        adapters = dict(cfg.get("adapters", {}))
        adapters["continue"] = dict(adapters.get("continue", {}))
        adapters["continue"]["sessions_dir"] = sessions_dir_var.get().strip()
        adapters["ghcp"] = dict(adapters.get("ghcp", {}))
        adapters["ghcp"]["workspace_storage_dir"] = ws_storage_var.get().strip()
        new_cfg["adapters"] = adapters

        # GHCP adapter context injection settings
        ghcp_adapter_cfg = dict(cfg.get("ghcp_adapter", {}))
        raw_limit = injection_limit_var.get().strip()
        if raw_limit:
            try:
                ghcp_adapter_cfg["injection_token_limit"] = int(raw_limit)
            except ValueError:
                validation_label.config(text="Injection token limit must be an integer.")
                return None
        new_cfg["ghcp_adapter"] = ghcp_adapter_cfg

        # Advanced
        _type_map = {k: t for k, _, t in _adv_fields}
        for key, var in adv_vars.items():
            raw = var.get().strip()
            if not raw:
                continue
            try:
                new_cfg[key] = _type_map[key](raw)
            except (ValueError, TypeError):
                validation_label.config(text=f"Invalid value for '{key}'.")
                return None

        return new_cfg

    def _apply_whitelist() -> None:
        """Persist the session whitelist from checked sessions. Shared by save paths."""
        if _session_vars:
            checked = {sid for sid, bvar in _session_vars.items() if bvar.get()}
            try:
                _db.save_session_whitelist(
                    checked,
                    _db.ACTOR_WHITELIST_MANAGER,
                )
            except Exception as exc:
                _log.warning(
                    "config_ui: failed to save session whitelist: %s", exc
                )

    def on_save_and_load() -> None:
        nonlocal result
        collected = _collect_cfg()
        if collected is None:
            return  # validation failed; label already updated
        result = collected
        _apply_whitelist()
        root.destroy()

    def on_save_only() -> None:
        collected = _collect_cfg()
        if collected is None:
            return  # validation failed; label already updated
        import config as _cfg_mod
        try:
            _cfg_mod.save(collected)
            _log.info("config_ui: config saved (Save - no launch)")
        except Exception as exc:
            _log.warning("config_ui: failed to save config: %s", exc)
        _apply_whitelist()
        _shutdown[0] = True
        root.destroy()

    def on_shutdown() -> None:
        _shutdown[0] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_shutdown)

    ttk.Button(btn_frame, text="Shut Down", command=on_shutdown).pack(side="right", padx=(4, 0))
    ttk.Button(btn_frame, text="Save", command=on_save_only).pack(side="right", padx=(4, 0))
    ttk.Button(btn_frame, text="Save & Load", command=on_save_and_load).pack(side="right")

    root.mainloop()
    if _shutdown[0]:
        return False
    return result
