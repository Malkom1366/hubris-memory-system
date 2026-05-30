"""Backend adapter registry and factory for HuBrIS.

Mirrors the frontend_adapters.py / registry.py pattern:
  - BackendAdapter: Protocol defining the interface all backend adapters must satisfy.
  - build_backend_adapter(cfg): Factory returning the configured adapter instance.

To add a new backend adapter:
  1. Create backend_adapter_<name>.py with a class having ADAPTER_NAME = "<name>".
  2. Import it here and register it in _REGISTRY.
"""

from typing import Protocol, runtime_checkable

from backend_adapter_ollama import OllamaAdapter


@runtime_checkable
class BackendAdapter(Protocol):
    """Protocol all backend adapters must satisfy."""

    ADAPTER_NAME: str

    def complete(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """
        Send a chat completion request to the backend and return the
        assembled response content string.

        Raises ValueError when the backend returns no content.
        Raises httpx.HTTPError (or equivalent) on transport failure.
        """
        ...

    def embed(self, text: str, model: str) -> bytes | None:
        """
        Embed text using the given model name.

        Returns the embedding as serialized float32 bytes compatible with
        sqlite-vec, or None on any error.
        """
        ...

    def list_models(self) -> list[str]:
        """
        Return a sorted list of model names available at this backend.

        Used by the startup config UI to populate model selection dropdowns.
        Returns [] when the endpoint is unreachable or the call fails for
        any reason. Must not raise.
        """
        ...


_REGISTRY: dict[str, type] = {
    OllamaAdapter.ADAPTER_NAME: OllamaAdapter,
}


def build_backend_adapter(cfg: dict) -> BackendAdapter:
    """
    Build and return the backend adapter specified by cfg["backend_adapter"].

    Falls back to "ollama" when the key is absent. Raises ValueError for
    unknown adapter names.
    """
    name = cfg.get("backend_adapter", "ollama")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backend adapter: {name!r}. Known: {list(_REGISTRY)}"
        )
    bcfg = cfg.get("backends", {}).get(name, {})
    return cls(
        endpoint=bcfg.get("endpoint", "http://localhost:11434"),
        api_key=bcfg.get("api_key", ""),
        timeout=float(bcfg.get("timeout", 600)),
    )


__all__ = ["BackendAdapter", "build_backend_adapter", "OllamaAdapter"]
