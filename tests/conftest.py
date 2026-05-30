"""
Shared pytest fixtures for the HuBrIS test suite.
"""

import pytest
import tools as _tools


@pytest.fixture(autouse=True)
def _set_runtime():
    """
    Populate _tools._runtime with a minimal non-None sentinel before every test
    so that the startup gate (_assert_ready) does not block tool calls.

    The fixture tears down by restoring the original value so tests that
    explicitly clear _runtime are not affected by ordering.
    """
    original = _tools._runtime
    _tools._runtime = _tools._HubrisRuntime(
        adapter=None,
        bound_session=None,
    )
    yield
    _tools._runtime = original
