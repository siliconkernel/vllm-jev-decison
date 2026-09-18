"""Tests state their own configuration; a deployment's variables must not leak in."""
import pytest

CONFIGURATION = ('VLLM_API_KEY', 'JEV_DECISON_CONCURRENCY', 'JEV_DECISON_TIMEOUT',
                 'JEV_DECISON_API_KEY', 'JEV_DECISON_UPSTREAM_API_KEY')


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in CONFIGURATION:
        monkeypatch.delenv(name, raising=False)
