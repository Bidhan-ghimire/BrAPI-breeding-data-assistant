from pathlib import Path
import os
import pytest

@pytest.fixture(autouse=True)
def synthetic_environment(monkeypatch):
    monkeypatch.setenv("PART2_IGNORE_DOTENV", "true")
    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", "https://synthetic.invalid/brapi/v2")
    monkeypatch.setenv("LLM_ALLOW_REMOTE", "true")
    monkeypatch.setenv("LLM_API_KEY", "synthetic-regression-placeholder")
