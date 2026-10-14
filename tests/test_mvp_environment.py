"""Configuration isolation: real local secrets are never opened by the offline suite."""
import pytest
import brapi_client
import llm


@pytest.mark.parametrize("loader", [brapi_client.load_settings, llm.load_llm_settings])
def test_ignore_default_dotenv_never_calls_reader(monkeypatch, loader):
    def forbidden(*args, **kwargs):
        raise AssertionError("dotenv reader must not be called")
    monkeypatch.setattr("dotenv.dotenv_values", forbidden)
    settings = loader(environ={"PART2_IGNORE_DOTENV": "true"})
    assert settings.base_url


@pytest.mark.parametrize("loader", [brapi_client.load_settings, llm.load_llm_settings])
def test_explicit_synthetic_settings_file_still_loads(tmp_path, loader):
    path = tmp_path / "synthetic-config.txt"
    path.write_text("BRAPI_BASE_URL=https://synthetic.invalid/brapi/v2\nLLM_MODEL=synthetic-model\n", encoding="utf-8")
    settings = loader(env_file=path, environ={"PART2_IGNORE_DOTENV": "true"})
    if loader is brapi_client.load_settings:
        assert settings.base_url == "https://synthetic.invalid/brapi/v2"
    else:
        assert settings.model == "synthetic-model"
