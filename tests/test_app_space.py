"""Hosted MCP launcher checks with a fake UI; no server or model is started."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest

import app
from run import PART2_DIR


APP_SPACE = Path(__file__).resolve().parents[1] / "app_space.py"


def load_launcher(monkeypatch, *, sdk=None):
    calls = []

    class FakeUI:
        def queue(self, **kwargs):
            calls.append(("queue", kwargs))
            return self

        def launch(self, **kwargs):
            calls.append(("launch", kwargs))

    ui = FakeUI()

    def build(**kwargs):
        calls.append(("build", kwargs))
        return ui

    monkeypatch.setattr(app, "build_ui", build)
    if sdk is not None:                     # part2 replaced app_mcp.require_memory_transport; the release imports the SDK in app_space
        monkeypatch.setitem(sys.modules, "mcp.shared.memory", None)
    spec = importlib.util.spec_from_file_location("_app_space_test", APP_SPACE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, calls, ui


@pytest.mark.parametrize("values,host,auth", [
    ({}, "127.0.0.1", None),
    ({"SPACE_ID": "synthetic/space", "APP_ALLOW_OPEN": "true"}, "0.0.0.0", None),
    ({"SPACE_ID": "synthetic/space", "APP_USERNAME": "test-user", "APP_PASSWORD": "test-password"},
     "0.0.0.0", ("test-user", "test-password")),
    ({"APP_USERNAME": "test-user", "APP_PASSWORD": "test-password"},
     "127.0.0.1", ("test-user", "test-password")),
])
def test_host_binding_and_existing_authentication_are_preserved(monkeypatch, values, host, auth):
    module, _, _ = load_launcher(monkeypatch)
    assert module.space_launch_options(values) == {
        "server_name": host, "auth": auth, "share": False, "inbrowser": False,
    }


@pytest.mark.parametrize("values,error", [
    ({"SPACE_ID": "synthetic/space"}, "without APP_USERNAME and APP_PASSWORD"),
    ({"APP_USERNAME": "test-user"}, "must both be set"),
    ({"APP_PASSWORD": "test-password"}, "must both be set"),
    ({"SPACE_ID": "synthetic/space", "APP_ALLOW_OPEN": "false"}, "without APP_USERNAME and APP_PASSWORD"),
])
def test_authentication_configuration_failures_still_refuse_launch(monkeypatch, values, error):
    module, calls, _ = load_launcher(monkeypatch)
    with pytest.raises(SystemExit, match=error):
        module.space_launch_options(values)
    assert not any(action == "launch" for action, _ in calls)


def test_environment_cannot_enable_share_or_widen_laptop_binding(monkeypatch):
    module, _, _ = load_launcher(monkeypatch)
    values = {"GRADIO_SHARE": "true", "GRADIO_SERVER_NAME": "0.0.0.0", "GRADIO_SERVER_PORT": "9999"}
    options = module.space_launch_options(values)
    assert options["share"] is False and options["server_name"] == "127.0.0.1"
    assert "server_port" not in options
    values.update(SPACE_ID="synthetic/space", APP_ALLOW_OPEN="true", GRADIO_SERVER_NAME="127.0.0.2")
    assert module.space_launch_options(values)["server_name"] == "0.0.0.0"


def test_space_runtime_port_is_left_to_gradio_ssr(monkeypatch):
    module, _, _ = load_launcher(monkeypatch)
    options = module.space_launch_options({"SPACE_ID": "synthetic/space", "APP_ALLOW_OPEN": "true",
                                          "GRADIO_SERVER_PORT": "7861", "GRADIO_NODE_SERVER_PORT": "7860"})
    assert "server_port" not in options and "node_port" not in options and "ssr_mode" not in options


def test_main_launches_existing_module_demo_with_theme_and_css(monkeypatch):
    module, calls, ui = load_launcher(monkeypatch)
    theme = object()
    monkeypatch.setattr(app, "build_theme", lambda: theme)
    monkeypatch.setattr(module, "space_launch_options", lambda: {
        "server_name": "0.0.0.0", "auth": ("test-user", "test-password"), "share": False, "inbrowser": False,
    })
    module.main()
    assert module.demo is ui
    assert [name for name, _ in calls].count("build") == 1
    assert calls[-1] == ("launch", {"server_name": "0.0.0.0", "auth": ("test-user", "test-password"),
                                    "share": False, "inbrowser": False, "theme": theme, "css": app.CSS})


def test_space_ignores_dotenv_before_ui_build_and_does_not_read_settings(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "synthetic/space")
    monkeypatch.setenv("PART2_IGNORE_DOTENV", "false")
    import os
    load_launcher(monkeypatch)
    assert os.environ["PART2_IGNORE_DOTENV"] == "true"


def test_missing_sdk_stops_import_before_building_or_launching(monkeypatch):
    def missing():
        raise RuntimeError("synthetic-private-detail")

    with pytest.raises(SystemExit, match="will not fall back") as error:
        load_launcher(monkeypatch, sdk=missing)
    assert "synthetic-private-detail" not in str(error.value)


def test_missing_sdk_preflight_does_not_load_live_model_settings(monkeypatch):
    module, _, _ = load_launcher(monkeypatch)

    def missing():
        raise RuntimeError("synthetic-private-detail")

    def forbidden(*args, **kwargs):
        pytest.fail("MCP failure must stop before live/model settings are loaded")

    monkeypatch.setattr(module, "require_memory_transport", missing)
    monkeypatch.setattr(app, "live_preflight", forbidden)
    problems = module.space_preflight("question")
    assert len(problems) == 1 and "will not fall back" in problems[0]
    assert "synthetic-private-detail" not in problems[0]
