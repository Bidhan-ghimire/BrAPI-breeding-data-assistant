"""Offline checks for Space startup and full-report access; no server is launched."""
from types import SimpleNamespace

import pytest

import app


def test_space_binding_authentication_and_local_default():
    assert app.launch_options({}) == {"server_name": "127.0.0.1", "auth": None}
    values = {"SPACE_ID": "owner/demo", "APP_USERNAME": "user", "APP_PASSWORD": "test-password"}
    assert app.launch_options(values) == {"server_name": "0.0.0.0", "auth": ("user", "test-password")}
    assert app.launch_options({**values, "GRADIO_SERVER_NAME": "127.0.0.2"})["server_name"] == "127.0.0.2"
    with pytest.raises(SystemExit, match="without APP_USERNAME and APP_PASSWORD"):
        app.launch_options({"SPACE_ID": "owner/demo"})
    with pytest.raises(SystemExit, match="must both be set"):
        app.launch_options({"APP_USERNAME": "user"})
    assert app.launch_options({"SPACE_ID": "owner/demo", "APP_ALLOW_OPEN": "true"})["auth"] is None


def test_hosted_notice_preserves_local_header_and_never_echoes_endpoint_credentials():
    assert app.hosted_notice({}) == ""
    notice = app.hosted_notice({"SPACE_ID": "owner/demo", "LLM_BASE_URL": "https://name:private-value@api.openai.com/v1"})
    assert "OpenAI" in notice and "before database-fetch approval" in notice
    assert "private-value" not in notice and "https://" not in notice
    other = app.hosted_notice({"SPACE_ID": "owner/demo", "LLM_BASE_URL": "https://another.example/v1"})
    assert "configured model provider" in other and "OpenAI" not in other


def test_configuration_failure_is_readable_and_does_not_echo_values(monkeypatch):
    def invalid(_question):
        raise ValueError("secret-that-must-not-appear")
    assert "APP_MAX_HTTP_ATTEMPTS" in app.live_preflight("question", invalid)[0]
    result = list(app.respond("question", [], {}, preflight=invalid))[-1]
    assert "cannot run yet" in result[0][-1]["content"]
    assert "secret-that-must-not-appear" not in str(result)
    assert result[1] == {}


def test_job_construction_failure_is_readable_and_does_not_echo_values(monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("secret-that-must-not-appear")
    monkeypatch.setattr(app, "Job", invalid)
    result = list(app.respond("question", [], {}, preflight=lambda q: []))[-1]
    assert "could not start" in result[0][-1]["content"]
    assert "secret-that-must-not-appear" not in str(result)
    assert result[1] == {}


def test_live_preflight_checks_remote_model_name_and_key_without_echoing_key(monkeypatch):
    import llm
    settings = llm.LlmSettings(base_url="https://api.openai.com/v1", api_key="ollama", allow_remote=True)
    monkeypatch.setattr(llm, "load_llm_settings", lambda: settings)
    config = lambda q: SimpleNamespace(validate=lambda: [])
    problems = app.live_preflight("question", config)
    assert any("LLM_MODEL" in p for p in problems)
    assert any("LLM_API_KEY" in p for p in problems)
    settings = llm.LlmSettings(base_url="https://api.openai.com/v1", api_key="private-key-sentinel", model="fake-model", allow_remote=True)
    assert app.live_preflight("question", config) == []
    settings = llm.LlmSettings(model="local-model")
    assert app.live_preflight("question", config) == []


def test_full_report_keeps_omitted_rows_and_final_acceptance_in_its_own_session(monkeypatch):
    long_report = "| ID |\n|---|\n" + "\n".join(f"| row-{i} |" for i in range(40))
    class FakeController:
        run_id = "test-report"
        accepted = False
        def answer_markdown(self):
            return long_report + ("\nAcceptance: accepted" if self.accepted else "\nAcceptance: pending")
    class FakeJob:
        def __init__(self, *args, **kwargs):
            self.controller = FakeController()
            self.answer_shown = False
        def start(self):
            return self
        def reply(self, text):
            self.controller.accepted = text == "accept"
        def next_event(self):
            if not self.controller.accepted:
                return "prompt", "accept / reject"
            return "done", SimpleNamespace(run_id="test-report", execution_status="completed", human_acceptance="accepted")
    monkeypatch.setattr(app, "Job", FakeJob)
    history, state, _ = list(app.respond("question", [], {}, preflight=lambda q: []))[-1]
    assert "row-39" not in history[-1]["content"]
    assert "row-39" in app.full_report(state)
    assert "Acceptance: pending" in app.full_report(state)
    history, state, _ = list(app.respond("accept", history, state, preflight=lambda q: []))[-1]
    assert "Acceptance: accepted" in app.full_report(state)
    assert "job" not in state
    assert app.full_report({}) == ""  # another session has no report
    assert app.full_report(app.start_over(state)[1]) == ""


def test_chat_event_queue_and_report_component_are_configured(monkeypatch):
    import gradio as gr
    monkeypatch.setattr(app, "source_markdown", lambda: "Synthetic source details")
    monkeypatch.setattr(app, "hosted_notice", lambda: "")
    ui = app.build_ui(preflight=lambda q: [])
    report_boxes = [b for b in ui.blocks.values() if isinstance(b, gr.Textbox) and b.label == "Full report"]
    assert len(report_boxes) == 1 and report_boxes[0].interactive is False
    chat_functions = [f for f in ui.fns.values() if f.name in {"on_submit", "on_press", "start_over"}]
    assert len(chat_functions) == 6
    assert all(f.concurrency_id == app.CHAT_CONCURRENCY_ID and f.concurrency_limit == 1 for f in chat_functions)
    assert all(f.api_visibility == "private" for f in chat_functions)
