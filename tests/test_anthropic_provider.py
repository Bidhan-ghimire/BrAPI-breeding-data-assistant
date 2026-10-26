"""
tests/test_anthropic_provider.py — Claude as a second model provider (LLM_PROVIDER=anthropic), offline.

What these tests prove, in plain words:
* Unset (or openai), nothing changes: the OpenAI client, its settings and its requests are as before, and the Claude
  checks never run.
* The Claude settings: a claude-... model, ANTHROPIC_API_KEY, LLM_ALLOW_REMOTE=true, an effort per role (front door and
  Retriever low, planner, next-step call and Analyst medium), a reply limit of 16000 and 120 s per request by default.
  Every refusal names the setting and never a value; the chat shows that reason, and any other settings error keeps the
  fixed line (a URL with a password is never shown). The hosted demo refuses Claude; extra-header and logging variables
  for the SDK are refused; --smoke --local never calls Claude. The run and the front door open a Claude client when
  LLM_PROVIDER=anthropic.
* Every request carries the asking role's effort and never temperature, top_p, top_k, thinking or tool_choice. Through
  the real SDK (an offline mock transport): only the key header, to api.anthropic.com, whatever ANTHROPIC_BASE_URL or
  ANTHROPIC_AUTH_TOKEN say; the opened client has no retries, our timeout and telemetry off, and is closed after.
* Claude's blocks (thinking with its signature, redacted thinking, text, tool calls) go back unchanged at both places
  the loop appends a turn; the tool results of one turn go back as ONE user turn, in order; nothing earlier is edited.
* A cut-off reply (reply limit or context window) is incomplete and its tool call never runs; a refusal is blocked,
  with or without its details; several text blocks are read as one answer.
* Budget errors (the monthly spend cap's 429 code, a set spend limit's 400, a used-up credit balance, HTTP 402) end the
  step after one request with a plain message, at every agent and at the front door (a greeting keeps the line written
  by code, with the budget line under it; after "drop" the chat says the earlier question ended); a rate limit is
  retried only with a retry time and when the API does not forbid it; nothing else is retried, and the provider's own
  words are never copied.
* Conversations Claude would refuse are refused before anything is sent; every tool the app serves converts.
The conftest network guard blocks every real socket; the SDK is faked or given a mock transport.
"""
from __future__ import annotations

import contextlib
import copy
import json
from types import SimpleNamespace

import anthropic
import anyio
import httpx2
import pytest
from anthropic.types import Message

import app
import chat_presenter as presenter
import conversation
import llm
import triage
from contracts import AnalysisReport, ToolResult
from llm import FakeModelClient, LlmConfigError, LlmSettings, ModelProviderError, reply_text, reply_tools
import run as run_module
from run import CLARIFY_PROMPT, RunConfig
from test_catalog_summary import seed_catalog
from test_conversation import WaitingJob
from test_triage import SOURCE, chat, decision, no_runs  # noqa: F401  (fixtures)

KEY = "sk-ant-PRIVATE-test-key"
CLAUDE = {"LLM_PROVIDER": "anthropic", "LLM_ALLOW_REMOTE": "true", "LLM_MODEL": "claude-opus-5-5", "ANTHROPIC_API_KEY": KEY}
FINAL = '{"status": "completed", "payload": {"answer": 5}}'
URL = "https://api.anthropic.com/v1/messages"
RAW = "SYNTHETIC private raw body must not be copied"


@pytest.fixture(autouse=True)
def no_sdk_variables_from_the_shell(monkeypatch):
    """The SDK reads these from the process whatever the app passes; a developer's shell must not change a test."""
    for name in ("SPACE_ID", *llm.ANTHROPIC_REFUSED_ENV):
        monkeypatch.delenv(name, raising=False)


def claude_settings(tmp_path, **changes) -> LlmSettings:
    environ = {key: value for key, value in {**CLAUDE, **changes}.items() if value is not None}
    return llm.load_llm_settings(env_file=tmp_path / "none.env", environ=environ)


# --- Claude's replies, as the SDK builds them ------------------------------------------------------------------------------

def thinking(signature):
    return {"type": "thinking", "thinking": "", "signature": signature}


def redacted(data):
    return {"type": "redacted_thinking", "data": data}


def text(words):
    return {"type": "text", "text": words}


def tool_use(call_id, a, b):
    return {"type": "tool_use", "id": call_id, "name": "add", "input": {"a": a, "b": b}}


def message(content, *, stop="end_turn", usage=None, stop_details=None) -> Message:
    return Message.model_validate({"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                                   "content": content, "stop_reason": stop, "stop_sequence": None, "stop_details": stop_details,
                                   "usage": usage or {"input_tokens": 10, "output_tokens": 5}})


def message_json(words="Hi"):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [text(words)],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}


class FakeMessages:
    """The SDK's messages resource: records every request (a deep copy) and plays the script."""

    def __init__(self, script):
        self.script = list(script)
        self.sent: list[dict] = []

    async def create(self, **kwargs):
        self.sent.append(copy.deepcopy(kwargs))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def open_client(settings, script):
    client = llm.AnthropicClient(settings)
    messages = FakeMessages(script)
    client._client = SimpleNamespace(messages=messages)          # the fake SDK: no socket
    return client, messages


class AddTools:
    SCHEMAS = [{"name": "add", "description": "Add two integers.",
                "inputSchema": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}]

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def schemas(self):
        return self.SCHEMAS

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, data={"sum": args["a"] + args["b"]}, complete=True)


def run(model, tmp_path, *, agent="retriever", sleeps=None):
    tools = AddTools()

    async def pause(seconds):
        sleeps.append(seconds)

    async def go():
        return await llm.run_agent_loop(model, tools, agent_name=agent, system_prompt="SYSTEM", user_message="What is 2+3?",
                                        budget=llm.Budget(), log_dir=tmp_path, run_id="claude_test",
                                        model_requested="claude-opus-5-5", **({"sleep": pause} if sleeps is not None else {}))
    return anyio.run(go), tools


def events(tmp_path, agent="retriever"):
    lines = (tmp_path / "claude_test" / f"llm_{agent}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def dumped(reply: Message):
    return [block.model_dump(mode="json", exclude_unset=True, by_alias=True) for block in reply.content]


def markers(request) -> list[tuple[str, object]]:
    """Every cache_control in a request (prompt caching, coach 2026-10-24 23:00 UTC), as (where, value)."""
    found: list[tuple[str, object]] = []

    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "cache_control":
                    found.append((where, value))
                else:
                    walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{where}[{i}]")
    walk(request, "request")
    return found


def stripped(request):
    """The request without its cache markers, in the shape it had before them: a system or a user text sent as one marked text
    block is its plain text again (the API reads a string as one text block)."""
    request = copy.deepcopy(request)

    def drop(node):
        if isinstance(node, dict):
            node.pop("cache_control", None)
            for value in node.values():
                drop(value)
        elif isinstance(node, list):
            for value in node:
                drop(value)
    drop(request)
    system = request.get("system")
    if isinstance(system, list) and len(system) == 1 and system[0].get("type") == "text":
        request["system"] = system[0]["text"]
    for turn in request.get("messages", []):
        content = turn.get("content")
        if turn.get("role") == "user" and isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
            turn["content"] = content[0]["text"]
    return request


# --- settings --------------------------------------------------------------------------------------------------------------

def test_unset_provider_keeps_the_openai_path_and_skips_every_claude_check(tmp_path):
    none = tmp_path / "none.env"
    settings = llm.load_llm_settings(env_file=none, environ={})
    assert settings.provider == "openai" and settings.base_url == "http://localhost:11434/v1"
    assert isinstance(llm.make_model_client(LlmSettings(model="m")), llm.OpenAICompatibleClient)
    assert llm.load_llm_settings(env_file=none, environ={"LLM_PROVIDER": " OpenAI "}).provider == "openai"
    with pytest.raises(llm.ProviderConfigError) as info:
        llm.load_llm_settings(env_file=none, environ={"LLM_PROVIDER": "claude " + KEY})
    assert KEY not in str(info.value) and "LLM_PROVIDER" in str(info.value)
    others = {"SPACE_ID": "owner/demo", "ANTHROPIC_LOG": "debug", "ANTHROPIC_CUSTOM_HEADERS": "x: y", "LLM_MODEL": "gpt-6-luna"}
    assert llm.load_llm_settings(env_file=none, environ=others).provider == "openai"     # Claude-only rules stay Claude's


def test_claude_settings_and_their_defaults(tmp_path):
    settings = claude_settings(tmp_path, LLM_BASE_URL="https://api.openai.com/v1", LLM_API_KEY="sk-openai-OTHER")
    assert settings.provider == "anthropic" and settings.base_url == llm.ANTHROPIC_BASE_URL == "https://api.anthropic.com"
    assert settings.api_key == KEY and settings.allow_remote and settings.model == "claude-opus-5-5"
    assert dict(settings.efforts) == {"front_door": "low", "coordinator": "medium", "retriever": "low", "analyst": "medium"}
    assert settings.anthropic_max_tokens == 16000 and settings.request_timeout == 120.0 and settings.key_source == "the environment"
    assert KEY not in repr(settings) and "sk-openai-OTHER" not in repr(settings) and "'anthropic'" in repr(settings)
    custom = claude_settings(tmp_path, LLM_EFFORT_RETRIEVER=" High ", LLM_EFFORT_FRONT_DOOR="max", LLM_ANTHROPIC_MAX_TOKENS="8000",
                             LLM_ANTHROPIC_TIMEOUT="300")
    assert custom.effort_for("retriever") == "high" and custom.effort_for("front_door") == "max"
    assert custom.effort_for("analyst") == "medium" and custom.effort_for("smoke") == "medium"
    assert custom.anthropic_max_tokens == 8000 and custom.request_timeout == 300.0
    env_file = tmp_path / "claude.env"
    env_file.write_text(f"ANTHROPIC_API_KEY={KEY}\n", encoding="utf-8")
    from_file = llm.load_llm_settings(env_file=env_file, environ={k: v for k, v in CLAUDE.items() if k != "ANTHROPIC_API_KEY"})
    assert from_file.key_source == ".env" and from_file.api_key == KEY
    assert isinstance(llm.make_model_client(settings), llm.AnthropicClient)


@pytest.mark.parametrize("changes, named", [
    ({"LLM_ALLOW_REMOTE": None}, "LLM_ALLOW_REMOTE"),
    ({"LLM_ALLOW_REMOTE": "false"}, "LLM_ALLOW_REMOTE"),
    ({"LLM_MODEL": "gpt-6-luna"}, "LLM_MODEL"),
    ({"LLM_MODEL": None}, "LLM_MODEL"),
    ({"ANTHROPIC_API_KEY": None}, "ANTHROPIC_API_KEY"),
    ({"ANTHROPIC_API_KEY": "   "}, "ANTHROPIC_API_KEY"),
    ({"SPACE_ID": "owner/demo"}, "hosted demo"),
    ({"ANTHROPIC_CUSTOM_HEADERS": "Authorization: Bearer stolen"}, "ANTHROPIC_CUSTOM_HEADERS"),
    ({"ANTHROPIC_LOG": "debug"}, "ANTHROPIC_LOG"),
    ({"LLM_EFFORT_ANALYST": "extreme"}, "LLM_EFFORT_ANALYST"),
    ({"LLM_EFFORT_FRONT_DOOR": "none"}, "LLM_EFFORT_FRONT_DOOR"),
    ({"LLM_ANTHROPIC_MAX_TOKENS": "64000"}, "LLM_ANTHROPIC_MAX_TOKENS"),
    ({"LLM_ANTHROPIC_MAX_TOKENS": "abc"}, "LLM_ANTHROPIC_MAX_TOKENS"),
    ({"LLM_ANTHROPIC_MAX_TOKENS": "1000"}, "LLM_ANTHROPIC_MAX_TOKENS"),
    ({"LLM_ANTHROPIC_TIMEOUT": "600"}, "LLM_ANTHROPIC_TIMEOUT"),
    ({"LLM_ANTHROPIC_TIMEOUT": "nan"}, "LLM_ANTHROPIC_TIMEOUT"),
], ids=["no-allow-remote", "allow-remote-false", "openai-model", "no-model", "no-key", "blank-key", "hosted", "custom-headers",
        "sdk-log", "bad-effort", "openai-effort", "max-tokens-high", "max-tokens-text", "max-tokens-low", "timeout-high", "timeout-nan"])
def test_claude_refusals_name_the_setting_and_never_a_value(tmp_path, changes, named):
    with pytest.raises(llm.ProviderConfigError) as info:
        claude_settings(tmp_path, **changes)
    said = str(info.value)
    assert named in said and KEY not in said
    assert not any(value in said for value in changes.values() if value and len(value.strip()) > 3)


def test_the_chat_shows_a_provider_reason_and_never_a_settings_value(tmp_path, monkeypatch):
    config = lambda q: SimpleNamespace(validate=lambda: [])
    for name, value in {**CLAUDE, "ANTHROPIC_API_KEY": ""}.items():          # the real loader, from the environment
        monkeypatch.setenv(name, value)
    problems = app.live_preflight("q", config)
    assert problems == ["ANTHROPIC_API_KEY is missing: add it to .env to use LLM_PROVIDER=anthropic"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert app.live_preflight("q", config) == []                              # no LLM_API_KEY line for Claude
    monkeypatch.setenv("LLM_PROVIDER", "openai")                              # any other settings error: the fixed line
    monkeypatch.setenv("LLM_ALLOW_REMOTE", "false")
    monkeypatch.setenv("LLM_BASE_URL", "https://name:private-value@api.example.com/v1")
    problems = app.live_preflight("q", config)
    assert "private-value" not in str(problems) and problems[0].startswith("the model endpoint cannot be used")


def test_the_run_and_the_front_door_open_claude_when_it_is_the_provider(tmp_path, monkeypatch):
    claude = claude_settings(tmp_path)
    monkeypatch.setattr(llm, "load_llm_settings", lambda: claude)
    assert isinstance(app.default_triage_model(), llm.AnthropicClient)

    async def open_the_runs_model():
        controller = SimpleNamespace(_real_client=None, models=SimpleNamespace(requested="configured model"))
        async with contextlib.AsyncExitStack() as stack:
            controller._stack = stack
            model = await run_module.Controller._real_model(controller)
            opened = (type(model), model._client is not None, model._client.max_retries, controller.models.requested)
        return opened, model._client

    opened, after = anyio.run(open_the_runs_model)
    assert opened == (llm.AnthropicClient, True, 0, "claude-opus-5-5") and after is None     # closed with the run


def test_the_local_smoke_never_calls_claude(tmp_path, monkeypatch):
    claude = claude_settings(tmp_path)
    monkeypatch.setattr(llm, "load_llm_settings", lambda: claude)
    with pytest.raises(LlmConfigError, match="paid requests"):
        anyio.run(llm.smoke, "local", tmp_path, lambda line: None)


# --- requests ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("agent, effort", [("retriever", "low"), ("coordinator_1", "medium"), ("coordinator_next_3", "medium"),
                                           ("analyst", "medium"), ("smoke", "medium")])
def test_every_request_carries_the_roles_effort_and_nothing_claude_refuses(tmp_path, agent, effort):
    client, sdk = open_client(claude_settings(tmp_path), [message([text(FINAL)])])
    result, _ = run(client, tmp_path, agent=agent)
    assert result.status == "completed" and result.payload == {"answer": 5} and result.model_reported == "claude-opus-5-5"
    sent = sdk.sent[0]
    assert set(sent) == {"model", "max_tokens", "system", "messages", "tools", "output_config"}   # no sampling, thinking, tool_choice
    assert sent["output_config"] == {"effort": effort} and sent["model"] == "claude-opus-5-5" and sent["max_tokens"] == 16000
    plain = stripped(sent)                                 # prompt caching (coach 2026-10-24 23:00 UTC): the markers aside
    assert plain["system"].startswith("SYSTEM\n\n") and plain["messages"] == [{"role": "user", "content": "What is 2+3?"}]
    assert plain["tools"] == [{"name": "add", "description": "Add two integers.", "input_schema": AddTools.SCHEMAS[0]["inputSchema"]}]
    assert dict(markers(sent)) == {"request.system[0]": {"type": "ephemeral"}, "request.messages[0].content[0]": {"type": "ephemeral"},
                                   "request.tools[0]": {"type": "ephemeral"}}
    start = events(tmp_path, agent)[0]
    assert start["event"] == "start" and start["reply_limit_sent"] == 16000 and start["effort_sent"] == effort
    assert start["max_tokens"] == 8192                                         # the caller's limit, as logged before


@pytest.mark.parametrize("timeout", ["30", "300"])
def test_the_real_sdk_sends_only_the_key_to_the_fixed_address_at_the_front_doors_effort(tmp_path, monkeypatch, timeout):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://redirect.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-PRIVATE")
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
    seen: list[httpx2.Request] = []

    def answer(request):
        seen.append(request)
        return httpx2.Response(200, json=message_json("Hi"))

    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(answer))
    model = llm.AnthropicClient(claude_settings(tmp_path, LLM_ANTHROPIC_TIMEOUT=timeout), http_client=http)
    reply = anyio.run(triage._complete, model, "PROMPT", "hello")              # opens and closes the client as the app does
    assert len(seen) == 1 and model._client is None and http.is_closed         # closed after the call
    request = seen[0]
    assert request.extensions["timeout"]["read"] == float(timeout)
    assert str(request.url) == URL and request.headers["x-api-key"] == KEY and "authorization" not in request.headers
    assert not any(name.lower() in ("traceparent", "tracestate") for name in request.headers)
    body = json.loads(request.content)
    assert body["output_config"] == {"effort": "low"} and body["max_tokens"] == 16000 and stripped(body)["system"] == "PROMPT"
    assert stripped(body)["messages"] == [{"role": "user", "content": "hello"}] and body["model"] == "claude-opus-5-5"
    assert dict(markers(body)) == {"request.system[0]": {"type": "ephemeral"}, "request.messages[0].content[0]": {"type": "ephemeral"}}
    assert not set(body) & {"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools"}
    assert reply.content == "Hi" and reply.finish_reason == "stop" and reply.usage.total_tokens == 15


def test_the_opened_sdk_client_never_retries_and_has_telemetry_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
    seen: list[httpx2.Request] = []

    def overloaded(request):
        seen.append(request)
        return httpx2.Response(529, headers={"retry-after-ms": "1"},
                               json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

    model = llm.AnthropicClient(claude_settings(tmp_path),
                                http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(overloaded)))

    async def go():
        async with model as opened:
            options = (opened._client.max_retries, opened._client.timeout, opened._client.open_telemetry["traces"]["enabled"])
            with pytest.raises(ModelProviderError) as info:
                await opened.complete([USER], [], max_tokens=10)
        return options, info.value

    options, error = anyio.run(go)
    assert options == (0, 120.0, False) and len(seen) == 1                      # one request: the SDK did not retry
    assert error.code == "http_error" and error.message == "provider returned HTTP 529"


# --- the conversation --------------------------------------------------------------------------------------------------------

def test_claudes_blocks_go_back_unchanged_and_one_turns_tool_results_form_one_user_turn(tmp_path):
    first = message([thinking("SIG-1"), redacted("OPAQUE"), text("Adding."), tool_use("toolu_a", 2, 3), tool_use("toolu_b", 1, 1)],
                    stop="tool_use")
    second = message([thinking("SIG-2"), text("The sum is 5.")])                       # not the JSON object: a repair turn
    third = message([text(FINAL)], usage={"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 30})
    client, sdk = open_client(claude_settings(tmp_path), [first, second, third])
    result, tools = run(client, tmp_path)
    assert result.status == "completed" and tools.calls == [("add", {"a": 2, "b": 3}), ("add", {"a": 1, "b": 1})]
    asked = sdk.sent[1]["messages"]
    assert asked[1] == {"role": "assistant", "content": dumped(first)}
    assert asked[1]["content"][:2] == [thinking("SIG-1"), redacted("OPAQUE")]          # signature and opaque data intact
    results = asked[2]
    assert len(asked) == 3 and results["role"] == "user" and [b["type"] for b in results["content"]] == ["tool_result"] * 2
    assert [b["tool_use_id"] for b in results["content"]] == ["toolu_a", "toolu_b"]
    assert json.loads(results["content"][0]["content"])["data"] == {"sum": 5}
    repaired = sdk.sent[2]["messages"]
    plain = stripped({"messages": repaired})["messages"]                       # the cache marker moves to the newest turn
    assert plain[:3] == stripped({"messages": asked})["messages"] and repaired[3] == {"role": "assistant", "content": dumped(second)}
    assert plain[4]["role"] == "user" and "not a valid JSON object" in plain[4]["content"] and len(repaired) == 5   # append-only
    assert all(sent["system"] == sdk.sent[0]["system"] and sent["tools"] == sdk.sent[0]["tools"] for sent in sdk.sent)
    assert result.usage.prompt_tokens == 10 + 10 + 130 and result.usage.completion_tokens == 5 + 5 + 20
    details = [e["usage_detail"] for e in events(tmp_path) if e["event"] == "assistant"]
    assert details[-1] == {"input_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 30, "output_tokens": 20}


@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded"])
def test_a_cut_off_reply_is_incomplete_and_its_tool_call_never_runs(tmp_path, stop):
    client, sdk = open_client(claude_settings(tmp_path), [message([thinking("S"), tool_use("toolu_x", 2, 3)], stop=stop)])
    result, tools = run(client, tmp_path)
    assert result.status == "incomplete" and tools.calls == [] and result.errors[0].code == "truncated" and len(sdk.sent) == 1


@pytest.mark.parametrize("details, shown", [({"type": "refusal", "category": "bio", "explanation": None}, "(category: bio)."),
                                            (None, "(no category given).")], ids=["with-details", "details-null"])
def test_a_refusal_is_blocked_with_or_without_its_details(tmp_path, details, shown):
    client, sdk = open_client(claude_settings(tmp_path), [message([], stop="refusal", stop_details=details), message([text(FINAL)])])
    result, tools = run(client, tmp_path)
    assert result.status == "blocked" and tools.calls == [] and len(sdk.sent) == 1
    assert result.errors[0].code == "model_refusal" and result.errors[0].message.endswith(shown)
    assert "safety check declined this request" in result.errors[0].message
    reply = llm.normalize_anthropic_response(message([], stop="refusal", stop_details=details))
    assert reply.finish_reason == "content_filter" and reply.refusal.endswith(shown)


def test_several_text_blocks_are_read_as_one_answer(tmp_path):
    client, _ = open_client(claude_settings(tmp_path), [message([thinking("S"), text('{"status": "completed", '),
                                                                 text('"payload": {"answer": 5}}')])])
    result, _ = run(client, tmp_path)
    assert result.status == "completed" and result.payload == {"answer": 5}


def test_a_stop_this_app_does_not_handle_ends_the_step(tmp_path):
    client, sdk = open_client(claude_settings(tmp_path), [message([text("...")], stop="pause_turn"), message([text(FINAL)])])
    result, _ = run(client, tmp_path)
    assert result.status == "failed" and result.errors[0].code == "provider_unexpected_stop" and len(sdk.sent) == 1


# --- errors ------------------------------------------------------------------------------------------------------------------

def api_error(status, kind, words, *, details=None, headers=None):
    error = {"type": kind, "message": words, **({"details": details} if details else {})}
    response = httpx2.Response(status, headers=headers or {}, request=httpx2.Request("POST", URL))
    return anthropic.APIStatusError(RAW, response=response, body={"type": "error", "error": error, "request_id": "req_test"})


SPEND_CAP = ("You have reached your API usage limits: your organization has crossed its monthly API usage threshold, set based "
             "on your organization's API tier. You will regain access on 2026-11-01 at 00:00 UTC.")
BUDGETS = {
    "spend-cap": api_error(429, "rate_limit_error", SPEND_CAP, details={"error_code": "enforced_spend_limit_reached"}),
    "spend-cap-with-retry-after": api_error(429, "rate_limit_error", SPEND_CAP, details={"error_code": "enforced_spend_limit_reached"},
                                            headers={"retry-after": "5"}),
    "workspace-limit": api_error(400, "invalid_request_error", "You have reached your specified workspace API usage limits. "
                                 "You will regain access on 2026-11-01 at 00:00 UTC."),
    "organization-limit": api_error(400, "invalid_request_error", "You have reached your specified API usage limits. "
                                    "You will regain access on 2026-11-01 at 00:00 UTC."),
    "credit": api_error(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API. Please go to "
                        "Plans & Billing to upgrade or purchase credits."),
    "billing": api_error(402, "billing_error", "There's an issue with your billing or payment information."),
}


@pytest.mark.parametrize("name", list(BUDGETS))
def test_a_used_up_budget_ends_the_step_after_one_request_with_a_plain_message(tmp_path, name):
    sleeps: list[float] = []
    client, sdk = open_client(claude_settings(tmp_path), [BUDGETS[name], message([text(FINAL)])])
    result, _ = run(client, tmp_path, sleeps=sleeps)
    assert result.status == "failed" and len(sdk.sent) == 1 and sleeps == []
    error = result.errors[0]
    expected = llm.BILLING_MESSAGE if name == "billing" else llm.BUDGET_MESSAGE
    assert error.code == "provider_quota" and error.message == expected and "No automatic retry was made." in error.message
    body_words = BUDGETS[name].body["error"]["message"]
    assert RAW not in error.message and body_words not in error.message


@pytest.mark.parametrize("where", ["planner", "next-step", "saved next-step", "retriever", "analyst"])
def test_a_used_up_budget_shows_the_plain_screen_at_every_agent(where):
    note = {"planner": "coordinator: agent error provider_quota: ", "next-step": "next step: next-step error provider_quota: ",
            "saved next-step": "supervisor: supervisor error provider_quota: ",          # runs saved before 2026-10-26
            "retriever": "agent error provider_quota: ", "analyst": "agent error provider_quota: "}[where] + llm.BUDGET_MESSAGE
    controller = SimpleNamespace(execution_status="failed", notes=[note] if where in ("planner", "next-step", "saved next-step") else [],
                                 retrieval=SimpleNamespace(status="failed", warnings=[note]) if where == "retriever" else None,
                                 analysis=AnalysisReport(status="failed", caveats=[note]) if where == "analyst" else None)
    shown = presenter.answer_summary(controller)
    assert "**Could not complete this question: Model service unavailable.**" in shown
    assert "The model budget is used up" in shown and "Check the model provider account quota or billing limit" in shown
    assert shown.count("The model budget is used up") == 1


def test_only_code_written_prefixes_count_as_a_provider_failure():
    for note in ("model: agent error provider_quota: x", "next step: agent error provider_quota: x",
                 "coordinator: next-step error provider_quota: x", "supervisor: agent error provider_quota: x",
                 "coordinator: supervisor error provider_quota: x"):
        assert presenter._provider_failure(SimpleNamespace(notes=[note], analysis=None, retrieval=None)) is None


def test_a_rate_limit_is_retried_only_with_a_retry_time(tmp_path):
    sleeps: list[float] = []
    limited = api_error(429, "rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit",
                        headers={"retry-after": "7"})
    client, sdk = open_client(claude_settings(tmp_path), [limited, message([text(FINAL)])])
    result, _ = run(client, tmp_path, sleeps=sleeps)
    assert result.status == "completed" and sleeps == [7.0] and len(sdk.sent) == 2 and sdk.sent[0] == sdk.sent[1]


def test_a_429_the_api_says_not_to_retry_is_not_retried(tmp_path):
    sleeps: list[float] = []
    limited = api_error(429, "rate_limit_error", "rate limited", headers={"retry-after": "7", "x-should-retry": "false"})
    client, sdk = open_client(claude_settings(tmp_path), [limited, message([text(FINAL)])])
    result, _ = run(client, tmp_path, sleeps=sleeps)
    assert result.status == "failed" and len(sdk.sent) == 1 and sleeps == []
    assert result.errors[0].code == "provider_rate_limit" and "disabled automatic retry" in result.errors[0].message


@pytest.mark.parametrize("kind, headers", [("rate_limit_error", {}), ("overloaded_error", {"retry-after": "7"})],
                         ids=["no-retry-time", "unrecognized-kind"])
def test_a_429_without_a_retry_time_is_not_retried_and_says_so_truly(tmp_path, kind, headers):
    sleeps: list[float] = []
    limited = api_error(429, kind, "rate limited", headers=headers)
    client, sdk = open_client(claude_settings(tmp_path), [limited, message([text(FINAL)])])
    result, _ = run(client, tmp_path, sleeps=sleeps)
    assert result.status == "failed" and len(sdk.sent) == 1 and sleeps == []
    assert result.errors[0].code == "provider_rate_limit_unknown"
    assert result.errors[0].message == ("The Claude API returned HTTP 429 without a retry time, or of a kind this app does "
                                        "not recognize. No automatic retry was made.")


@pytest.mark.parametrize("error, code, words", [
    (api_error(401, "authentication_error", "invalid x-api-key"), "provider_http_error", "refused the key (HTTP 401)"),
    (api_error(403, "permission_error", "not allowed"), "provider_http_error", "may not use this model"),
    (api_error(404, "not_found_error", "model: claude-x"), "provider_http_error", "LLM_MODEL"),
    (api_error(529, "overloaded_error", "Overloaded"), "provider_http_error", "provider returned HTTP 529"),
    (api_error(400, "invalid_request_error", "messages.1: bad"), "provider_http_error", "provider returned HTTP 400"),
    (anthropic.AnthropicError("something else"), "provider_provider_error", "AnthropicError"),
    (ValueError("bad JSON"), "provider_provider_error", "it may or may not have been sent"),
], ids=["401", "403", "404", "529", "other-400", "other-sdk-error", "sdk-value-error"])
def test_other_errors_end_the_step_once_without_the_providers_words_or_the_key(tmp_path, error, code, words):
    client, sdk = open_client(claude_settings(tmp_path), [error, message([text(FINAL)])])
    result, _ = run(client, tmp_path, sleeps=[])
    assert result.status == "failed" and len(sdk.sent) == 1
    said = result.errors[0].message
    assert result.errors[0].code == code and words in said and KEY not in said and RAW not in said


def test_a_used_up_budget_at_the_front_door_is_passed_up_and_other_errors_are_not():
    quota = ModelProviderError("quota", llm.BUDGET_MESSAGE)
    with pytest.raises(ModelProviderError):
        triage.ask_model("hello", lambda: FakeModelClient([quota]), source="s", reference="r")
    with pytest.raises(ModelProviderError):
        triage.ask_to_talk("hello", lambda: FakeModelClient([quota]), label="unsure", source="s", reference="r")
    with pytest.raises(ModelProviderError):
        conversation.ask_model("hello", lambda: FakeModelClient([quota]), prompt="p")
    other = lambda: FakeModelClient([ModelProviderError("rate_limit_unknown", "HTTP 429")])
    assert triage.ask_model("hello", other, source="s", reference="r") is None
    assert conversation.ask_model("hello", other, prompt="p") is None


def test_the_chat_says_the_budget_is_used_up_and_starts_nothing(chat, no_runs):  # noqa: F811
    model = FakeModelClient([ModelProviderError("quota", llm.BUDGET_MESSAGE), reply_text("never asked")])
    history, _ = chat.turn("How many studies have yield data in 2020?", triage_model=lambda: model)
    said = history[-1]["content"]
    assert said == f"**The model service cannot answer right now.** {llm.BUDGET_MESSAGE} No question was started."
    assert len(history) == 2 and len(model.requests) == 1 and no_runs == []          # the progress line was replaced


def test_a_greeting_keeps_the_line_written_by_code_with_the_budget_line_under_it(chat, no_runs):  # noqa: F811
    without_model, _ = chat.turn("hello how are you?")
    model = FakeModelClient([ModelProviderError("quota", llm.BUDGET_MESSAGE)])
    history, _ = chat.turn("hello how are you?", triage_model=lambda: model)
    said = history[-1]["content"]
    assert said == without_model[-1]["content"] + f"\n\n**The model service cannot answer right now.** {llm.BUDGET_MESSAGE}"
    assert len(history) == 2 and len(model.requests) == 1 and no_runs == []


def test_a_budget_error_in_the_talk_call_ends_the_turn_plainly(chat, no_runs):  # noqa: F811
    seed_catalog(chat.cache, SOURCE)                                                # the catalog in view: a talk call follows
    model = FakeModelClient([reply_text(decision("unsure", "Hm?")), ModelProviderError("quota", llm.BUDGET_MESSAGE)])
    history, _ = chat.turn("bkla bkla", triage_model=lambda: model)
    assert history[-1]["content"].endswith(f"{llm.BUDGET_MESSAGE} No question was started.")
    assert len(history) == 2 and len(model.requests) == 2 and no_runs == []


def test_while_a_question_waits_a_used_up_budget_sends_nothing_to_it(tmp_path):
    job = WaitingJob(CLARIFY_PROMPT)
    model = FakeModelClient([ModelProviderError("quota", llm.BUDGET_MESSAGE)])
    out = list(app.respond("sounds good to me", [], {"job": job}, make_config=lambda q: RunConfig(question=q, out_dir=tmp_path),
                           make_models=lambda: None, preflight=lambda q: [], triage_model=lambda: model))
    history = out[-1][0]
    assert history[-1]["content"].endswith(f"{llm.BUDGET_MESSAGE} Your message was not sent; the question is still waiting.")
    assert len(history) == 2 and job.replies == [] and out[-1][1]["job"] is job and out[-1][2] == ""


def test_after_drop_a_budget_error_says_the_earlier_question_ended(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "Job", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run may start")))
    job = WaitingJob(CLARIFY_PROMPT)
    model = FakeModelClient([ModelProviderError("quota", llm.BUDGET_MESSAGE)])
    state = {"job": job, "new_question": "How many studies have yield data in 2020?"}
    out = list(app.respond("drop", [], state, make_config=lambda q: RunConfig(question=q, out_dir=tmp_path, cache_dir=tmp_path),
                           make_models=lambda: None, preflight=lambda q: [], triage_model=lambda: model))
    history, state = out[-1][0], out[-1][1]
    assert history[-1]["content"].endswith("The earlier question was dropped as you asked; no new question was started.")
    assert job.closed and "job" not in state and len(model.requests) == 1


def test_after_drop_a_budget_error_in_the_talk_call_says_so_too(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "Job", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run may start")))
    monkeypatch.setenv("BRAPI_BASE_URL", SOURCE)
    seed_catalog(tmp_path / "cache", SOURCE)                                      # the catalog in view: a talk call follows
    job = WaitingJob(CLARIFY_PROMPT)
    model = FakeModelClient([reply_text(decision("unsure", "Hm?")), ModelProviderError("quota", llm.BUDGET_MESSAGE)])
    state = {"job": job, "new_question": "bkla bkla"}
    config = lambda q: RunConfig(question=q, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    out = list(app.respond("drop", [], state, make_config=config, make_models=lambda: None, preflight=lambda q: [],
                           triage_model=lambda: model))
    assert out[-1][0][-1]["content"].endswith("The earlier question was dropped as you asked; no new question was started.")
    assert job.closed and len(model.requests) == 2


# --- conversations Claude would refuse, and the tools -------------------------------------------------------------------------

SYSTEM = {"role": "system", "content": "S"}
USER = {"role": "user", "content": "q"}


def asked_tools(*ids):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": i, "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}} for i in ids]}


def result_for(call_id):
    return {"role": "tool", "tool_call_id": call_id, "name": "add", "content": '{"ok": true}'}


@pytest.mark.parametrize("messages, why", [
    ([SYSTEM, USER, {"role": "system", "content": "late"}], "a system message after the first message"),
    ([SYSTEM, USER, {"role": "assistant", "content": "Sure,"}], "must end with the person's message"),
    ([SYSTEM, USER, result_for("t1")], "a tool result has no matching tool call"),
    ([SYSTEM, USER, asked_tools("t1"), USER], "a tool call has no result"),
    ([SYSTEM, USER, asked_tools("t1")], "a tool call has no result"),
    ([SYSTEM, USER, asked_tools("t1", "t2"), result_for("t2"), result_for("t1")], "tool results out of order"),
    ([SYSTEM, USER, USER], "two turns in a row"),
    ([SYSTEM, {"role": "robot", "content": "x"}], "unknown role"),
    ([SYSTEM, USER, {"role": "assistant", "content": None, "provider_content": ()}, USER], "an empty assistant turn"),
], ids=["late-system", "prefill", "orphan-result", "unanswered-then-user", "unanswered-last", "out-of-order", "two-users",
        "unknown-role", "empty-assistant"])
def test_conversations_claude_would_refuse_are_never_sent(tmp_path, messages, why):
    client, sdk = open_client(claude_settings(tmp_path), [])
    with pytest.raises(ModelProviderError) as info:
        anyio.run(lambda: client.complete(messages, [], max_tokens=10))
    assert info.value.code == "invalid_request" and why in info.value.message and sdk.sent == []


def test_a_turn_written_in_the_openai_shape_is_sent_as_claudes_blocks():
    request = stripped(llm.to_anthropic_request([SYSTEM, USER, asked_tools("t1", "t2"), result_for("t1"), result_for("t2")], []))
    assert request["system"] == "S" and "tools" not in request
    assert request["messages"][1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1, "b": 2}},
        {"type": "tool_use", "id": "t2", "name": "add", "input": {"a": 1, "b": 2}}]}
    assert request["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": '{"ok": true}'},
        {"type": "tool_result", "tool_use_id": "t2", "content": '{"ok": true}'}]}


def test_every_tool_the_app_serves_converts_to_a_claude_tool():
    import analyst_tools
    import brapi_mcp_server
    from mcp_bridge import to_model_tools

    schemas = brapi_mcp_server.tool_schemas() + analyst_tools.analyst_tool_schemas()
    converted = llm.to_anthropic_request([USER], to_model_tools(schemas))["tools"]
    assert [tool["name"] for tool in converted] == [schema["name"] for schema in schemas] and len(converted) >= 12
    assert all(tool["input_schema"] is schema["inputSchema"] and tool["input_schema"]["type"] == "object"
               for tool, schema in zip(converted, schemas))
    with pytest.raises(ModelProviderError, match="no object input schema"):
        llm.to_anthropic_request([USER], [{"type": "function", "function": {"name": "bad", "parameters": {"type": "array"}}}])


# --- other models are untouched ---------------------------------------------------------------------------------------------

def test_other_models_are_untouched_and_a_role_view_has_no_lifetime_of_its_own(tmp_path):
    fake = FakeModelClient([reply_tools([("add", '{"a": 2, "b": 3}')]), reply_text(FINAL)])
    result, _ = run(fake, tmp_path)
    assert result.status == "completed" and llm.for_role(fake, "retriever") is fake
    assert all("provider_content" not in m for request in fake.requests for m in request["messages"])
    assert not any("usage_detail" in e for e in events(tmp_path))
    client = llm.AnthropicClient(claude_settings(tmp_path))
    view = client.for_role("analyst")
    assert view.effort == "medium" and client.for_role("retriever").effort == "low" and client.effort == "medium"
    with pytest.raises(RuntimeError, match="no lifetime of its own"):
        anyio.run(view.__aenter__)
    with pytest.raises(ModelProviderError, match="not open"):
        anyio.run(lambda: view.complete([USER], [], max_tokens=10))
    assert [llm.agent_role(name) for name in ("coordinator_2", "coordinator_next_10", "retriever", "analyst", "smoke", "coordinator")] == [
        "coordinator", "coordinator", "retriever", "analyst", "other", "coordinator"]
