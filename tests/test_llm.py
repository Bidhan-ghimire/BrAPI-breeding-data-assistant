"""
tests/test_llm.py — the model may only REQUEST tools; this loop validates, budgets,
dispatches sequentially and reports honestly. Every test uses a FAKE model client
and fake or synthetic tools: zero real model requests, zero BrAPI requests.
"""
from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

import llm
from contracts import AgentResult, ToolError, ToolResult, Usage
from llm import Budget, FakeModelClient, LlmConfigError, LlmSettings, ModelProviderError, reply_text, reply_tools

PART2 = Path(__file__).resolve().parents[1]


# --- fakes ------------------------------------------------------------------------------------------

class RecordingTools:
    """Two tiny tools with strict schemas; records calls in order; can be scripted to fail or stall."""

    SCHEMAS = [
        {"name": "add", "description": "Add two integers.",
         "inputSchema": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}},
        {"name": "get_observations", "description": "fetch (budgeted)",
         "inputSchema": {"type": "object", "additionalProperties": False, "required": ["study_db_id", "variable_db_id"],
                         "properties": {"study_db_id": {"type": "string"}, "variable_db_id": {"type": "string"}}}},
        {"name": "pick", "description": "enum test",
         "inputSchema": {"type": "object", "additionalProperties": False, "required": ["entity"],
                         "properties": {"entity": {"type": "string", "enum": ["studies", "variables"]}}}},
    ]

    def __init__(self, *, fail_with: Exception | None = None, big_result: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self.fail_with = fail_with
        self.big_result = big_result

    async def schemas(self):
        return [dict(s) for s in self.SCHEMAS]

    async def call(self, name, args):
        self.calls.append((name, dict(args or {})))
        if self.fail_with is not None:
            raise self.fail_with
        if name == "add":
            return ToolResult(ok=True, data={"sum": args["a"] + args["b"]}, complete=True)
        if name == "get_observations":
            rows = [{"observationDbId": f"O{i}", "value": str(i)} for i in range(2000 if self.big_result else 3)]
            return ToolResult(ok=True, data={"artifact_id": "art_abcdef_0001", "row_count": len(rows), "rows": rows,
                                             "long_text": "x" * 5000},
                              artifact_ids=["art_abcdef_0001"], request_ids=[f"req_{i}" for i in range(12)], complete=True)
        return ToolResult(ok=True, data={"picked": args["entity"]}, complete=True)


FINAL = json.dumps({"status": "completed", "payload": {"answer": 5}})


def run_loop(model, tools, tmp_path, *, budget=None, allowed=None, clock=None, **over):
    async def go():
        return await llm.run_agent_loop(
            model, tools, agent_name="t", system_prompt="be exact", user_message="add 2 and 3",
            budget=budget or Budget(), allowed_tools=allowed, log_dir=tmp_path / "out", run_id="run_t",
            model_requested="fake-model", clock=clock or (lambda: 0.0), **over)

    return anyio.run(go)


def log_lines(tmp_path, result: AgentResult) -> list[dict]:
    text = (tmp_path / "out" / result.log_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- 1. settings: local first, remote refused, secrets hidden --------------------------------------

def test_settings_default_to_the_local_endpoint_and_refuse_remote_unless_allowed(tmp_path):
    settings = llm.load_llm_settings(env_file=tmp_path / "none.env", environ={})
    assert settings.base_url == "http://localhost:11434/v1" and settings.model is None and not settings.allow_remote
    with pytest.raises(LlmConfigError) as info:
        llm.load_llm_settings(env_file=tmp_path / "none.env", environ={"LLM_BASE_URL": "https://api.example.com/v1", "LLM_API_KEY": "sk-SECRET-123"})
    assert "refused" in str(info.value) and "sk-SECRET-123" not in str(info.value)
    remote_ok = llm.load_llm_settings(env_file=tmp_path / "none.env",
                                      environ={"LLM_BASE_URL": "https://api.example.com/v1", "LLM_ALLOW_REMOTE": "true"})
    assert remote_ok.allow_remote and not llm.is_loopback(remote_ok.base_url)
    assert llm.is_loopback("http://127.0.0.1:11434/v1") and llm.is_loopback("http://[::1]:11434/v1")


def test_env_file_is_read_relative_settings_and_real_env_wins(tmp_path):
    env_file = tmp_path / "x.env"
    env_file.write_text("LLM_BASE_URL=http://localhost:11434/v1/\nLLM_MODEL=gpt-oss:20b\nLLM_API_KEY=ollama\n", encoding="utf-8")
    settings = llm.load_llm_settings(env_file=env_file, environ={})
    assert settings.model == "gpt-oss:20b" and settings.base_url == "http://localhost:11434/v1"
    assert llm.load_llm_settings(env_file=env_file, environ={"LLM_MODEL": "other"}).model == "other"


def test_settings_repr_never_shows_the_key():
    settings = LlmSettings(api_key="sk-VERY-SECRET")
    assert "sk-VERY-SECRET" not in repr(settings) and "***" in repr(settings)


def test_reply_limit_field_is_chosen_in_env_sent_as_named_and_logged(tmp_path):
    """Ollama reads max_tokens; newer OpenAI models reject it and want max_completion_tokens (HTTP 400 on 2026-09-28).
    The field is a setting, never a guess; the adapter sends exactly the chosen one; the log says which one carried the limit."""
    none = tmp_path / "none.env"
    assert llm.load_llm_settings(env_file=none, environ={}).reply_limit_param == "max_tokens"
    chosen = llm.load_llm_settings(env_file=none, environ={"LLM_REPLY_LIMIT_PARAM": "max_completion_tokens"})
    assert chosen.reply_limit_param == "max_completion_tokens" and "max_completion_tokens" in repr(chosen)
    with pytest.raises(LlmConfigError, match="LLM_REPLY_LIMIT_PARAM"):
        llm.load_llm_settings(env_file=none, environ={"LLM_REPLY_LIMIT_PARAM": "tokens_max"})
    assert llm.REPLY_LIMIT_PARAMS == ("max_tokens", "max_completion_tokens")

    sent: list[dict] = []

    class Msg:
        content, tool_calls, refusal = "done", [], None

    class Choice:
        message, finish_reason = Msg(), "stop"

    class Resp:
        choices, model, usage = [Choice()], "m", None

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return Resp()

    class Chat:
        completions = Completions()

    class Sdk:
        chat = Chat()

    async def send(param: str):
        client = llm.OpenAICompatibleClient(LlmSettings(model="m", reply_limit_param=param))
        client._client = Sdk()                                              # the fake SDK: no socket, records the request
        return await client.complete([{"role": "user", "content": "hi"}], [], max_tokens=321)

    assert anyio.run(send, "max_completion_tokens").content == "done"
    assert sent[-1]["max_completion_tokens"] == 321 and "max_tokens" not in sent[-1] and sent[-1]["model"] == "m"
    anyio.run(send, "max_tokens")
    assert sent[-1]["max_tokens"] == 321 and "max_completion_tokens" not in sent[-1]
    # the start event of every run records the field, from the model's own settings when it has any
    fake = FakeModelClient([reply_text('{"status": "completed", "payload": {"answer": 5}}')])
    fake.settings = LlmSettings(model="m", reply_limit_param="max_completion_tokens")
    result = run_loop(fake, RecordingTools(), tmp_path)
    starts = [e for e in log_lines(tmp_path, result) if e["event"] == "start"]
    assert starts[0]["reply_limit_param"] == "max_completion_tokens" and starts[0]["max_tokens"] == 8192
    plain = run_loop(FakeModelClient([reply_text('{"status": "completed", "payload": {"answer": 5}}')]), RecordingTools(), tmp_path / "plain")
    assert [e["reply_limit_param"] for e in log_lines(tmp_path / "plain", plain) if e["event"] == "start"] == ["max_tokens"]


def test_reasoning_effort_is_sent_only_when_set_and_logged(tmp_path):
    """Some OpenAI reasoning models refuse tools on the chat endpoint unless reasoning_effort is 'none' (HTTP 400 on 2026-09-28).
    Blank = the field is not sent at all (Ollama and the provider's default stay untouched); a set value goes out and into the log."""
    none = tmp_path / "none.env"
    assert llm.load_llm_settings(env_file=none, environ={}).reasoning_effort is None
    assert llm.load_llm_settings(env_file=none, environ={"LLM_REASONING_EFFORT": " None "}).reasoning_effort == "none"
    with pytest.raises(LlmConfigError, match="LLM_REASONING_EFFORT"):
        llm.load_llm_settings(env_file=none, environ={"LLM_REASONING_EFFORT": "maximum"})
    assert "reasoning_effort='none'" in repr(LlmSettings(reasoning_effort="none"))

    sent: list[dict] = []

    class Msg:
        content, tool_calls, refusal = "done", [], None

    class Choice:
        message, finish_reason = Msg(), "stop"

    class Resp:
        choices, model, usage = [Choice()], "m", None

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return Resp()

    class Chat:
        completions = Completions()

    class Sdk:
        chat = Chat()

    async def send(effort):
        client = llm.OpenAICompatibleClient(LlmSettings(model="m", reasoning_effort=effort))
        client._client = Sdk()
        return await client.complete([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "add"}}], max_tokens=5)

    anyio.run(send, None)
    assert "reasoning_effort" not in sent[-1] and sent[-1]["tool_choice"] == "auto"
    anyio.run(send, "none")
    assert sent[-1]["reasoning_effort"] == "none" and sent[-1]["max_tokens"] == 5
    fake = FakeModelClient([reply_text('{"status": "completed", "payload": {"answer": 5}}')])
    fake.settings = LlmSettings(model="m", reasoning_effort="none")
    result = run_loop(fake, RecordingTools(), tmp_path)
    assert [e["reasoning_effort"] for e in log_lines(tmp_path, result) if e["event"] == "start"] == ["none"]


@pytest.mark.parametrize("base_url, store", [
    ("https://api.openai.com/v1", False), ("https://API.OpenAI.com/v1", False),
    ("http://localhost:11434/v1", None), ("https://api.openai.com.example.net/v1", None),
    ("https://another.example/v1", None),
], ids=["openai", "openai-capitals", "local", "look-alike-host", "other-provider"])
def test_store_false_is_sent_to_openai_only(base_url, store):
    """v1.1 choice 5. OpenAI's docs (2026-10-20): "Chat completions are stored by default for new accounts." and "To
    disable storage in either API, set `store: false`." Other OpenAI-compatible servers (a local one) may not know the
    field, so it goes to api.openai.com only."""
    sent: list[dict] = []

    class Msg:
        content, tool_calls, refusal = "done", [], None

    class Choice:
        message, finish_reason = Msg(), "stop"

    class Resp:
        choices, model, usage = [Choice()], "m", None

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return Resp()

    class Chat:
        completions = Completions()

    class Sdk:
        chat = Chat()

    async def send():
        client = llm.OpenAICompatibleClient(LlmSettings(model="m", base_url=base_url, allow_remote=True))
        client._client = Sdk()
        return await client.complete([{"role": "user", "content": "hi"}], [], max_tokens=5)

    anyio.run(send)
    assert sent[-1].get("store", None) is store and ("store" in sent[-1]) is (store is not None)


def test_real_client_refuses_remote_and_blank_model_before_any_network():
    with pytest.raises(LlmConfigError):
        llm.OpenAICompatibleClient(LlmSettings(base_url="https://api.example.com/v1", model="m"))
    with pytest.raises(LlmConfigError):
        llm.OpenAICompatibleClient(LlmSettings(model=None))
    client = llm.OpenAICompatibleClient(LlmSettings(model="gpt-oss:20b"))   # constructing is offline; nothing is sent
    assert client.settings.model == "gpt-oss:20b"


# --- 2. the happy path and the AgentResult shape ---------------------------------------------------

def test_tool_round_trip_gives_a_completed_result_with_counts_and_log(tmp_path):
    usage1 = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    usage2 = Usage(prompt_tokens=150, completion_tokens=20, total_tokens=170)
    model = FakeModelClient([reply_tools([("add", '{"a": 2, "b": 3}')], usage=usage1), reply_text(FINAL, usage=usage2)])
    tools = RecordingTools()
    result = run_loop(model, tools, tmp_path)
    assert result.status == "completed" and result.payload == {"answer": 5}
    assert result.model_calls == 2 and result.tool_calls == 1 and tools.calls == [("add", {"a": 2, "b": 3})]
    assert result.usage == Usage(prompt_tokens=250, completion_tokens=30, total_tokens=280)
    assert result.model_requested == "fake-model" and result.model_reported == "fake-model"
    roles = [m.role for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "system"]
    tool_msg = result.messages[3]
    assert tool_msg.name == "add" and tool_msg.tool_call_id == "call_0" and json.loads(tool_msg.content)["data"] == {"sum": 5}
    assert [m.correlation_id for m in result.messages] == ["setup", "setup", "m1", "t1", "m2", "terminal"]
    events = [e["event"] for e in log_lines(tmp_path, result)]
    assert events == ["start", "assistant", "tool_request", "tool_result", "assistant", "terminal"]
    assert result.log_path == "run_t/llm_t.jsonl"
    # the model saw the tool result as a compact JSON summary, not raw objects
    assert json.loads(model.requests[1]["messages"][-1]["content"])["ok"] is True


def test_missing_usage_is_unknown_not_zero(tmp_path):
    model = FakeModelClient([reply_tools([("add", '{"a": 1, "b": 1}')], usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
                             reply_text(FINAL)])                     # second reply reports no usage
    result = run_loop(model, RecordingTools(), tmp_path)
    assert result.status == "completed" and result.usage is None


def test_needs_clarification_is_its_own_status(tmp_path):
    model = FakeModelClient([reply_text(json.dumps({"status": "needs_clarification", "question": "Which season?"}))])
    result = run_loop(model, RecordingTools(), tmp_path)
    assert result.status == "needs_clarification" and result.payload == {"question": "Which season?"} and result.tool_calls == 0


# --- 3. validation of every requested tool call -----------------------------------------------------

def _first_tool_error(model) -> dict:
    return json.loads(model.requests[1]["messages"][-1]["content"])["error"]


def test_malformed_json_arguments_go_back_as_a_typed_error_then_bounded(tmp_path):
    model = FakeModelClient([reply_tools([("add", "{not json")]), reply_tools([("add", '{"a": 2, "b": 3}')]), reply_text(FINAL)])
    tools = RecordingTools()
    result = run_loop(model, tools, tmp_path)
    assert result.status == "completed" and tools.calls == [("add", {"a": 2, "b": 3})]
    assert _first_tool_error(model)["code"] == "invalid_argument" and "not valid JSON" in _first_tool_error(model)["message"]
    assert result.tool_calls == 2                                       # the malformed request still counted


def test_list_instead_of_object_and_unknown_tool_and_extra_field_and_bad_enum(tmp_path):
    cases = [
        ("add", "[2, 3]", "JSON object"),
        ("run_shell", '{"cmd": "dir"}', "unknown tool"),
        ("add", '{"a": 2, "b": 3, "approval": true}', "additional"),
        ("add", '{"a": "2", "b": 3}', "type"),
        ("pick", '{"entity": "germplasm"}', "is not one of"),      # jsonschema's wording for a bad enum value
    ]
    for name, args, fragment in cases:
        model = FakeModelClient([reply_tools([(name, args)]), reply_text(FINAL)])
        tools = RecordingTools()
        result = run_loop(model, tools, tmp_path, budget=Budget(max_repairs=5))
        assert result.status == "completed", (name, args)
        assert tools.calls == [], (name, args)                                   # nothing invalid was dispatched
        error = _first_tool_error(model)
        assert error["code"] == "invalid_argument" and fragment.lower() in error["message"].lower(), (name, args, error)


def test_repair_budget_is_bounded(tmp_path):
    model = FakeModelClient([reply_tools([("add", "{bad")]), reply_tools([("add", "{bad")]), reply_tools([("add", "{bad")]), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path, budget=Budget(max_repairs=2))
    assert result.status == "failed" and result.errors[0].code == "malformed_arguments" and result.model_calls == 3


def test_unauthorized_tool_for_this_agent_is_refused_without_dispatch(tmp_path):
    model = FakeModelClient([reply_tools([("get_observations", '{"study_db_id": "S1", "variable_db_id": "V1"}')]), reply_text(FINAL)])
    tools = RecordingTools()
    result = run_loop(model, tools, tmp_path, allowed={"add", "pick"})
    assert result.status == "completed" and tools.calls == []
    assert _first_tool_error(model)["code"] == "not_authorized"
    assert model.requests[0]["tools"] == ["add", "pick"]                    # the model was not even offered it


# --- 4. budgets: shared, counted per requested call, never reset ------------------------------------

def test_eight_calls_in_one_turn_are_eight_calls_dispatched_sequentially(tmp_path):
    calls = [("add", json.dumps({"a": i, "b": 1})) for i in range(8)]
    model = FakeModelClient([reply_tools(calls), reply_text(FINAL)])
    tools = RecordingTools()
    result = run_loop(model, tools, tmp_path, budget=Budget(max_tool_calls=10))
    assert result.status == "completed" and result.tool_calls == 8
    assert [a["a"] for _, a in tools.calls] == list(range(8))                 # in order, one at a time
    assert [m.correlation_id for m in result.messages if m.role == "tool"] == [f"t{i}" for i in range(1, 9)]


def test_tool_call_budget_stops_mid_turn_and_reports_limit_reached(tmp_path):
    calls = [("add", json.dumps({"a": i, "b": 1})) for i in range(8)]
    model = FakeModelClient([reply_tools(calls), reply_text(FINAL)])
    tools = RecordingTools()
    result = run_loop(model, tools, tmp_path, budget=Budget(max_tool_calls=5))
    assert result.status == "limit_reached" and len(tools.calls) == 5 and result.tool_calls == 6
    assert result.errors[0].code == "budget_exhausted" and "8 calls were requested" in result.errors[0].message
    assert len(model.requests) == 1                                          # no further model call after the stop


def test_budget_is_shared_across_two_agents_and_not_reset(tmp_path):
    shared = Budget(max_model_calls=3)
    first = run_loop(FakeModelClient([reply_tools([("add", '{"a": 1, "b": 1}')]), reply_text(FINAL)]), RecordingTools(), tmp_path, budget=shared)
    assert first.status == "completed" and shared.model_calls == 2
    second = run_loop(FakeModelClient([reply_tools([("add", '{"a": 1, "b": 1}')]), reply_text(FINAL)]), RecordingTools(), tmp_path, budget=shared)
    assert second.status == "limit_reached" and shared.model_calls == 3     # only one call was left for the second agent
    assert second.errors[0].code == "budget_exhausted"


def test_fetch_budget_and_elapsed_budget(tmp_path):
    fetch = ("get_observations", '{"study_db_id": "S1", "variable_db_id": "V1"}')
    model = FakeModelClient([reply_tools([fetch, fetch, fetch]), reply_text(FINAL)])
    tools = RecordingTools()
    result = run_loop(model, tools, tmp_path, budget=Budget(max_fetches=2))
    assert result.status == "limit_reached" and len(tools.calls) == 2 and "fetch budget" in result.errors[0].message

    ticks = iter([0.0, 0.0, 500.0, 500.0, 500.0, 500.0])
    slow = run_loop(FakeModelClient([reply_tools([("add", '{"a": 1, "b": 1}')]), reply_text(FINAL)]), RecordingTools(), tmp_path,
                    budget=Budget(max_elapsed_seconds=300.0), clock=lambda: next(ticks, 500.0))
    assert slow.status == "limit_reached" and "elapsed" in slow.errors[0].message


# --- 5. distinct failure kinds -----------------------------------------------------------------------

def test_length_truncation_is_incomplete_not_success(tmp_path):
    result = run_loop(FakeModelClient([reply_text('{"status": "completed", "payload": {"answer": 5', finish_reason="length")]), RecordingTools(), tmp_path)
    assert result.status == "incomplete" and result.errors[0].code == "truncated"


def test_the_reply_limit_leaves_room_for_thinking_reaches_the_model_and_is_recorded(tmp_path):
    """A reasoning model writes its thinking as tokens that count against the reply limit. The default gives it room, the
    limit reaches the model on EVERY call, and the log of a run says which limit that run used."""
    assert llm.DEFAULT_MAX_TOKENS == 8192
    model = FakeModelClient([reply_tools([("add", '{"a": 2, "b": 3}')]), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path / "default")
    assert result.status == "completed" and [request["max_tokens"] for request in model.requests] == [8192, 8192]
    starts = [event for event in log_lines(tmp_path / "default", result) if event["event"] == "start"]
    assert len(starts) == 1 and starts[0]["max_tokens"] == 8192
    # a caller may still choose its own limit; the loop passes it on unchanged and records it
    small = FakeModelClient([reply_text(FINAL)])
    result = run_loop(small, RecordingTools(), tmp_path / "small", max_tokens=256)
    assert [request["max_tokens"] for request in small.requests] == [256]
    assert [event["max_tokens"] for event in log_lines(tmp_path / "small", result) if event["event"] == "start"] == [256]
    # more room does not change the rule: a reply that is cut off before its first visible word is no answer
    cut = run_loop(FakeModelClient([reply_text("", finish_reason="length")]), RecordingTools(), tmp_path / "cut")
    assert cut.status == "incomplete" and cut.errors[0].code == "truncated" and cut.payload is None


def test_refusal_is_blocked(tmp_path):
    refusal = llm.ModelReply(content=None, tool_calls=[], finish_reason="stop", model="fake-model", usage=None, refusal="I cannot help with that.")
    result = run_loop(FakeModelClient([refusal]), RecordingTools(), tmp_path)
    assert result.status == "blocked" and result.errors[0].code == "model_refusal"


def test_empty_reply_and_provider_error_are_failures_with_distinct_codes(tmp_path):
    empty = run_loop(FakeModelClient([reply_text("")] * 3), RecordingTools(), tmp_path)   # sent again until the repairs run out
    assert empty.status == "failed" and empty.errors[0].code == "empty_reply" and empty.model_calls == 3
    broken = run_loop(FakeModelClient([ModelProviderError("connection", "could not reach http://localhost:11434/v1")]), RecordingTools(), tmp_path)
    assert broken.status == "failed" and broken.errors[0].code == "provider_connection"


def test_invalid_final_json_gets_one_bounded_repair_then_fails(tmp_path):
    model = FakeModelClient([reply_text("The answer is five."), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path)
    assert result.status == "completed" and result.model_calls == 2
    assert "not a valid JSON object" in model.requests[1]["messages"][-1]["content"]
    stubborn = run_loop(FakeModelClient([reply_text("five"), reply_text("5!"), reply_text("FIVE"), reply_text(FINAL)]), RecordingTools(), tmp_path,
                        budget=Budget(max_repairs=2))
    assert stubborn.status == "failed" and stubborn.errors[0].code == "invalid_json"


def test_oversized_output_is_a_failure(tmp_path):
    result = run_loop(FakeModelClient([reply_text("x" * 30000)]), RecordingTools(), tmp_path, max_output_chars=20000)
    assert result.status == "failed" and result.errors[0].code == "oversized_output"


def test_payload_with_authority_keys_or_nan_is_rejected(tmp_path):
    sneaky = json.dumps({"status": "completed", "payload": {"fetch_approval": {"max_http_attempts": 99}}})
    result = run_loop(FakeModelClient([reply_text(sneaky)]), RecordingTools(), tmp_path)
    assert result.status == "failed" and result.errors[0].code == "payload_rejected"


def test_tool_transport_failures_end_the_run_without_leaking_internals(tmp_path):
    from mcp_bridge import BridgeTimeout

    model = FakeModelClient([reply_tools([("add", '{"a": 1, "b": 1}')]), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(fail_with=BridgeTimeout("timeout calling 'add'")), tmp_path)
    assert result.status == "failed" and result.errors[0].code == "tool_timeout"
    text = json.dumps([m.model_dump() for m in result.messages]) + json.dumps([e.model_dump() for e in result.errors])
    assert "Traceback" not in text


# --- 6. compact summaries: valid JSON, explicit omissions, handles kept -------------------------------

def test_compact_summary_keeps_handles_and_marks_omissions(tmp_path):
    model = FakeModelClient([reply_tools([("get_observations", '{"study_db_id": "S1", "variable_db_id": "V1"}')]), reply_text(FINAL)])
    run_loop(model, RecordingTools(big_result=True), tmp_path)
    summary = json.loads(model.requests[1]["messages"][-1]["content"])          # valid JSON by construction
    assert summary["artifact_ids"] == ["art_abcdef_0001"] and summary["request_ids_omitted"] == 7
    assert summary["data"]["row_count"] == 2000 and len(summary["data"]["rows"]) == 6
    assert summary["data"]["rows"][-1] == {"omitted_items": 1995}
    assert summary["data"]["long_text"]["omitted_string_chars"] == 5000
    assert len(json.dumps(summary)) < 6500


def test_compact_summary_drops_data_but_never_slices_when_still_too_big():
    huge = ToolResult(ok=True, data={f"k{i}": "v" * 300 for i in range(100)}, artifact_ids=["art_abcdef_0002"], complete=True)
    summary = llm.compact_tool_result(huge, max_chars=1000)
    assert summary["data"]["omitted_data"] is True and summary["data"]["keys_omitted"] == 95
    assert summary["artifact_ids"] == ["art_abcdef_0002"] and "note" in summary
    json.dumps(summary)                                                          # still valid JSON


# --- 7. cancellation and logging -----------------------------------------------------------------------

def test_cancellation_mid_model_call_is_logged_and_propagates(tmp_path):
    gate = anyio.Event()
    model = FakeModelClient([reply_text(FINAL)], gate=gate)          # the fake model waits on the gate forever
    outcome = {}

    async def scenario():
        async with anyio.create_task_group() as tg:
            async def worker():
                try:
                    await llm.run_agent_loop(model, RecordingTools(), agent_name="c", system_prompt="s", user_message="u",
                                             budget=Budget(), log_dir=tmp_path / "out", run_id="run_c", model_requested="fake-model")
                except anyio.get_cancelled_exc_class():
                    outcome["cancelled"] = True
                    raise
            tg.start_soon(worker)
            await anyio.sleep(0.05)
            tg.cancel_scope.cancel()

    anyio.run(scenario)
    assert outcome.get("cancelled") is True
    events = [json.loads(l)["event"] for l in (tmp_path / "out" / "run_c" / "llm_c.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["start", "cancelled"]


def test_log_never_contains_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-DO-NOT-LEAK")
    model = FakeModelClient([reply_tools([("add", '{"a": 2, "b": 3}')]), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path)
    blob = (tmp_path / "out" / result.log_path).read_text(encoding="utf-8") + result.model_dump_json()
    assert "sk-DO-NOT-LEAK" not in blob


# --- 8. the smoke harness -----------------------------------------------------------------------------

def test_mock_smoke_completes_with_zero_real_model_requests(tmp_path):
    lines = []
    code = anyio.run(llm.smoke, "mock", tmp_path / "out", lines.append)
    assert code == 0
    text = "\n".join(lines)
    assert "status=completed" in text and "model_calls=2" in text and "tool_calls=1" in text
    assert "real_model_requests=0" in text and "usage=unknown" in text and "BrAPI calls: none" in text
    assert "SMOKE PASSED" in text


def test_smoke_cli_argument_rules(capsys, tmp_path):
    assert llm.main(["--smoke"]) == 2                                   # must choose mock or local
    assert llm.main(["--smoke", "--mock-model", "--local"]) == 2
    assert llm.main(["--smoke", "--mock-model", "--out-dir", str(tmp_path / "out")]) == 0
    assert "SMOKE PASSED" in capsys.readouterr().out


def test_normalize_openai_response_shape():
    class Fn:
        name = "add"
        arguments = '{"a": 2, "b": 3}'

    class Call:
        id = "call_x"
        type = "function"
        function = Fn()

    class Msg:
        content = None
        tool_calls = [Call()]
        refusal = None

    class Choice:
        message = Msg()
        finish_reason = "tool_calls"

    class UsageObj:
        prompt_tokens, completion_tokens, total_tokens = 10, 5, 15

    class Resp:
        choices = [Choice()]
        model = "gpt-oss:20b"
        usage = UsageObj()

    reply = llm.normalize_openai_response(Resp())
    assert reply.tool_calls[0].name == "add" and reply.tool_calls[0].arguments_json == '{"a": 2, "b": 3}'
    assert reply.model == "gpt-oss:20b" and reply.usage.total_tokens == 15 and reply.finish_reason == "tool_calls"

    class NoUsage(Resp):
        usage = None

    assert llm.normalize_openai_response(NoUsage()).usage is None


def test_smoke_check_accepts_the_number_under_any_label_but_not_a_wrong_number():
    assert llm._smoke_answer_found({"sum": 5}, 5) and llm._smoke_answer_found({"answer": 5.0, "note": "x"}, 5)
    assert not llm._smoke_answer_found({"answer": 6}, 5) and not llm._smoke_answer_found({"answer": "5"}, 5)
    assert not llm._smoke_answer_found({"answer": True}, 1) and not llm._smoke_answer_found(None, 5)


# Eval 2026-10-17 (Opus 5.5, q14): the Retriever's reply held only thinking; the step failed at once.
def test_an_empty_reply_is_sent_again_with_nothing_appended(tmp_path):
    model = FakeModelClient([reply_text(""), reply_text("   "), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path, budget=Budget(max_repairs=2))
    assert result.status == "completed" and result.model_calls == 3
    assert model.requests[0]["messages"] == model.requests[1]["messages"] == model.requests[2]["messages"]
    repairs = [e for e in log_lines(tmp_path, result) if e["event"] == "repair"]
    assert [e["reason"] for e in repairs] == ["empty reply: the same request was sent again"] * 2


def test_empty_replies_share_the_repair_budget_with_invalid_json(tmp_path):
    model = FakeModelClient([reply_text(""), reply_text("not json"), reply_text(""), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path, budget=Budget(max_repairs=2))
    assert result.status == "failed" and result.errors[0].code == "empty_reply" and result.model_calls == 3


def test_a_cut_off_empty_reply_is_still_incomplete(tmp_path):
    model = FakeModelClient([reply_text("", finish_reason="length"), reply_text(FINAL)])
    result = run_loop(model, RecordingTools(), tmp_path)
    assert result.status == "incomplete" and result.model_calls == 1



def test_compact_summary_cuts_lists_and_text_before_dropping_all_data():
    """Review of fixes (1)-(4), 2026-10-23: an anova_trial preview of 6005 characters lost ALL its data, so the analyst saw no
    number (E16.1 on Luna). Shorter lists and text come first; the cap itself stays."""
    rows = [{"name": f"row {i}", "text": "w" * 390, "more": "x" * 390, "other": "y" * 390} for i in range(5)]
    big = ToolResult(ok=True, data={"h2": 0.75, "rows": rows}, artifact_ids=["art_abcdef_0003"], complete=True)
    first = json.dumps({"data": {"h2": 0.75, "rows": rows}})
    assert len(first) > 6000                                                  # the first pass is over the cap
    summary = llm.compact_tool_result(big)
    assert "omitted_data" not in summary["data"] and summary["data"]["h2"] == 0.75 and len(json.dumps(summary)) <= 6000
    assert summary["data"]["rows"][-1] == {"omitted_items": 2} and "cut to 3 item(s)" in summary["data_note"]
