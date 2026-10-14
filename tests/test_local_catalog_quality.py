"""Regressions from a local run where season 2026 was hidden after five rows.

Synthetic records and scripted model replies only; the network guard remains on.
"""
import json

import anyio
import pytest

import brapi_mcp_server as srv
from agents.retriever import run_retriever
from contracts import CollectionResult, ToolResult
from llm import Budget, FakeModelClient, compact_tool_result, reply_text, reply_tools
from part2_support import synthetic_context


def collection(rows, complete=True):
    return CollectionResult(records=rows, returned_count=len(rows),
                            reported_total=len(rows) if complete else len(rows) + 1,
                            complete=complete, status=('complete' if rows else 'empty') if complete else 'incomplete')


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    context = synthetic_context(cache_dir=tmp_path / 'cache',
                                out_dir=tmp_path / 'out', run_id='run_catalog_quality')
    seasons = [{'seasonDbId': str(year), 'seasonName': f'Synthetic season {year}', 'year': year}
               for year in range(2001, 2027)]
    monkeypatch.setattr(context.client, 'seasons', lambda **kw: collection(seasons))
    programs = [{'programDbId': f'P{i}', 'programName': f'Synthetic program {i}'} for i in range(26)]
    monkeypatch.setattr(context.client, 'programs', lambda **kw: collection(programs))
    return context


@pytest.mark.parametrize('args', [{'season_id': '2026'}, {'year': '2026'}, {'name_contains': 'SEASON 2026'}])
def test_target_beyond_preview_is_reachable_by_lookup(ctx, args):
    result = srv.dispatch(ctx, 'list_seasons', args)
    assert result.ok, result.error
    view = compact_tool_result(result)['data']
    assert [r['seasonDbId'] for r in view['seasons']] == ['2026']
    assert view['total_matches'] == view['returned'] == 1
    assert view['next_offset'] is None and view['display_complete']


@pytest.mark.parametrize('tool,key,id_key', [('list_seasons', 'seasons', 'seasonDbId'),
                                          ('list_programs', 'programs', 'programDbId')])
def test_following_model_visible_continuation_loses_no_records(ctx, tool, key, id_key):
    ids = []
    offset = 0
    for _ in range(10):
        result = srv.dispatch(ctx, tool, {'offset': offset})
        assert result.ok
        view = compact_tool_result(result)['data']
        assert view['returned'] == len(view[key]) <= 5
        assert view['display_complete'] == (view['next_offset'] is None)
        ids.extend(row[id_key] for row in view[key])
        if view['next_offset'] is None:
            break
        assert view['next_offset'] > offset
        offset = view['next_offset']
    expected = [str(y) for y in range(2001, 2027)] if tool == 'list_seasons' else [f'P{i}' for i in range(26)]
    assert ids == expected


@pytest.mark.parametrize('key', ['candidates', 'variables'])
def test_existing_candidate_pages_continue_at_first_hidden_row(key):
    result = ToolResult(ok=True, complete=True, data={key: [{'id': str(i)} for i in range(25)],
                        'offset': 10, 'next_offset': 35, 'total_matches': 80, 'returned': 25})
    view = compact_tool_result(result)['data']
    assert view['next_offset'] == 15
    assert view['returned'] == 5 and not view['display_complete']
    assert result.data['next_offset'] == 35  # keep the full tool evidence unchanged


def test_wide_candidates_shrink_without_skipping_hidden_rows():
    result = ToolResult(ok=True, complete=True, data={
        'variables': [{'id': str(i), 'name': 'a' * 300, 'description': 'b' * 300} for i in range(25)],
        'offset': 0, 'next_offset': None, 'total_matches': 25, 'returned': 25})
    summary = compact_tool_result(result, max_chars=1600)
    view = summary['data']
    assert len(json.dumps(summary, ensure_ascii=True)) <= 1600
    assert 0 < view['returned'] < 5
    assert view['next_offset'] == view['returned']
    assert not view['display_complete']


def test_program_filter_matches_beyond_first_page(ctx):
    result = srv.dispatch(ctx, 'list_programs', {'name_contains': 'PROGRAM 25'})
    assert result.ok
    assert compact_tool_result(result)['data']['programs'][0]['programDbId'] == 'P25'


def test_season_filters_are_combined_and_do_not_invent_matches(ctx):
    result = srv.dispatch(ctx, 'list_seasons', {'season_id': '2026', 'year': '2001'})
    view = compact_tool_result(result)['data']
    assert result.ok and result.complete
    assert view['seasons'] == [] and view['total_matches'] == 0
    assert view['next_offset'] is None and view['display_complete']


def test_incomplete_catalog_stays_incomplete_after_a_match(ctx, monkeypatch):
    monkeypatch.setattr(ctx.client, 'seasons', lambda **kw: collection([
        {'seasonDbId': '2026', 'year': 2026}], complete=False))
    result = srv.dispatch(ctx, 'list_seasons', {'season_id': '2026'})
    view = compact_tool_result(result)['data']
    assert result.ok and result.complete is False
    assert view['seasons'][0]['seasonDbId'] == '2026'
    assert view['next_offset'] is None and view['display_complete'] is False


@pytest.mark.parametrize('tool,args', [
    ('list_seasons', {'offset': -1}), ('list_seasons', {'limit': 101}),
    ('list_seasons', {'year': 2026}), ('list_seasons', {'approval': {'max_http_attempts': 999}}),
    ('list_programs', {'limit': 0}), ('list_programs', {'base_url': 'https://example.org'}),
])
def test_new_lookup_arguments_keep_strict_bounds_and_authority_rejection(ctx, tool, args):
    result = srv.dispatch(ctx, tool, args)
    assert not result.ok and result.error.code == 'invalid_argument'


def test_retriever_can_resolve_hidden_season_and_export_full_study_artifact(ctx, tmp_path):
    model = FakeModelClient([
        reply_tools([('list_seasons', '{"season_id":"2026"}')]),
        reply_tools([('export_metadata', '{"entity":"studies"}')]),
        reply_text(json.dumps({'status': 'completed', 'payload': {
            'resolved': [{'kind': 'season', 'id': '2026'}], 'studies': [], 'notes': []}})),
    ])

    async def scenario():
        return await run_retriever('List studies from season 2026', model=model, ctx=ctx,
                                   approval=None, budget=Budget(), run_id=ctx.client.run_id,
                                   log_dir=tmp_path / 'logs', expects_observations=False)

    report, agent = anyio.run(scenario)
    assert report.status == 'completed' and report.complete
    assert any(entity.kind == 'season' and entity.id == '2026' for entity in report.resolved)
    assert report.artifacts
    view = json.loads(model.requests[1]['messages'][-1]['content'])['data']
    assert view['seasons'][0]['seasonDbId'] == '2026'
    assert agent.model_reported == 'fake-model'
