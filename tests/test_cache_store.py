"""
tests/test_cache_store.py — the cache key identifies the QUESTION exactly, and
stored answers are kept as tamper-checked, versioned evidence.

All data is SYNTHETIC. Every file is written under pytest's temporary folder
(tmp_path). No network, no model.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import cache_store as cs
from cache_store import CacheCorruptError, CacheError, CacheIdentity, CacheStore, cache_key, canonical_json

T0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
CASSAVA = "https://cassavabase.org/brapi/v2"
OTHER = "https://test-server.brapi.org/brapi/v2"
REDACTED = "https://cassavabase.org/brapi/v2/studies?pageSize=5"


def ident(**over) -> CacheIdentity:
    base = dict(base_url=CASSAVA, endpoint="/studies", params={"pageSize": 5, "page": 0})
    base.update(over)
    return CacheIdentity(**base)


def put(store: CacheStore, identity: CacheIdentity, body: bytes = b'{"result": {"data": []}}', **over):
    args = dict(http_status=200, fetched_at_utc=T0, source_url_redacted=REDACTED, content_type="application/json")
    args.update(over)
    return store.put(identity, body, **args)


# --- 1. the key: a fingerprint of the question ------------------------------------

def test_key_is_sha256_of_canonical_json():
    identity = ident()
    text = canonical_json(identity)
    assert json.loads(text) == {
        "access_namespace": "public", "base_url": CASSAVA, "endpoint": "/studies",
        "format_version": "1", "params": {"page": 0, "pageSize": 5},
    }
    assert " " not in text                      # compact
    assert len(cache_key(identity)) == 64 and cache_key(identity) == cache_key(ident())


def test_parameter_order_does_not_change_the_key():
    a = ident(params={"pageSize": 5, "page": 0})
    b = ident(params={"page": 0, "pageSize": 5})
    assert cache_key(a) == cache_key(b)


def test_different_servers_give_different_keys_even_with_identical_requests():
    assert cache_key(ident(base_url=CASSAVA)) != cache_key(ident(base_url=OTHER))
    # same host, different path component (v1 vs v2): still a different server for us
    assert cache_key(ident(base_url="https://cassavabase.org/brapi/v1")) != cache_key(ident(base_url=CASSAVA))


def test_parameters_are_not_flattened_into_key_value_strings():
    tricky = ident(params={"a": "1&b=2"})
    plain = ident(params={"a": "1", "b": "2"})
    assert cache_key(tricky) != cache_key(plain)
    assert "1&b=2" in canonical_json(tricky)     # kept as one value, escaped inside JSON


def test_value_types_matter_and_unsupported_types_are_rejected():
    assert cache_key(ident(params={"page": 0})) != cache_key(ident(params={"page": "0"}))
    assert cache_key(ident(params={"active": True})) != cache_key(ident(params={"active": 1}))
    for bad in ({"pageSize": 5.0}, {"x": None}, {"x": [1, 2]}, {"x": {"y": 1}}, {"": "blank name"}):
        with pytest.raises(ValidationError):
            ident(params=bad)


def test_format_version_and_access_namespace_separate_keys():
    base = ident()
    assert cache_key(ident(format_version="2")) != cache_key(base)
    assert cache_key(ident(access_namespace="private")) != cache_key(base)
    with pytest.raises(ValidationError):
        ident(access_namespace="Not Valid")


def test_identity_rejects_unnormalized_base_url_and_bad_endpoint():
    with pytest.raises(ValidationError):
        ident(base_url="https://cassavabase.org/brapi/v2/")        # trailing slash
    with pytest.raises(ValidationError):
        ident(base_url="https://user:pw@cassavabase.org/brapi/v2")   # credentials
    with pytest.raises(ValidationError):
        ident(endpoint="studies")                                    # no leading slash


# --- 2. storing and reading evidence ----------------------------------------------

def test_put_then_get_returns_exact_bytes_and_metadata(tmp_path):
    store = CacheStore(tmp_path)
    body = b'{"metadata": {"pagination": {"totalCount": 2}}, "result": {"data": [{"studyDbId": "S1"}]}}\r\n\xff\x00'
    entry = put(store, ident(), body)
    got_entry, got_body = store.get(ident())
    assert got_body == body                                  # exact bytes, including the odd tail
    assert got_entry == entry
    assert entry.version == 1 and entry.byte_length == len(body)
    assert entry.response_sha256 == cs.sha256_bytes(body) and entry.key == cache_key(ident())
    assert entry.relative_path == f"{entry.key[:2]}/{entry.key}/v0001"
    assert (tmp_path / entry.relative_path / "response.bin").read_bytes() == body


def test_miss_returns_none_not_an_empty_response(tmp_path):
    store = CacheStore(tmp_path)
    assert store.get(ident()) is None
    assert store.versions(ident()) == []


def test_cache_hit_preserves_the_original_fetch_time(tmp_path):
    store = CacheStore(tmp_path)
    put(store, ident(), fetched_at_utc=T0)
    entry, _ = store.get(ident())
    assert entry.fetched_at_utc == T0
    assert entry.stored_at_utc >= T0                         # written now, fetched then
    record = entry.request_record(request_id="req_cache_1", run_id="run_1",
                                  requested_at_utc=T0 + timedelta(days=3), attempt=1, duration_seconds=0.01)
    assert record.origin == "cache" and record.fetched_at_utc == T0
    assert record.response_sha256 == entry.response_sha256 and record.http_status == 200
    assert record.requested_at_utc == T0 + timedelta(days=3)


def test_refresh_creates_a_new_version_and_never_overwrites(tmp_path):
    store = CacheStore(tmp_path)
    first = put(store, ident(), b"first answer")
    second = put(store, ident(), b"second answer", fetched_at_utc=T0 + timedelta(hours=1))
    assert (first.version, second.version) == (1, 2)
    assert first.response_sha256 != second.response_sha256    # same key, two different answers
    assert store.versions(ident()) == [1, 2]
    latest, body = store.get(ident())
    assert latest.version == 2 and body == b"second answer"
    old_entry, old_body = store.get_version(ident(), 1)
    assert old_body == b"first answer" and old_entry == first  # still there, untouched


def test_entries_for_different_servers_never_mix(tmp_path):
    store = CacheStore(tmp_path)
    put(store, ident(base_url=CASSAVA), b"cassava bytes")
    put(store, ident(base_url=OTHER), b"other bytes")
    assert store.get(ident(base_url=CASSAVA))[1] == b"cassava bytes"
    assert store.get(ident(base_url=OTHER))[1] == b"other bytes"


def test_store_serves_only_the_public_namespace_and_its_own_format(tmp_path):
    store = CacheStore(tmp_path)
    with pytest.raises(CacheError):
        put(store, ident(access_namespace="private"))
    with pytest.raises(CacheError):
        store.get(ident(format_version="2"))
    with pytest.raises(CacheError):
        store.put(ident(), "not bytes", http_status=200, fetched_at_utc=T0, source_url_redacted=REDACTED)


def test_put_rejects_bad_metadata(tmp_path):
    store = CacheStore(tmp_path)
    with pytest.raises(ValidationError):
        put(store, ident(), fetched_at_utc=datetime(2026, 9, 25))          # naive time
    with pytest.raises(ValidationError):
        put(store, ident(), http_status=999)
    with pytest.raises(ValidationError):
        put(store, ident(), source_url_redacted="https://cassavabase.org/brapi/v2/studies?token=abc")


# --- 3. corruption is an error, never silence -------------------------------------

def _entry_path(store: CacheStore, identity: CacheIdentity, version: int = 1):
    return store.key_dir(identity) / f"v{version:04d}" / "entry.json"


def test_malformed_entry_json_raises_clear_error(tmp_path):
    store = CacheStore(tmp_path)
    put(store, ident())
    _entry_path(store, ident()).write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(CacheCorruptError) as info:
        store.get(ident())
    assert "malformed" in str(info.value) and "entry.json" in str(info.value)


def test_hash_mismatch_is_rejected(tmp_path):
    store = CacheStore(tmp_path)
    entry = put(store, ident(), b"original bytes")
    (tmp_path / entry.relative_path / "response.bin").write_bytes(b"tampered bytes")
    with pytest.raises(CacheCorruptError) as info:
        store.get(ident())
    assert "do not match the stored hash" in str(info.value)


def test_entry_belonging_to_another_identity_is_rejected(tmp_path):
    store = CacheStore(tmp_path)
    put(store, ident())
    other = ident(params={"pageSize": 999})
    # copy the valid entry into the other identity's slot: key/identity inside will not match
    src = store.key_dir(ident()) / "v0001"
    dst = store.key_dir(other) / "v0001"
    dst.mkdir(parents=True)
    for name in ("entry.json", "response.bin"):
        (dst / name).write_bytes((src / name).read_bytes())
    with pytest.raises(CacheCorruptError) as info:
        store.get(other)
    assert "identity does not match" in str(info.value)


def test_interrupted_write_is_not_a_valid_entry(tmp_path):
    store = CacheStore(tmp_path)
    # Simulate a crash after response.bin was written but before entry.json existed.
    half = store.key_dir(ident()) / "v0001"
    half.mkdir(parents=True)
    (half / "response.bin").write_bytes(b"partial")
    (half / "entry.json.tmp-deadbeef").write_bytes(b"{}")     # leftover temp file
    assert store.versions(ident()) == []
    assert store.get(ident()) is None                          # a miss, not a hit on half data
    fresh = put(store, ident(), b"complete answer")
    assert fresh.version == 2                                  # the half-written slot is never reused
    assert store.get(ident())[1] == b"complete answer"


def test_quarantine_moves_the_bad_version_aside_and_still_raises(tmp_path):
    store = CacheStore(tmp_path)
    entry = put(store, ident(), b"good")
    (tmp_path / entry.relative_path / "response.bin").write_bytes(b"bad")
    with pytest.raises(CacheCorruptError):
        store.get(ident(), quarantine=True)
    names = sorted(p.name for p in store.key_dir(ident()).iterdir())
    assert names[0].startswith("v0001.corrupt-") and len(names) == 1
    assert store.get(ident()) is None                          # the corrupt entry is out of the way
    assert put(store, ident(), b"refetched").version == 2      # and its number is not reused


def test_missing_response_file_is_corrupt_not_empty(tmp_path):
    store = CacheStore(tmp_path)
    entry = put(store, ident(), b"payload")
    (tmp_path / entry.relative_path / "response.bin").unlink()
    with pytest.raises(CacheCorruptError) as info:
        store.get(ident())
    assert "response.bin" in str(info.value)


def test_get_version_of_unknown_version_is_an_error(tmp_path):
    store = CacheStore(tmp_path)
    put(store, ident())
    with pytest.raises(CacheError):
        store.get_version(ident(), 7)
