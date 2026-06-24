import json

from tokenslim.ccr import SENTINEL_KEY
from tokenslim.compressors.smartcrusher import (
    ArrayKind,
    SmartCrusher,
    analyze_fields,
    classify_array,
)
from tokenslim.config import Config


def _crush(data, **cfg):
    config = Config(crush_keep_head=5, crush_keep_tail=3, crush_min_items=12, **cfg)
    return json.loads(SmartCrusher(config)(json.dumps(data)))


# --- classifier / field analysis (#21) -----------------------------------


def test_classify_array_kinds():
    assert classify_array([{"a": 1}, {"a": 2}]) is ArrayKind.OBJECTS
    assert classify_array([1, 2, 3]) is ArrayKind.NUMBERS
    assert classify_array(["a", "b"]) is ArrayKind.STRINGS
    assert classify_array([1, "a", {"b": 2}]) is ArrayKind.MIXED
    assert classify_array([]) is ArrayKind.EMPTY


def test_field_analysis_detects_id_and_status():
    items = [{"id": i, "status": "ok"} for i in range(50)]
    items[10]["status"] = "error"
    stats = analyze_fields(items)
    assert stats["id"].is_id_like is True
    assert stats["status"].is_status_like is True
    # "error" appears once in 50 -> rare.
    assert "error" in stats["status"].rare_values
    assert "ok" not in stats["status"].rare_values


def test_field_analysis_no_short_circuit_on_many_codes():
    # 60 distinct status codes, one of them rare. Must still flag the rare one
    # rather than giving up because cardinality is high.
    items = [{"status": f"code_{i % 59}"} for i in range(120)]
    items.append({"status": "UNIQUE_ONCE"})
    stats = analyze_fields(items)
    # status is high-cardinality so it isn't "status-like" by low-card rule,
    # but the explicit name hint still makes it status-like.
    assert stats["status"].is_status_like is True
    assert "UNIQUE_ONCE" in stats["status"].rare_values


# --- core crushing (#20) --------------------------------------------------


def test_crush_drops_middle_and_emits_sentinel():
    data = [{"id": i, "v": "x"} for i in range(100)]
    out = _crush(data)
    # head(5) + tail(3) + 1 sentinel.
    assert len(out) == 9
    sentinel = next(o for o in out if isinstance(o, dict) and SENTINEL_KEY in o)
    rec = sentinel[SENTINEL_KEY]
    assert rec["total"] == 100
    assert rec["kept"] == 8
    assert rec["dropped"] == 92
    assert len(rec["hash"]) == 16


def test_crush_keeps_head_and_tail_items():
    data = [{"id": i} for i in range(100)]
    out = _crush(data)
    ids = [o["id"] for o in out if isinstance(o, dict) and "id" in o]
    assert ids[:5] == [0, 1, 2, 3, 4]
    assert ids[-3:] == [97, 98, 99]


def test_short_array_is_not_crushed():
    data = [{"id": i} for i in range(10)]  # below crush_min_items
    out = _crush(data)
    assert out == data


def test_number_array_is_crushed():
    data = list(range(100))
    out = _crush(data)
    assert any(isinstance(o, dict) and SENTINEL_KEY in o for o in out)
    assert out[0] == 0
    assert out[-1] == 99


def test_ccr_disabled_drops_silently():
    data = [{"id": i} for i in range(100)]
    config = Config(crush_keep_head=5, crush_keep_tail=3, crush_min_items=12, ccr=False)
    out = json.loads(SmartCrusher(config)(json.dumps(data)))
    assert not any(isinstance(o, dict) and SENTINEL_KEY in o for o in out)
    assert len(out) == 8


# --- error & rare-value preservation (#22) --------------------------------


def test_error_rows_survive_crushing():
    data = [{"id": i, "status": "ok"} for i in range(100)]
    data[50] = {"id": 50, "status": "ok", "msg": "ERROR: disk failed"}
    out = _crush(data)
    blob = json.dumps(out)
    assert "disk failed" in blob
    # The error row sits in the dropped middle, so its survival proves the rule.
    kept_ids = [o["id"] for o in out if isinstance(o, dict) and "id" in o]
    assert 50 in kept_ids


def test_rare_status_values_survive():
    data = [{"id": i, "status": "ok"} for i in range(100)]
    data[60] = {"id": 60, "status": "cancelled"}
    out = _crush(data)
    blob = json.dumps(out)
    assert "cancelled" in blob


def test_custom_error_keywords():
    data = [{"id": i, "note": "fine"} for i in range(100)]
    data[55] = {"id": 55, "note": "KABOOM happened"}
    config = Config(
        crush_keep_head=5,
        crush_keep_tail=3,
        crush_min_items=12,
        error_keywords=("kaboom",),
    )
    out = json.loads(SmartCrusher(config)(json.dumps(data)))
    assert "KABOOM" in json.dumps(out)


def test_nested_array_is_crushed_recursively():
    data = {"results": [{"id": i} for i in range(100)], "meta": {"page": 1}}
    out = json.loads(
        SmartCrusher(Config(crush_keep_head=5, crush_keep_tail=3, crush_min_items=12))(
            json.dumps(data)
        )
    )
    assert out["meta"] == {"page": 1}
    assert any(isinstance(o, dict) and SENTINEL_KEY in o for o in out["results"])


def test_invalid_json_passthrough():
    assert (
        SmartCrusher(Config())(
            "{not json",
        )
        == "{not json"
    )


def test_non_array_json_is_minified_only():
    out = SmartCrusher(Config())('{"a": 1,  "b":   2}')
    assert out == '{"a":1,"b":2}'


def test_z_score_outliers_preservation():
    # Mean is near 1, std is small, but 100.0 is an outlier (>2 sigma)
    data = [1.0] * 50 + [100.0] + [1.0] * 50
    out = _crush(data)
    assert 100.0 in out


def test_variance_change_point_preservation():
    # Sequence of values with a variance shift from small to large
    data = [1.0] * 50 + [11.0, -9.0] * 25
    out = _crush(data)
    # The transition around index 50 should be preserved
    # Let's check that we kept at least one element near the transition (e.g. index 49/50)
    assert 1.0 in out[5:-3] or 11.0 in out[5:-3] or -9.0 in out[5:-3]


def test_query_anchor_preservation():
    data = [{"id": i, "val": "normal"} for i in range(100)]
    data[50] = {"id": 50, "val": "find_me_anchor"}
    config = Config(
        crush_keep_head=5,
        crush_keep_tail=3,
        crush_min_items=12,
        query="where is find_me_anchor?",
    )
    out = json.loads(SmartCrusher(config)(json.dumps(data)))
    ids = [item["id"] for item in out if isinstance(item, dict) and "id" in item]
    assert 50 in ids


def test_k_split_budget():
    data = [{"id": i} for i in range(100)]

    # Budget of 5: head + tail should be exactly 5
    config = Config(
        crush_keep_head=5,
        crush_keep_tail=3,
        max_items_after_crush=5,
    )
    out = json.loads(SmartCrusher(config)(json.dumps(data)))
    # 5 items kept + 1 sentinel
    assert len(out) == 6

    # Budget of 1: head + tail should be exactly 1 (fixes k=1 overshoot bug)
    config = Config(
        crush_keep_head=5,
        crush_keep_tail=3,
        max_items_after_crush=1,
    )
    out = json.loads(SmartCrusher(config)(json.dumps(data)))
    # 1 item kept + 1 sentinel
    assert len(out) == 2
