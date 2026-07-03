import json

import pytest
from click.testing import CliRunner

from tokenslim import Finding, analyze_sessions, apply_rules, propose_rules, read_sessions
from tokenslim.cli import main
from tokenslim.learn import LEARN_END_MARKER, LEARN_START_MARKER, RULES_HEADING


def _ev(kind, payload, session="s1", ts=1.0):
    return {"ts": ts, "session_id": session, "kind": kind, "payload": payload}


def _write_sessions(dirpath, events):
    """Write synthetic events as one JSONL file per session_id."""
    by_session = {}
    for event in events:
        by_session.setdefault(event["session_id"], []).append(event)
    for session, session_events in by_session.items():
        lines = "\n".join(json.dumps(e) for e in session_events)
        (dirpath / f"{session}.jsonl").write_text(lines + "\n", encoding="utf-8")


def _fail_call(tool="bash", session="s1"):
    return _ev("tool_call", {"tool": tool, "status": "error"}, session=session)


# --- analyze_sessions: repeated tool failures --------------------------------


def test_detects_repeated_tool_failures():
    events = [
        _fail_call(session="s1"),
        _fail_call(session="s1"),
        _fail_call(session="s2"),
        _ev("tool_call", {"tool": "grep"}, session="s1"),  # success: not counted
    ]
    findings = analyze_sessions(events)
    tool_findings = [f for f in findings if f.kind == "repeated-tool-failure"]
    assert len(tool_findings) == 1, findings
    f = tool_findings[0]
    assert f.evidence_count == 3
    assert f.sessions == ("s1", "s2")
    assert "`bash`" in f.proposed_rule
    assert 0.0 < f.confidence <= 0.95


def test_tool_failures_below_threshold_no_finding():
    findings = analyze_sessions([_fail_call(), _fail_call()])
    assert findings == []
    # Threshold is tunable.
    assert analyze_sessions([_fail_call(), _fail_call()], min_tool_failures=2)


def test_failure_via_following_outcome_and_no_double_count():
    # Plain tool_call followed by a failure outcome: attributed to the tool.
    events = []
    for _ in range(2):
        events.append(_ev("tool_call", {"tool": "edit"}))
        events.append(_ev("outcome", {"status": "failure"}))
    # A call that fails by itself AND is followed by a failure outcome
    # counts once, not twice.
    events.append(_fail_call(tool="edit"))
    events.append(_ev("outcome", {"status": "failure"}))
    findings = analyze_sessions(events)
    (f,) = [x for x in findings if x.kind == "repeated-tool-failure"]
    assert f.evidence_count == 3, f
    assert "`edit`" in f.proposed_rule


def test_failure_flags_beyond_status():
    events = [
        _ev("tool_call", {"tool": "web", "error": "boom"}),
        _ev("tool_call", {"tool": "web", "exit_code": 2}),
        _ev("tool_call", {"tool": "web", "success": False}),
    ]
    (f,) = analyze_sessions(events)
    assert f.kind == "repeated-tool-failure" and f.evidence_count == 3


# --- analyze_sessions: user corrections --------------------------------------


def test_detects_repeated_user_corrections():
    correction = {"status": "corrected", "detail": {"correction": "use `rg --files`"}}
    events = [
        _ev("tool_call", {"tool": "find"}, session="s1"),
        _ev("outcome", correction, session="s1"),
        _ev("tool_call", {"tool": "find"}, session="s2"),
        _ev("outcome", correction, session="s2"),
    ]
    findings = analyze_sessions(events)
    (f,) = [x for x in findings if x.kind == "user-correction"]
    assert f.evidence_count == 2
    assert f.sessions == ("s1", "s2")
    assert f.proposed_rule == "Do use `rg --files` instead of `find`."


def test_correction_requires_adjacent_assistant_action():
    # Correction outcome with no preceding assistant action in the session.
    lone = [_ev("outcome", {"status": "corrected", "correction": "do X"})] * 2
    assert analyze_sessions(lone) == []
    # A non-action event between the tool_call and the outcome breaks adjacency.
    events = [
        _ev("tool_call", {"tool": "find"}),
        _ev("note", {}),
        _ev("outcome", {"status": "corrected", "correction": "do X"}),
    ] * 2
    assert [f for f in analyze_sessions(events) if f.kind == "user-correction"] == []


# --- analyze_sessions: inefficient compression --------------------------------


def test_detects_inefficient_compression():
    bad = {"ratio": 0.01, "orig_tokens": 5000, "new_tokens": 4950}
    events = [
        _ev("compress", bad, session="s1"),
        _ev("compress", bad, session="s2"),
        _ev("compress", {"ratio": 0.5, "orig_tokens": 5000}),  # efficient
        _ev("compress", {"ratio": 0.01, "orig_tokens": 50}),  # small payload
    ]
    findings = analyze_sessions(events)
    (f,) = [x for x in findings if x.kind == "inefficient-compression"]
    assert f.evidence_count == 2
    assert f.sessions == ("s1", "s2")
    assert "min_bytes" in f.proposed_rule


def test_single_inefficient_compress_is_not_a_pattern():
    events = [_ev("compress", {"ratio": 0.0, "orig_tokens": 9000})]
    assert analyze_sessions(events) == []


def test_analyze_tolerates_malformed_events():
    events = [
        "not a dict",
        {"kind": "tool_call"},  # no payload
        {"payload": {"tool": "x"}},  # no kind
        _ev("outcome", {"status": "failure"}),  # no preceding action
        _ev("compress", {"ratio": "NaN?", "orig_tokens": None}),
    ]
    assert analyze_sessions(events) == []


# --- propose_rules -------------------------------------------------------------


def test_propose_rules_deterministic_and_deduped():
    f1 = Finding("repeated-tool-failure", 3, ("s1",), "Check `bash` first.", 0.5)
    f2 = Finding("user-correction", 2, ("s1", "s2"), "Do X instead of Y.", 0.4)
    dup = Finding("user-correction", 2, ("s3",), "Do X instead of Y.", 0.4)
    block = propose_rules([f1, f2, dup])
    assert block == propose_rules([dup, f2, f1]), "output must not depend on input order"
    assert block.startswith(RULES_HEADING + "\n\n")
    assert block.count("Do X instead of Y.") == 1
    assert "- Check `bash` first. (why: repeated-tool-failure seen 3x across 1 session)" in block
    assert "(why: user-correction seen 2x across 2 sessions)" in block
    assert propose_rules([]) == ""


# --- apply_rules ----------------------------------------------------------------


def test_apply_rules_dry_run_never_writes(tmp_path):
    target = tmp_path / "CLAUDE.md"
    diff = apply_rules("## Learned rules (tokenslim)\n\n- Rule.", target)
    assert diff.startswith("---") and "+- Rule." in diff
    assert not target.exists(), "dry_run must not create the file"
    target.write_text("existing\n", encoding="utf-8")
    apply_rules("## Learned rules (tokenslim)\n\n- Rule.", target, dry_run=True)
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_apply_rules_is_idempotent(tmp_path):
    target = tmp_path / "AGENTS.md"
    block = "## Learned rules (tokenslim)\n\n- Always frobnicate."
    first = apply_rules(block, target, dry_run=False)
    assert first != ""
    content = target.read_text(encoding="utf-8")
    assert content.count(LEARN_START_MARKER) == 1
    assert content.count(LEARN_END_MARKER) == 1
    second = apply_rules(block, target, dry_run=False)
    assert second == "", "second apply must be a no-op"
    assert target.read_text(encoding="utf-8") == content


def test_apply_rules_updates_only_managed_section(tmp_path):
    target = tmp_path / "CLAUDE.md"
    original = "# My project\n\nHand-written instructions.\n"
    target.write_text(original, encoding="utf-8")
    apply_rules("## Learned rules (tokenslim)\n\n- Old rule.", target, dry_run=False)
    v1 = target.read_text(encoding="utf-8")
    assert v1.startswith(original), "existing content must be untouched"
    assert "- Old rule." in v1
    # Updating replaces the managed section in place, leaving the rest alone.
    apply_rules("## Learned rules (tokenslim)\n\n- New rule.", target, dry_run=False)
    v2 = target.read_text(encoding="utf-8")
    assert v2.startswith(original)
    assert "- New rule." in v2
    assert "- Old rule." not in v2
    assert v2.count(LEARN_START_MARKER) == 1


def test_apply_rules_sanitizes_markers_in_rule_text(tmp_path):
    # Rule text comes from arbitrary session payloads: literal markers in the
    # block must not break the managed section or its idempotency.
    target = tmp_path / "CLAUDE.md"
    nested = "<!-- tokenslim:learn:" + LEARN_END_MARKER + "end -->"
    block = (
        f"{RULES_HEADING}\n\n"
        f"- Beware {LEARN_END_MARKER} in corrections.\n"
        f"- Nested {nested} splice.\n"
        f"- Also {LEARN_START_MARKER} in text.\n"
    )
    first = apply_rules(block, target, dry_run=False)
    assert first != ""
    content = target.read_text(encoding="utf-8")
    assert content.count(LEARN_START_MARKER) == 1
    assert content.count(LEARN_END_MARKER) == 1
    assert "Beware" in content and "splice" in content
    # Idempotent even with hostile rule text.
    assert apply_rules(block, target, dry_run=False) == "", "second apply must be a no-op"
    assert target.read_text(encoding="utf-8") == content


def test_apply_rules_refuses_unbalanced_markers(tmp_path):
    # Orphan start marker (hand edit / merge conflict): rewriting could
    # delete user content, so apply_rules must refuse and leave the file alone.
    target = tmp_path / "CLAUDE.md"
    original = f"# Project\n\n{LEARN_START_MARKER}\nuser content after an orphan marker\n"
    target.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="malformed managed section"):
        apply_rules(f"{RULES_HEADING}\n\n- Rule.", target, dry_run=False)
    assert target.read_text(encoding="utf-8") == original

    # End marker before start marker is malformed too.
    reordered = f"{LEARN_END_MARKER}\nmiddle\n{LEARN_START_MARKER}\n"
    target.write_text(reordered, encoding="utf-8")
    with pytest.raises(ValueError, match="malformed managed section"):
        apply_rules(f"{RULES_HEADING}\n\n- Rule.", target)
    assert target.read_text(encoding="utf-8") == reordered


# --- CLI ------------------------------------------------------------------------


def _seed_capture_dir(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    events = [_fail_call(session="s1") for _ in range(3)]
    events += [
        _ev("compress", {"ratio": 0.0, "orig_tokens": 4000}, session="s2"),
        _ev("compress", {"ratio": 0.01, "orig_tokens": 4000}, session="s2"),
    ]
    _write_sessions(sessions, events)
    return sessions


def test_cli_learn_no_sessions(tmp_path):
    empty = tmp_path / "sessions"
    empty.mkdir()
    result = CliRunner().invoke(main, ["learn", "--sessions", str(empty)])
    assert result.exit_code == 0, result.output
    assert "no sessions found" in result.output


def test_cli_learn_preview_default_changes_nothing(tmp_path):
    sessions = _seed_capture_dir(tmp_path)
    target = tmp_path / "CLAUDE.md"
    result = CliRunner().invoke(
        main, ["learn", "--sessions", str(sessions), "--target", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert "repeated-tool-failure" in result.output
    assert "inefficient-compression" in result.output
    assert "+" + LEARN_START_MARKER in result.output, "preview must show the diff"
    assert "Preview only" in result.output
    assert not target.exists(), "preview must not write the target"


def test_cli_learn_apply_writes_and_is_idempotent(tmp_path):
    sessions = _seed_capture_dir(tmp_path)
    target = tmp_path / "AGENTS.md"
    args = ["learn", "--sessions", str(sessions), "--target", str(target), "--apply"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert f"Applied learned rules to {target}" in result.output
    content = target.read_text(encoding="utf-8")
    assert LEARN_START_MARKER in content and LEARN_END_MARKER in content
    assert RULES_HEADING in content
    assert "`bash`" in content
    # Re-running against the same sessions is a no-op.
    again = CliRunner().invoke(main, args)
    assert again.exit_code == 0, again.output
    assert "already up to date" in again.output
    assert target.read_text(encoding="utf-8") == content


def test_cli_learn_non_utf8_target_errors_cleanly(tmp_path):
    sessions = _seed_capture_dir(tmp_path)
    target = tmp_path / "CLAUDE.md"
    target.write_bytes(b"# proyecto\n\ncorrecci\xf3n en latin-1\n")  # not valid UTF-8
    result = CliRunner().invoke(
        main, ["learn", "--sessions", str(sessions), "--target", str(target)]
    )
    assert result.exit_code == 1, result.output
    assert f"Error updating {target}" in result.output
    assert not isinstance(result.exception, UnicodeDecodeError), "must not propagate a traceback"


def test_cli_learn_malformed_target_markers_error(tmp_path):
    sessions = _seed_capture_dir(tmp_path)
    target = tmp_path / "CLAUDE.md"
    original = f"{LEARN_START_MARKER}\norphan start, no end marker\n"
    target.write_text(original, encoding="utf-8")
    result = CliRunner().invoke(
        main, ["learn", "--sessions", str(sessions), "--target", str(target), "--apply"]
    )
    assert result.exit_code == 1, result.output
    assert "malformed managed section" in result.output
    assert target.read_text(encoding="utf-8") == original, "file must be left untouched"


def test_cli_learn_events_without_patterns(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write_sessions(sessions, [_ev("tool_call", {"tool": "ok"})])
    result = CliRunner().invoke(main, ["learn", "--sessions", str(sessions)])
    assert result.exit_code == 0, result.output
    assert "no recurring failure patterns" in result.output


def test_read_sessions_feeds_analyze(tmp_path):
    sessions = _seed_capture_dir(tmp_path)
    findings = analyze_sessions(read_sessions(str(sessions)))
    kinds = sorted(f.kind for f in findings)
    assert kinds == ["inefficient-compression", "repeated-tool-failure"]


def test_apply_rules_preserves_crlf_line_endings(tmp_path):
    """A CRLF target must keep its bytes outside the managed section (#42 review)."""
    target = tmp_path / "CLAUDE.md"
    target.write_bytes(b"# proyecto\r\nuser line\r\n")
    block = f"{RULES_HEADING}\n- Prefer rg over grep. (why: faster)"

    apply_rules(block, target, dry_run=False)

    data = target.read_bytes()
    assert data.startswith(b"# proyecto\r\nuser line\r\n")
    assert LEARN_START_MARKER.encode() + b"\r\n" in data
    assert b"- Prefer rg over grep. (why: faster)\r\n" in data
    # Re-application stays idempotent on CRLF files too.
    assert apply_rules(block, target, dry_run=False) == ""
    assert target.read_bytes() == data
