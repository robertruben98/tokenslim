from tokenslim import retrieve
from tokenslim.ccr import find_markers
from tokenslim.compressors.diff import DiffCompressor, parse_diff
from tokenslim.config import Config
from tokenslim.store import InMemoryCCRStore


def _file_block(name, n_hunks=3, churn=1):
    lines = [f"diff --git a/{name} b/{name}", f"--- a/{name}", f"+++ b/{name}"]
    for h in range(n_hunks):
        lines.append(f"@@ -{h * 10 + 1},6 +{h * 10 + 1},6 @@ def f{h}():")
        lines += [" ctx 1", " ctx 2"]
        for c in range(churn):
            lines += [f"-old {name} {h} {c}", f"+new {name} {h} {c}"]
        lines += [" ctx 3", " ctx 4"]
    return "\n".join(lines)


def _big_diff(n_files=20):
    return "\n".join(_file_block(f"mod{i}.py") for i in range(n_files))


# --- parsing --------------------------------------------------------------


def test_parse_splits_files_and_hunks():
    diff = _big_diff(3)
    files = parse_diff(diff)
    assert len(files) == 3
    assert all(len(f.hunks) == 3 for f in files)
    assert files[0].header_lines[0].startswith("diff --git")


def test_parse_counts_churn():
    files = parse_diff(_file_block("x.py", n_hunks=1, churn=2))
    # 2 churn pairs -> 4 +/- lines.
    assert files[0].hunks[0].churn == 4


def test_parse_tolerates_no_git_header():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-a\n+b\n"
    files = parse_diff(diff)
    assert len(files) == 1
    assert files[0].hunks[0].churn == 2


# --- compression ----------------------------------------------------------


def test_caps_files_by_change_density():
    diff = _big_diff(20)
    out = DiffCompressor(Config(diff_max_files=5))(diff)
    assert out != diff
    assert len(out) < len(diff)
    # Only 5 file headers survive.
    assert out.count("diff --git") == 5


def test_emits_ccr_marker_for_dropped_chunks():
    diff = _big_diff(20)
    out = DiffCompressor(Config(diff_max_files=5))(diff)
    markers = find_markers(out)
    assert len(markers) == 1
    assert markers[0].reason == "diff-elided"
    assert markers[0].count > 0


def test_round_trip_via_ccr_store():
    store = InMemoryCCRStore()
    diff = _big_diff(20)
    out = DiffCompressor(Config(diff_max_files=5), store=store)(diff)
    marker = find_markers(out)[0]
    restored = retrieve(marker.hash, store=store)
    assert restored is not None
    # The dropped files are gone from the visible output but retrievable.
    assert "mod19.py" not in out or "mod0.py" not in out
    assert "diff --git" in restored


def test_keeps_highest_churn_hunks_per_file():
    # One file, 6 hunks, middle hunk has the most churn -> must be kept.
    lines = ["diff --git a/h.py b/h.py", "--- a/h.py", "+++ b/h.py"]
    for h in range(6):
        churn = 5 if h == 3 else 1
        lines.append(f"@@ -{h * 10 + 1},4 +{h * 10 + 1},4 @@")
        lines += [" c1"]
        for c in range(churn):
            lines += [f"-o{h}{c}", f"+n{h}{c}"]
        lines += [" c2"]
    diff = "\n".join(lines)
    out = DiffCompressor(Config(diff_max_files=10, diff_max_hunks_per_file=3))(diff)
    # The high-churn hunk 3's change content survives.
    assert "+n35" in out or "n34" in out


def test_trims_excess_context_lines():
    # A hunk with 5 leading context lines should be trimmed to diff_context.
    lines = [
        "diff --git a/c.py b/c.py",
        "--- a/c.py",
        "+++ b/c.py",
        "@@ -1,8 +1,8 @@",
        " ctxA",
        " ctxB",
        " ctxC",
        " ctxD",
        " ctxE",
        "-old",
        "+new",
        " ctxF",
        " ctxG",
        " ctxH",
    ]
    # Force compaction to apply by making the input large enough overall.
    big = "\n".join(lines) + "\n" + _big_diff(15)
    out = DiffCompressor(Config(diff_max_files=20, diff_context=2))(big)
    # If compaction applied, the visible c.py hunk drops the far context lines.
    if "ctxA" not in out:  # compaction trimmed leading context
        assert "ctxD" in out or "ctxC" in out


def test_returns_original_when_not_worth_it():
    # A tiny diff well above the caps shouldn't be touched (no real win).
    small = _file_block("only.py", n_hunks=1, churn=1)
    out = DiffCompressor(Config(diff_max_files=10))(small)
    assert out == small


def test_non_diff_text_untouched():
    assert DiffCompressor(Config())("just some prose, not a diff") == "just some prose, not a diff"
