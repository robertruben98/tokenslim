from tokenslim.ccr import find_markers
from tokenslim.compressors.html import HtmlExtractor
from tokenslim.config import Config
from tokenslim.detector import ContentType
from tokenslim.router import ContentRouter
from tokenslim.store import InMemoryCCRStore

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quarterly  Report</title>
<link rel="stylesheet" href="site.css">
<style>body { color: red; }</style>
<script src="app.js"></script>
<script>trackEverything({"noise": true});</script>
</head>
<body>
<header><h1>MegaCorp Site</h1>
<nav><ul><li><a href="/">Home</a></li><li><a href="/about">About</a></li></ul></nav></header>
<main>
<h2>Revenue Summary</h2>
<p class="lead" data-tracking-id="zz9">Revenue grew <strong>12%</strong> in the third quarter.</p>
<ul><li>EMEA up 8%</li><li>APAC up 21%</li></ul>
<table><tr><th>Region</th><th>Growth</th></tr><tr><td>EMEA</td><td>8%</td></tr></table>
<pre>raw   spacing   kept</pre>
<blockquote>Guidance unchanged.</blockquote>
<p>Read the <a href="https://example.com/full">full report</a> online.</p>
</main>
<aside>Ads and widgets</aside>
<form action="/subscribe"><input name="email"><button>Subscribe now</button></form>
<iframe src="https://ads.example.com/banner"></iframe>
<footer>Copyright MegaCorp Site. <a href="/privacy">Privacy</a></footer>
<!-- secret comment noise -->
</body>
</html>
"""


def _extract(text: str, store=None, **cfg) -> str:
    return HtmlExtractor(Config(**cfg), store)(text)


def test_keeps_main_content():
    out = _extract(_PAGE)
    assert "Revenue grew 12% in the third quarter." in out
    assert "## Revenue Summary" in out
    assert "- EMEA up 8%" in out and "- APAC up 21%" in out
    assert "Region | Growth" in out and "EMEA | 8%" in out
    assert "raw   spacing   kept" in out  # <pre> whitespace preserved
    assert "Guidance unchanged." in out
    assert len(out) < len(_PAGE), out


def test_title_becomes_heading_line():
    out = _extract(_PAGE)
    assert out.splitlines()[0] == "# Quarterly Report"


def test_drops_boilerplate():
    out = _extract(_PAGE)
    for noise in (
        "trackEverything",  # script
        "color: red",  # style
        "site.css",  # attribute noise
        "data-tracking-id",  # attribute noise
        "MegaCorp Site",  # header + footer subtrees
        "Home",  # nav
        "Ads and widgets",  # aside
        "Subscribe now",  # form
        "ads.example.com",  # iframe
        "Privacy",  # footer
        "secret comment noise",  # comment
    ):
        assert noise not in out, f"{noise!r} leaked into: {out}"


def test_link_text_kept_url_dropped_by_default():
    out = _extract(_PAGE)
    assert "full report" in out
    assert "https://example.com/full" not in out


def test_keep_links_config_keeps_urls():
    out = _extract(_PAGE, html_keep_links=True)
    assert "full report (https://example.com/full)" in out


def test_ccr_roundtrip_retrieves_original_html():
    store = InMemoryCCRStore()
    out = _extract(_PAGE, store=store)
    markers = find_markers(out)
    assert len(markers) == 1
    assert markers[0].reason == "html-boilerplate-removed"
    assert store.get(markers[0].hash) == _PAGE


def test_tiny_input_unchanged():
    assert _extract("<p>hi</p>") == "<p>hi</p>"


def test_not_smaller_returns_original():
    # Almost pure text: extraction + marker would be longer than the input.
    text = "<p>" + "lots of words here " * 20 + "</p>"
    assert _extract(text) == text


def test_no_extractable_content_returns_original():
    text = (
        "<html><head><script>var x = 1; console.log('boilerplate only page');</script>"
        "</head><body><nav><a href='/'>Home</a></nav><footer>legal</footer></body></html>"
    )
    assert _extract(text) == text


def test_never_raises_on_malformed_html():
    junk = "<html><body><div><p>ok text " + "<<<>" * 40 + "\n<span>unclosed"
    out = _extract(junk)
    assert isinstance(out, str)


def test_router_routes_html_to_extractor():
    router = ContentRouter(config=Config(min_bytes=0))
    result = router.route(_PAGE)
    assert result.content_type is ContentType.HTML
    assert result.compressor == "html-extractor"
    assert result.changed is True
    assert "Revenue grew" in result.text
