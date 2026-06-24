from tokenslim.compressors.text import TextCompressor
from tokenslim.config import Config
from tokenslim.store import get_store


def _compress(text, **cfg):
    config = Config(**cfg)
    store = get_store(config) if config.ccr else None
    return TextCompressor(config, store)(text)


def test_basic_prose_compression():
    text = (
        "This is the first sentence of the paragraph. "
        "Here is the second sentence of the paragraph "
        "which we will make slightly longer. "
        "And this is the third sentence of the paragraph "
        "which we will also make longer to ensure space savings. "
        "Finally, this is the fourth sentence of the paragraph."
    )
    # Target ratio 0.25: should keep round(4 * 0.25) = 1 sentence (the first one due to lead bias).
    out = _compress(text, target_ratio=0.25, min_bytes=0)
    assert "This is the first sentence of the paragraph." in out
    assert "Here is the second sentence" not in out
    assert "[tokenslim:ccr]" in out


def test_position_bias():
    paragraph = (
        "First sentence should have the highest position score. "
        "Middle sentence has no position score bonus and we will "
        "make it extremely long so that dropping it saves a lot of "
        "space and the compression is worth the overhead of the CCR marker. "
        "Last sentence has some position score bonus."
    )
    # Target ratio 0.66: keeps round(3 * 0.66) = 2 sentences.
    # The first (base + 3.0) and the last (base + 1.5) should be kept. The middle (base) is dropped.
    out = _compress(paragraph, target_ratio=0.66, min_bytes=0)
    assert "First sentence" in out
    assert "Last sentence" in out
    assert "Middle sentence" not in out
    assert "[tokenslim:ccr]" in out


def test_query_relevance():
    paragraph = (
        "The cat sat on the mat and this is a very long sentence "
        "about the cat sitting on the mat to make sure we save space. "
        "The weather is lovely today. "
        "Apples are sweet and red and we are describing how sweet "
        "and red they are in a long sentence."
    )
    # Query is "weather": the middle sentence should get a +3.0 query relevance bonus
    # and be kept, even though it has no position bias over the first/last sentences.
    out = _compress(paragraph, target_ratio=0.33, query="weather", min_bytes=0)
    assert "The weather is lovely today." in out
    assert "The cat sat on the mat" not in out
    assert "Apples are sweet and red" not in out


def test_structural_elements_intact():
    text = (
        "# Heading 1\n\n"
        "This is a paragraph of prose. It contains multiple sentences "
        "and we are making it very long so that dropping the middle sentence saves space. "
        "The middle sentence is also here and is very long and verbose. "
        "The third sentence is also here and wraps up the paragraph.\n\n"
        "- List item 1\n"
        "- List item 2\n\n"
        "```python\ndef foo():\n    return 42\n```"
    )
    out = _compress(text, target_ratio=0.33, min_bytes=0)
    # Structural elements must remain completely untouched
    assert "# Heading 1" in out
    assert "- List item 1" in out
    assert "- List item 2" in out
    assert "def foo():" in out
    # Prose paragraph is compressed
    assert "[tokenslim:ccr]" in out


def test_ccr_stashing_and_retrieval():
    text = (
        "This is the start of the text. "
        "We are going to drop this sentence which is extremely long and "
        "detailed and contains a lot of unnecessary information that we do "
        "not need for the summary. "
        "This is the end of the text."
    )
    config = Config(target_ratio=0.66, min_bytes=0, ccr=True)
    store = get_store(config)
    compressor = TextCompressor(config, store)
    out = compressor(text)

    # Verify dropped sentence is not in the output but a marker is
    assert "We are going to drop this sentence" not in out
    assert "[tokenslim:ccr]" in out

    # Retrieve the stashed content
    from tokenslim.ccr import parse_marker

    marker = parse_marker(out)
    assert marker is not None

    retrieved = store.get(marker.hash)
    assert retrieved == (
        "We are going to drop this sentence which is extremely long and "
        "detailed and contains a lot of unnecessary information that we do "
        "not need for the summary."
    )
