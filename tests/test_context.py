from tokenslim.context import SharedContext


def test_shared_context_init():
    sc = SharedContext(items=["hello world", "goodnight moon"])
    assert len(sc.items) == 2
    assert sc.items[0] == "hello world"
    assert sc.items[1] == "goodnight moon"


def test_shared_context_deduplication():
    sc = SharedContext(threshold=0.5)
    # Add first item
    assert sc.add_item("The quick brown fox jumps over the lazy dog") is True
    assert len(sc.items) == 1

    # Add overlapping item (high similarity)
    # "The quick brown fox jumps over a lazy hound" has high word overlap
    assert sc.add_item("The quick brown fox jumps over a lazy hound") is False
    assert len(sc.items) == 1

    # Since the new one is shorter or equal, we keep the original (or longer if we replace).
    # Let's test keeping the longer one:
    sc2 = SharedContext(threshold=0.5)
    sc2.add_item("quick brown fox")
    assert sc2.add_item("very quick brown fox") is False
    # But because "very quick brown fox" is longer, it should replace "quick brown fox"
    assert sc2.items[0] == "very quick brown fox"


def test_shared_context_similarity():
    sc = SharedContext()
    sim = sc._similarity("cat sat on mat", "dog sat on mat")
    # words: cat sat on mat -> cat, sat, mat (excluding "on")
    # words: dog sat on mat -> dog, sat, mat (excluding "on")
    # intersection: sat, mat -> 2
    # union: cat, dog, sat, mat -> 4
    # similarity: 2 / 4 = 0.5
    assert sim == 0.5


def test_shared_context_serialize_deserialize():
    sc = SharedContext(items=["this is a nice day", "hello my friend"])
    payload = sc.serialize(compress=False)

    sc2 = SharedContext.deserialize(payload)
    assert len(sc2.items) == 2
    assert sc2.items == sc.items


def test_shared_context_serialize_compressed():
    sc = SharedContext(
        items=[
            "This is a very long paragraph that we want to compress "
            "for inter-agent context handoff.",
            "Another separate fact about coding and testing.",
        ]
    )
    payload = sc.serialize(compress=True, target_ratio=0.5)

    sc2 = SharedContext.deserialize(payload)
    assert len(sc2.items) == 2
    # Compressed items might be slightly modified/compressed, but they
    # should be deserialized successfully.
    assert isinstance(sc2.items[0], str)
