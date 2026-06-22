from tokenslim.tokenizer import HeuristicTokenizer, count_tokens, get_tokenizer


def test_empty_text_is_zero_tokens():
    assert count_tokens("") == 0


def test_nonempty_text_is_at_least_one_token():
    assert count_tokens("hi") >= 1


def test_token_count_grows_with_length():
    short = count_tokens("the quick brown fox")
    long = count_tokens("the quick brown fox " * 50)
    assert long > short


def test_heuristic_is_in_a_sane_range():
    # ~4 chars/token rule of thumb: 400 chars -> roughly 100 tokens.
    text = "word " * 80  # 400 chars, 80 words
    n = HeuristicTokenizer().count(text)
    assert 50 <= n <= 150


def test_default_tokenizer_is_heuristic_without_model():
    assert get_tokenizer().name == "heuristic"


def test_get_tokenizer_is_cached():
    assert get_tokenizer(None) is get_tokenizer(None)
