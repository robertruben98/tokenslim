from tokenslim.relevance import BM25Scorer, Scorer, tokenize


def test_is_scorer_protocol():
    assert isinstance(BM25Scorer(), Scorer)


def test_tokenize_lowercases_and_splits():
    assert tokenize("DB_Connection failed!") == ["db_connection", "failed"]


def test_empty_candidates():
    assert BM25Scorer().score("query", []) == []


def test_empty_query_scores_zero():
    assert BM25Scorer().score("", ["anything here"]) == [0.0]


def test_relevant_doc_scores_higher():
    docs = [
        "the user clicked the submit button",
        "database connection failed with a timeout error",
        "rendered the homepage template",
    ]
    scores = BM25Scorer().score("database connection error", docs)
    assert scores[1] == max(scores)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_term_frequency_saturates():
    # More occurrences help, but with diminishing returns (k1 saturation):
    # a doc mentioning the term twice shouldn't score 2x a doc mentioning once.
    scores = BM25Scorer().score("error", ["error", "error error error error"])
    assert scores[1] > scores[0]
    assert scores[1] < 2 * scores[0]


def test_ranking_orders_candidates():
    docs = ["apple banana", "cherry", "apple apple banana banana"]
    scores = BM25Scorer().score("apple banana", docs)
    # Both apple/banana docs outrank the unrelated "cherry".
    assert scores[2] > scores[1]
    assert scores[0] > scores[1]
