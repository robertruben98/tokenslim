from tokenslim.evals import EVAL_FIXTURES, run_eval_suite


def test_evals_suite_run_all():
    results = run_eval_suite("all")
    assert results["total"] == len(EVAL_FIXTURES)
    assert results["baseline_correct"] == len(EVAL_FIXTURES)
    assert results["compressed_correct"] == len(EVAL_FIXTURES)
    assert results["baseline_accuracy"] == 1.0
    assert results["compressed_accuracy"] == 1.0
    assert results["saved_tokens"] > 0
    assert results["ratio"] > 0.0


def test_evals_suite_filtering():
    results = run_eval_suite("gsm8k")
    # Only math_outliers is in gsm8k suite
    assert results["total"] == 1
    assert results["results"][0]["id"] == "math_outliers"


def test_evals_empty_suite():
    results = run_eval_suite("non_existent_suite")
    assert results["total"] == 0
    assert results["baseline_correct"] == 0
