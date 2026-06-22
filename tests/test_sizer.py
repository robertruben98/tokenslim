from tokenslim.sizer import compute_optimal_k


def test_zero_items_is_zero():
    assert compute_optimal_k(0) == 0


def test_no_overshoot_at_budget_one():
    # The historical bug kept 2 when the budget resolved to 1.
    assert compute_optimal_k(1, target_ratio=0.2) == 1
    assert compute_optimal_k(2, target_ratio=0.1, decay=0.9) == 1


def test_monotonic_nondecreasing_in_n():
    ks = [compute_optimal_k(n, target_ratio=0.2) for n in (1, 10, 100, 1000, 10000)]
    assert ks == sorted(ks)


def test_tighter_ratio_keeps_fewer_or_equal():
    n = 1000
    assert compute_optimal_k(n, 0.1) <= compute_optimal_k(n, 0.3) <= compute_optimal_k(n, 0.6)


def test_never_exceeds_n():
    for n in (1, 3, 7, 50):
        assert compute_optimal_k(n, target_ratio=1.0) <= n


def test_min_k_respected_but_capped_at_n():
    assert compute_optimal_k(100, target_ratio=0.01, min_k=5) >= 5
    # min_k can't force keeping more than we have.
    assert compute_optimal_k(2, target_ratio=0.01, min_k=10) == 2


def test_max_k_caps_result():
    assert compute_optimal_k(10000, target_ratio=0.9, max_k=20) == 20


def test_decay_pulls_large_inputs_down():
    # Higher decay -> fewer kept for the same large n.
    assert compute_optimal_k(10000, 0.3, decay=0.8) < compute_optimal_k(10000, 0.3, decay=0.2)
