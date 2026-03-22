from scripts.evaluate import (
    evaluate_ranking,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    simulate_adaptation,
)


def test_metrics() -> None:
    pred = ["a", "b", "c", "d"]
    truth = {"b", "d"}
    assert precision_at_k(pred, truth, 2) == 0.5
    assert recall_at_k(pred, truth, 4) == 1.0
    assert mrr(pred, truth) == 0.5
    assert ndcg_at_k(pred, truth, 4) > 0.0


def test_ranking_bundle_and_simulation() -> None:
    pred = ["p1", "p2", "p3"]
    truth = {"p2"}
    metrics = evaluate_ranking(pred, truth, 3, ["cs.AI", "cs.LG", "cs.AI"], {"p1": 0.8, "p2": 0.2, "p3": 0.1})
    assert metrics.precision_at_k >= 0.0
    sim = simulate_adaptation(rounds=10, seed=1)
    assert "avg_reward" in sim
