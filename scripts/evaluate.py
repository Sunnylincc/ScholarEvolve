from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass


@dataclass(slots=True)
class RankingMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    novelty: float
    diversity: float


def precision_at_k(pred: list[str], truth: set[str], k: int) -> float:
    selected = pred[:k]
    return sum(1 for x in selected if x in truth) / max(k, 1)


def recall_at_k(pred: list[str], truth: set[str], k: int) -> float:
    selected = pred[:k]
    return sum(1 for x in selected if x in truth) / max(len(truth), 1)


def mrr(pred: list[str], truth: set[str]) -> float:
    for i, item in enumerate(pred, start=1):
        if item in truth:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred: list[str], truth: set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(pred[:k], start=1):
        rel = 1.0 if item in truth else 0.0
        dcg += rel / math.log2(idx + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(truth)) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def novelty_score(pred: list[str], item_popularity: dict[str, float], k: int) -> float:
    items = pred[:k]
    if not items:
        return 0.0
    return sum(-math.log(item_popularity.get(item, 1e-6) + 1e-9) for item in items) / len(items)


def diversity_score(categories: list[str]) -> float:
    if not categories:
        return 0.0
    counts = Counter(categories)
    probs = [v / len(categories) for v in counts.values()]
    return -sum(p * math.log(p + 1e-9) for p in probs)


def evaluate_ranking(
    pred: list[str], truth: set[str], k: int, categories: list[str], popularity: dict[str, float]
) -> RankingMetrics:
    return RankingMetrics(
        precision_at_k=precision_at_k(pred, truth, k),
        recall_at_k=recall_at_k(pred, truth, k),
        mrr=mrr(pred, truth),
        ndcg_at_k=ndcg_at_k(pred, truth, k),
        novelty=novelty_score(pred, popularity, k),
        diversity=diversity_score(categories[:k]),
    )


def simulate_adaptation(rounds: int = 30, seed: int = 7) -> dict[str, float]:
    random.seed(seed)
    topic_weight = {"ai": 0.0, "systems": 0.0, "bio": 0.0}
    target = "systems"
    reward_sum = 0.0

    for t in range(rounds):
        ranked_topics = sorted(topic_weight, key=topic_weight.get, reverse=True)
        shown = ranked_topics[:2]
        click_prob = 0.75 if target in shown else 0.15
        reward = 1.0 if random.random() < click_prob else 0.0
        reward_sum += reward

        for topic in shown:
            topic_weight[topic] *= 0.95
            topic_weight[topic] += reward if topic == target else -0.1 * reward

    return {"avg_reward": reward_sum / rounds, "final_target_weight": topic_weight[target]}


def run_ablation() -> dict[str, dict[str, float]]:
    baseline = simulate_adaptation(rounds=40, seed=11)
    no_decay = simulate_adaptation(rounds=40, seed=11)
    return {"baseline": baseline, "no_decay_placeholder": no_decay}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation helpers and adaptation simulation")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.ablation:
        print(run_ablation())
    else:
        print(simulate_adaptation())
