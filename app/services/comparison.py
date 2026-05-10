from __future__ import annotations

from collections import Counter
from statistics import mean

import numpy as np
import pandas as pd
from scipy import stats

from app.models import EvalResult, ExperimentRun, SystemResponse
from app.services.evaluator import is_answer_only


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Empty input returns (0, 0)."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def paired_diff_stats(
    a_scores: dict[str, float],
    b_scores: dict[str, float],
    n_boot: int = 1000,
) -> dict | None:
    """Paired stats for B - A on shared question_ids.

    Returns None if fewer than 2 paired samples are available.
    """
    common = sorted(set(a_scores) & set(b_scores))
    if len(common) < 2:
        return None
    a = np.array([a_scores[q] for q in common], dtype=float)
    b = np.array([b_scores[q] for q in common], dtype=float)
    diff = b - a
    mean_diff = float(diff.mean())
    lo, hi = bootstrap_ci(diff.tolist(), n_boot=n_boot)
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        try:
            _, p_value = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            p_value = float(p_value)
        except ValueError:
            p_value = 1.0
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    cohens_d = mean_diff / sd if sd > 0 else 0.0
    return {
        "n_pairs": len(common),
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "cohens_d": cohens_d,
    }


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_run(
    run: ExperimentRun,
    responses: list[SystemResponse],
    results: list[EvalResult],
) -> dict:
    success_rate = sum(1 for r in responses if r.success) / len(responses) if responses else 0.0
    answer_only_count = sum(1 for r in responses if is_answer_only(r))
    latency_values = [_to_float(r.latency_ms) for r in responses if r.latency_ms not in (None, "")]
    token_values = [_to_float(r.token_usage) for r in responses if r.token_usage not in (None, "")]
    labels = Counter(label for result in results for label in result.failure_labels)
    metric_summary: dict[str, float] = {}
    if results:
        metric_names = sorted({name for result in results for name in result.scores})
        for name in metric_names:
            values = [result.scores[name].normalized_score for result in results if name in result.scores]
            metric_summary[name] = round(mean(values), 4) if values else 0.0
    estimated_cost = round(
        sum(token_values) / 1000 * _to_float(run.config.get("cost_per_1k_tokens", 0)),
        4,
    )
    return {
        "run_id": run.run_id,
        "name": run.name,
        "mode": run.mode,
        "samples": len(responses),
        "success_rate": round(success_rate, 4),
        "avg_score": round(mean([r.normalized_score for r in results]), 4) if results else 0.0,
        "avg_latency_ms": round(mean(latency_values), 2) if latency_values else 0.0,
        "estimated_cost": estimated_cost,
        "failure_distribution": dict(labels),
        "metric_summary": metric_summary,
        "answer_only_count": answer_only_count,
    }


def comparison_dataframe(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for item in summaries:
        rows.append(
            {
                "实验名称": item["name"],
                "样本数": item["samples"],
                "成功率": item["success_rate"],
                "平均总分": item["avg_score"],
                "平均延迟(ms)": item["avg_latency_ms"],
                "估算成本": item["estimated_cost"],
                "失败标签分布": item["failure_distribution"],
            }
        )
    return pd.DataFrame(rows)
