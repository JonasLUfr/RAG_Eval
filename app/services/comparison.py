from __future__ import annotations

from collections import Counter
from statistics import mean

import pandas as pd

from app.models import EvalResult, ExperimentRun, SystemResponse


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
