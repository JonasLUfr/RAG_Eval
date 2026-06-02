"""主流 IR / RAG 评测使用的秩序感知（rank-aware）检索指标。

提供 MRR、Recall@k、Precision@k 三类指标，可叠加到任意 evaluator 的输出上，
不依赖 LLM 调用、不依赖具体 evaluator 实现。

相关性判定方式：单条 retrieved_context 与 expected_evidence 的字符级 Jaccard
相似度 ≥ RELEVANCE_THRESHOLD 视为相关，与 EvaluationEngine 现有 hit_rate 阈值一致。
这是粗略近似，对于精细评测建议在 expected_evidence 列填更精炼的标准证据短语。
"""
from __future__ import annotations

import json
import re

from app.models.schemas import ScoreItem


RELEVANCE_THRESHOLD = 0.12
DEFAULT_KS = (1, 3, 5)


# 给 UI tooltip 使用：显示标签 + 简短说明
RANK_METRIC_TOOLTIPS: dict[str, tuple[str, str]] = {
    "mrr": (
        "MRR",
        "Mean Reciprocal Rank：第一条命中证据的位置倒数。命中第 1 位 = 1.0，第 2 位 = 0.5，"
        "第 5 位 = 0.2。回答「至少要找对一条且排在前面」的问题。",
    ),
    "recall_at_1": (
        "Recall@1",
        "前 1 条检索结果中是否包含相关证据。等于 0 或 1。"
        "等于 1 表示最相关的内容已经被排到首位。",
    ),
    "recall_at_3": (
        "Recall@3",
        "前 3 条检索结果中是否包含相关证据。从 Recall@1 → Recall@3 的提升说明"
        "证据存在但没排在第 1 位，可能需要重排。",
    ),
    "recall_at_5": (
        "Recall@5",
        "前 5 条检索结果中是否包含相关证据。如果 Recall@5 仍低，说明召回阶段就没找到证据，"
        "需要优化 query、索引或扩大 top-k。",
    ),
    "precision_at_1": (
        "Precision@1",
        "前 1 条检索结果中相关证据的比例。值越高说明首位结果质量越高。",
    ),
    "precision_at_3": (
        "Precision@3",
        "前 3 条检索结果中相关证据的比例。值低表示噪声多，需要重排或减少返回窗口。",
    ),
    "precision_at_5": (
        "Precision@5",
        "前 5 条检索结果中相关证据的比例。Precision@k 随 k 增大通常递减，"
        "递减太快说明长尾噪声多。",
    ),
}


def rank_metric_names(ks: tuple[int, ...] = DEFAULT_KS) -> list[str]:
    names = ["mrr"]
    for k in ks:
        names.append(f"recall_at_{k}")
    for k in ks:
        names.append(f"precision_at_{k}")
    return names


def strict_rank_metric_names(ks: tuple[int, ...] = DEFAULT_KS) -> list[str]:
    names = ["strict_mrr"]
    for k in ks:
        names.append(f"strict_recall_at_{k}")
    for k in ks:
        names.append(f"strict_precision_at_{k}")
    return names


def _tokens(text: str) -> set[str]:
    text = re.sub(r"\s+", "", str(text).lower())
    if not text:
        return set()
    chars = set(text)
    bigrams = {text[i : i + 2] for i in range(max(0, len(text) - 1))}
    return chars | bigrams


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_relevant(context: str, evidence: str, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    return _jaccard(context, evidence) >= threshold


def compute_rank_metrics(
    retrieved_contexts: list[str],
    expected_evidence: str,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, ScoreItem]:
    """计算 rank-aware 检索指标。

    无 expected_evidence 或检索上下文全空 → 返回空 dict（指标对该样本无定义）。
    其他场景：
    - retrieved_contexts 为空但有 evidence → 全 0 分（找不到证据）
    - 单段 expected_evidence 下，Recall@k 退化为 0/1（覆盖与否）
    """
    if not expected_evidence or not str(expected_evidence).strip():
        return {}
    contexts = [c for c in (retrieved_contexts or []) if str(c).strip()]
    out: dict[str, ScoreItem] = {}

    # MRR
    mrr_value = 0.0
    mrr_pos = -1
    for i, c in enumerate(contexts):
        if _is_relevant(c, expected_evidence):
            mrr_value = 1.0 / (i + 1)
            mrr_pos = i + 1
            break
    if mrr_pos > 0:
        mrr_reason = f"首个命中位置 = {mrr_pos}，倒数即得分。"
    else:
        mrr_reason = "前 N 条均未命中预期证据。" if contexts else "无检索上下文。"
    out["mrr"] = ScoreItem(
        raw_score=round(mrr_value, 4),
        normalized_score=round(mrr_value, 4),
        reason=mrr_reason,
    )

    for k in ks:
        top_k = contexts[:k]
        if not top_k:
            recall = 0.0
            precision = 0.0
            r_reason = "无检索上下文。"
            p_reason = "无检索上下文。"
        else:
            hits = sum(1 for c in top_k if _is_relevant(c, expected_evidence))
            recall = 1.0 if hits >= 1 else 0.0  # 单 evidence 下 Recall@k 退化为 0/1
            precision = hits / len(top_k)
            r_reason = f"前 {k} 条命中 {hits} 条相关证据。"
            p_reason = f"前 {k} 条中 {hits}/{len(top_k)} 条相关。"
        out[f"recall_at_{k}"] = ScoreItem(
            raw_score=round(recall, 4),
            normalized_score=round(recall, 4),
            reason=r_reason,
        )
        out[f"precision_at_{k}"] = ScoreItem(
            raw_score=round(precision, 4),
            normalized_score=round(precision, 4),
            reason=p_reason,
        )

    return out


def compute_strict_rank_metrics(
    returned_items: list[str],
    relevant_context_ids: list[str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, ScoreItem]:
    """Compute strict rank metrics from returned IDs and human relevant IDs."""
    relevant = {_normalize_id(x) for x in relevant_context_ids if _normalize_id(x)}
    if not relevant:
        return {}
    ranked = [_candidate_ids(item) for item in returned_items if str(item).strip()]
    if not ranked:
        return {}

    out: dict[str, ScoreItem] = {}
    first_pos = -1
    for i, ids in enumerate(ranked):
        if ids & relevant:
            first_pos = i + 1
            break
    mrr_value = 1.0 / first_pos if first_pos > 0 else 0.0
    out["strict_mrr"] = ScoreItem(
        raw_score=round(mrr_value, 4),
        normalized_score=round(mrr_value, 4),
        reason=f"首个相关 ID 位于第 {first_pos} 位。" if first_pos > 0 else "返回结果未命中人工相关 ID。",
    )

    for k in ks:
        top_k = ranked[:k]
        matched: set[str] = set()
        for ids in top_k:
            matched |= ids & relevant
        recall = len(matched) / len(relevant) if relevant else 0.0
        precision = sum(1 for ids in top_k if ids & relevant) / len(top_k) if top_k else 0.0
        out[f"strict_recall_at_{k}"] = ScoreItem(
            raw_score=round(recall, 4),
            normalized_score=round(recall, 4),
            reason=f"前 {k} 条命中 {len(matched)}/{len(relevant)} 个相关 ID。",
        )
        out[f"strict_precision_at_{k}"] = ScoreItem(
            raw_score=round(precision, 4),
            normalized_score=round(precision, 4),
            reason=f"前 {k} 条中 {sum(1 for ids in top_k if ids & relevant)}/{len(top_k)} 条命中相关 ID。" if top_k else "无返回 ID。",
        )
    return out


def _normalize_id(value: object) -> str:
    return str(value).strip().lower()


def _candidate_ids(item: object) -> set[str]:
    text = str(item).strip()
    if not text:
        return set()
    out = {_normalize_id(text)}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("id", "chunk_id", "chunkId", "doc_id", "docId", "document_id", "source_id", "source"):
            value = parsed.get(key)
            if value not in (None, ""):
                out.add(_normalize_id(value))
    for pattern in (
        r"(?:chunk_id|chunkId|doc_id|docId|document_id|source_id|source|id)\s*[:=]\s*([A-Za-z0-9_.:/#-]+)",
        r"\b([A-Za-z0-9_.-]+#chunk[A-Za-z0-9_.-]+)\b",
    ):
        for match in re.findall(pattern, text):
            out.add(_normalize_id(match))
    return {x for x in out if x}
