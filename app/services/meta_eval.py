"""人工抽检 / LLM 裁判元评估。

从已有 EvalResult 中随机抽样、导出双盲打分模板、回填后计算 Spearman ρ、
Cohen's κ、Pearson r 三个对齐指标。结果以 JSON 文件形式按 run_id 持久化。
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from app.models import EvalResult, EvalSample, SystemResponse


def _meta_eval_dir(base_dir: Path) -> Path:
    """约定 base_dir = config.db_path.parent（即 app/data/）。"""
    target = base_dir / "meta_evals"
    target.mkdir(parents=True, exist_ok=True)
    return target


def sample_for_review(
    results: list[EvalResult],
    responses: list[SystemResponse],
    samples_by_id: dict[str, EvalSample],
    n: int = 30,
    seed: int | None = None,
    max_contexts: int = 3,
    context_max_chars: int = 500,
) -> pd.DataFrame:
    """从可信样本中随机抽 N 条，构造双盲打分模板 DataFrame。

    可信样本定义：response.success=True 且 result.evaluation_status 不是 judge_failed。
    返回随机洗牌后的 DataFrame，列名固定为模板 schema。
    """
    responses_by_id = {r.response_id: r for r in responses}
    candidates: list[tuple[EvalResult, SystemResponse]] = []
    for result in results:
        if getattr(result, "evaluation_status", "") == "judge_failed":
            continue
        response = responses_by_id.get(result.response_id)
        if response is None or not response.success:
            continue
        candidates.append((result, response))

    if not candidates:
        return pd.DataFrame(columns=[
            "question_id", "question", "answer",
            "retrieved_contexts", "human_overall_score", "human_notes",
        ])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(candidates))[: min(n, len(candidates))]

    rows = []
    for idx in indices:
        result, response = candidates[idx]
        sample = samples_by_id.get(result.question_id)
        question = response.question or (sample.question if sample else "")
        ctxs = [c for c in (response.retrieved_contexts or []) if str(c).strip()]
        truncated = [
            f"[{i + 1}] {str(c)[:context_max_chars]}"
            for i, c in enumerate(ctxs[:max_contexts])
        ]
        rows.append({
            "question_id": result.question_id,
            "question": question,
            "answer": response.answer or "",
            "retrieved_contexts": "\n\n".join(truncated),
            "human_overall_score": "",
            "human_notes": "",
        })
    return pd.DataFrame(rows)


def export_template_csv(df: pd.DataFrame) -> bytes:
    """utf-8-sig 编码，与现有 offer_dataframe_download 一致，避免 Excel 中文乱码。"""
    return df.to_csv(index=False).encode("utf-8-sig")


def import_annotations(
    uploaded_bytes: bytes,
    expected_qids: list[str],
) -> tuple[dict[str, dict], list[str]]:
    """解析回填的 CSV，返回 ({qid: {"score": int, "notes": str}}, warnings)。

    校验规则：
    - human_overall_score 必须是 1-5 整数；为空 / 非数字 / 越界 → 跳过 + warning
    - 不在 expected_qids 内的行 → 跳过 + warning（防止用户改动了 question_id）
    """
    warnings: list[str] = []
    expected = set(expected_qids)

    try:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception as exc:
        return {}, [f"无法解析 CSV：{exc}"]

    required_cols = {"question_id", "human_overall_score"}
    missing = required_cols - set(df.columns)
    if missing:
        return {}, [f"CSV 缺少必需列：{sorted(missing)}"]

    annotations: dict[str, dict] = {}
    for row_idx, row in df.iterrows():
        qid = str(row["question_id"]).strip()
        if not qid or qid == "nan":
            continue
        if qid not in expected:
            warnings.append(f"第 {row_idx + 2} 行 question_id={qid} 不在抽样集合中，已跳过")
            continue
        raw = row["human_overall_score"]
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        try:
            score = int(float(raw))
        except (TypeError, ValueError):
            warnings.append(f"第 {row_idx + 2} 行评分非数字（{raw}），已跳过")
            continue
        if not 1 <= score <= 5:
            warnings.append(f"第 {row_idx + 2} 行评分越界（{score}，应 1-5），已跳过")
            continue
        notes = "" if pd.isna(row.get("human_notes")) else str(row.get("human_notes", "")).strip()
        annotations[qid] = {"score": score, "notes": notes}
    return annotations, warnings


def _bucket_judge_score(value: float) -> int:
    """将 0-1 连续分映射到 1-5 离散等级，与人工 1-5 同尺度对齐 κ。"""
    if value < 0.2:
        return 1
    if value < 0.4:
        return 2
    if value < 0.6:
        return 3
    if value < 0.8:
        return 4
    return 5


def compute_alignment(
    human: dict[str, int],
    judge: dict[str, float],
) -> dict:
    """计算 Spearman ρ、Pearson r、二次加权 Cohen's κ。

    n<3 时统计量无意义，返回 None 字段；ρ/r 在零方差时 scipy 返回 nan，统一转 None。
    """
    common = sorted(set(human) & set(judge))
    n = len(common)
    out: dict = {
        "n_pairs": n,
        "spearman_rho": None,
        "spearman_p": None,
        "pearson_r": None,
        "pearson_p": None,
        "kappa_quadratic": None,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    if n < 3:
        return out

    h = np.array([human[q] for q in common], dtype=float)
    j = np.array([judge[q] for q in common], dtype=float)
    j_buckets = np.array([_bucket_judge_score(v) for v in j], dtype=int)
    h_int = h.astype(int)

    def _f(x):
        x = float(x)
        return None if np.isnan(x) else round(x, 4)

    try:
        rho, p = stats.spearmanr(h, j)
        out["spearman_rho"] = _f(rho)
        out["spearman_p"] = _f(p)
    except Exception:
        pass
    try:
        r, p = stats.pearsonr(h, j)
        out["pearson_r"] = _f(r)
        out["pearson_p"] = _f(p)
    except Exception:
        pass
    try:
        kappa = cohen_kappa_score(h_int, j_buckets, weights="quadratic", labels=[1, 2, 3, 4, 5])
        out["kappa_quadratic"] = _f(kappa)
    except Exception:
        pass
    return out


def save_meta_eval(run_id: str, payload: dict, base_dir: Path) -> Path:
    path = _meta_eval_dir(base_dir) / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_meta_eval(run_id: str, base_dir: Path) -> dict | None:
    path = _meta_eval_dir(base_dir) / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
