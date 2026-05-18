"""Semantic similarity scoring using local sentence-transformers models."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from app.models.schemas import EvalResult, EvalSample, ScoreItem, SystemResponse

if TYPE_CHECKING:
    from app.core.config import AppConfig

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "多语言轻量模型（约420MB，中英文均可，推荐首选）",
    "paraphrase-multilingual-mpnet-base-v2": "多语言高质量模型（约1GB，效果更佳）",
    "shibing624/text2vec-base-chinese": "中文专用模型（约400MB，纯中文场景最佳）",
}

_model_cache: dict = {}

# 默认补全的命名空间：sentence-transformers 上传方
_DEFAULT_ORG = "sentence-transformers"


class EmbeddingModelNotCachedError(RuntimeError):
    """本地未缓存目标嵌入模型，且调用方未授权联网下载。"""

    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = cache_dir
        super().__init__(
            f"嵌入模型 {model_name} 未在本地缓存（缓存目录：{cache_dir}），且未授权联网下载。"
        )


def get_hf_cache_dir() -> Path:
    """返回 HuggingFace Hub 模型缓存目录（遵循 HF_HOME / HUGGINGFACE_HUB_CACHE 环境变量）。"""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _candidate_cache_dirs(model_name: str) -> list[Path]:
    """列出该模型可能落地的缓存目录候选。"""
    hub = get_hf_cache_dir()
    candidates: list[Path] = []
    if "/" in model_name:
        org, name = model_name.split("/", 1)
        candidates.append(hub / f"models--{org}--{name}")
    else:
        candidates.append(hub / f"models--{_DEFAULT_ORG}--{model_name}")
        candidates.append(hub / f"models--{model_name}")
    # 兼容旧版 sentence-transformers 的 torch 缓存路径
    st_legacy = Path.home() / ".cache" / "torch" / "sentence_transformers"
    candidates.append(st_legacy / model_name.replace("/", "_"))
    return candidates


def is_model_cached(model_name: str) -> bool:
    """检测目标嵌入模型是否已存在于本机缓存（含至少一个 snapshot 目录）。"""
    for d in _candidate_cache_dirs(model_name):
        if not d.exists():
            continue
        # HF hub 结构：models--xxx/snapshots/<rev>/<files>
        snap_dir = d / "snapshots"
        if snap_dir.exists() and any(snap_dir.iterdir()):
            return True
        # 旧版直接放模型文件
        if any(d.glob("*.bin")) or any(d.glob("*.safetensors")) or (d / "config.json").exists():
            return True
    return False


def expected_cache_path(model_name: str) -> Path:
    """返回该模型推荐的下载落地目录（用于提示用户手动下载到哪里）。"""
    return _candidate_cache_dirs(model_name)[0]


def _load_model(model_name: str, allow_download: bool = False):
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "语义嵌入评估需要 sentence-transformers，请运行：pip install sentence-transformers"
        ) from exc

    cached = is_model_cached(model_name)
    if not cached and not allow_download:
        raise EmbeddingModelNotCachedError(model_name, expected_cache_path(model_name))

    logger.info("加载嵌入模型 %s (allow_download=%s) ...", model_name, allow_download)
    # 已缓存时强制走 local_files_only，避免内网环境因 HF 探活请求被阻塞而长时间转圈
    if cached and not allow_download:
        _model_cache[model_name] = SentenceTransformer(model_name, local_files_only=True)
    else:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _cos_sim(model, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    from sentence_transformers import util
    ea = model.encode(a, convert_to_tensor=True)
    eb = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _si(v: float, reason: str) -> ScoreItem:
    v = _clamp(v)
    return ScoreItem(raw_score=round(v, 4), normalized_score=round(v, 4), reason=reason)


class EmbeddingEvaluator:
    def __init__(self, config: "AppConfig", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.config = config
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def evaluate_one(self, item: tuple) -> EvalResult:
        sample, response = item
        reference = response.reference_answer or (sample.reference_answer if sample else "")
        expected_evidence = sample.expected_evidence if sample else ""
        answer = response.answer or ""
        question = response.question or ""
        contexts = response.retrieved_contexts + response.citations

        if not response.success or not answer.strip():
            return self._empty_result(response, "答案为空或请求失败。")

        from app.services.evaluator import is_answer_only
        if is_answer_only(response):
            contexts = []  # 排除 [""] 这类伪空列表

        m = self.model
        sim_ref = _cos_sim(m, answer, reference) if reference else 0.5
        sim_q = _cos_sim(m, answer, question)

        scores = {
            "correctness": _si(sim_ref, "答案与参考答案的语义相似度（嵌入模型计算）。"),
            "relevance": _si(max(sim_q, sim_ref * 0.7), "答案与问题的语义相关性。"),
            "completeness": _si(sim_ref, "答案语义覆盖参考答案的程度（嵌入相似度估计）。"),
        }
        answer_only = len(contexts) == 0
        if not answer_only:
            sim_ctx = max(_cos_sim(m, answer, ctx) for ctx in contexts)
            ctx_q_sims = [_cos_sim(m, question, ctx) for ctx in contexts]
            ctx_relevance = mean(ctx_q_sims)
            ctx_precision = sum(1 for s in ctx_q_sims if s > 0.4) / len(contexts)
            ctx_recall = max(_cos_sim(m, reference, ctx) for ctx in contexts) if reference else 0.0
            evidence_cov = (
                max(_cos_sim(m, expected_evidence, ctx) for ctx in contexts)
                if expected_evidence else ctx_recall
            )
            hit = 1.0 if ctx_relevance > 0.4 or evidence_cov > 0.4 else 0.0
            hallucination = _clamp(1.0 - sim_ctx)
            scores["faithfulness"] = _si(sim_ctx, "答案与检索上下文的语义匹配程度。")
            scores["hallucination_risk"] = ScoreItem(
                raw_score=round(hallucination, 4),
                normalized_score=round(hallucination, 4),
                reason="答案缺乏上下文支撑的部分估计（越低越好）。",
            )
            scores["hit_rate"] = _si(hit, "检索上下文是否命中了与问题相关的内容。")
            scores["context_relevance"] = _si(ctx_relevance, "检索上下文与问题的语义相关性均值。")
            scores["context_precision"] = _si(ctx_precision, "检索片段中相关片段的比例（语义阈值 0.4）。")
            scores["context_recall"] = _si(ctx_recall, "参考答案内容被检索上下文覆盖的程度。")
            scores["evidence_coverage"] = _si(evidence_cov, "期望证据被检索上下文覆盖的程度。")

        agg = [
            (1 - scores["hallucination_risk"].normalized_score) if k == "hallucination_risk"
            else scores[k].normalized_score
            for k in scores
        ]
        normalized = mean(agg)
        labels = self._failure_labels(response, scores)

        prefix = "[仅答案模式] " if answer_only else ""
        return EvalResult(
            question_id=response.question_id,
            response_id=response.response_id,
            scores=scores,
            normalized_score=round(normalized, 4),
            judge_reason=f"{prefix}语义嵌入评分（模型：{self.model_name}）",
            judge_model=f"embedding:{self.model_name}",
            score_version=self.config.score_version,
            failure_labels=labels,
        )

    def _failure_labels(self, response: SystemResponse, scores: dict) -> list[str]:
        if not response.success:
            return ["cannot_judge"]
        labels = []
        if "correctness" in scores and scores["correctness"].normalized_score < 0.4:
            labels.append("wrong_answer")
        if "completeness" in scores and scores["completeness"].normalized_score < 0.4:
            labels.append("incomplete_answer")
        if "faithfulness" in scores and scores["faithfulness"].normalized_score < 0.4:
            labels.append("unsupported_answer")
        if response.retrieved_contexts and "hit_rate" in scores and scores["hit_rate"].normalized_score < 0.5:
            labels.append("retrieval_issue")
        if response.retrieved_contexts and "evidence_coverage" in scores and scores["evidence_coverage"].normalized_score < 0.4:
            labels.append("missing_evidence")
        return labels

    def _empty_result(self, response: SystemResponse, reason: str) -> EvalResult:
        scores = {
            name: ScoreItem(raw_score=0.0, normalized_score=0.0, reason=reason)
            for name in [
                "correctness", "relevance", "faithfulness", "completeness", "hallucination_risk",
                "hit_rate", "context_relevance", "context_precision", "context_recall", "evidence_coverage",
            ]
        }
        return EvalResult(
            question_id=response.question_id,
            response_id=response.response_id,
            scores=scores,
            normalized_score=0.0,
            judge_reason=reason,
            judge_model=f"embedding:{self.model_name}",
            score_version=self.config.score_version,
            failure_labels=["cannot_judge"],
        )
