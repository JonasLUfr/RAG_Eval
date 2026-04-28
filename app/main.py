from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import load_config
from app.core.logging_config import setup_logging
from app.models import EvalSample, ExperimentRun, ProjectContext, SystemResponse
from app.models.schemas import FAILURE_LABELS
from app.services.comparison import comparison_dataframe, summarize_run
from app.services.connectors import ConnectorConfig, ExternalAPIConnector
from app.services.evaluator import (
    EvaluationEngine,
    METRIC_COMBINATION_GUIDE,
    METRIC_DETAILED_INFO,
    METRIC_MODE_EXPLANATIONS,
    METRIC_USER_INFO,
    is_answer_only,
)
from app.services.exporter import ExportCenter
from app.services.importer import dataframe_to_responses, dataframe_to_samples, read_uploaded_table
from app.services.llm_client import OpenAICompatibleClient
from app.services.seed import ensure_seed_data
from app.services.source_loader import SourceFileTooLargeError, parse_source_file
from app.services.testset_generator import TestsetGenerator, TestsetLLMSettings
from app.storage import SQLiteStore


st.set_page_config(
    page_title="RAG_Eval 评测工作台",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


def _answer_only_badge(responses: list[SystemResponse]) -> str | None:
    """若批次内存在仅答案模式样本，返回提示文案；否则返回 None。

    检测复用 evaluator.is_answer_only：判定标准为
      retrieved_contexts 为空列表/None **且** citations 为空列表/None。
    与评估器逐样本判定使用同一 predicate，保证显示数 = 实际触发数。
    """
    if not responses:
        return None
    n = sum(1 for r in responses if is_answer_only(r))
    if n == 0:
        return None
    total = len(responses)
    return (
        f"本批 {total} 条样本中，**{n} 条**因被测系统未返回检索证据（无 contexts 且无 citations），"
        "已自动按「仅答案模式」评估，仅产出 3 个证据无关指标："
        "**correctness / relevance / completeness**。"
        "其余 7 个指标对这些样本不适用，因此 scores 字典与综合得分均按 3 指标聚合，避免假 0 分拖累分数。"
    )


def _show_df(df: pd.DataFrame, *, hide_index: bool = True, column_config=None, **_) -> None:
    """st.dataframe wrapper that falls back to a markdown table when PyArrow is unavailable."""
    if _HAS_PYARROW:
        st.dataframe(df, use_container_width=True, hide_index=hide_index, column_config=column_config)
        return
    if df.empty:
        st.caption("暂无数据")
        return
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join([":---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        cells = [str(v).replace("|", "｜").replace("\n", " ")[:150] for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))


def _edit_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """st.data_editor wrapper that keeps the app usable when PyArrow is unavailable."""
    if _HAS_PYARROW:
        return st.data_editor(df, **kwargs)
    st.warning("PyArrow 无法加载，表格编辑暂时不可用；当前以只读表格显示。")
    _show_df(df, hide_index=kwargs.get("hide_index", True))
    return df.copy()


def _show_bar_chart(df: pd.DataFrame, **kwargs) -> None:
    """st.bar_chart wrapper that falls back to a matplotlib bar chart when PyArrow is unavailable."""
    if _HAS_PYARROW:
        st.bar_chart(df, **kwargs)
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        df.plot(kind="bar", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        _show_df(df.reset_index())


def _show_altair(chart, **kwargs) -> None:
    """st.altair_chart wrapper that falls back to _show_bar_chart when PyArrow is unavailable."""
    if _HAS_PYARROW:
        st.altair_chart(chart, **kwargs)
        return
    # Altair chart objects expose their data; extract and show as table
    try:
        data = chart.data
        if hasattr(data, "to_dict"):
            _show_df(data)
        else:
            st.caption("图表不可用（PyArrow 未加载）")
    except Exception:
        st.caption("图表不可用（PyArrow 未加载）")


def _show_horizontal_bar_chart(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    color: str = "#0f6f68",
    height_per_row: int = 34,
    value_format: str = ".3f",
    text_label: str | None = None,
) -> None:
    """Render readable horizontal bars with labels that do not rotate vertically."""
    if df.empty or category not in df.columns or value not in df.columns:
        _show_df(df)
        return
    cols = [category, value]
    if text_label and text_label in df.columns:
        cols.append(text_label)
    chart_df = df[cols].copy()
    chart_df[value] = pd.to_numeric(chart_df[value], errors="coerce").fillna(0)
    chart_df[category] = chart_df[category].astype(str)
    if text_label and text_label in chart_df.columns:
        chart_df[text_label] = chart_df[text_label].astype(str)
    height = max(180, min(520, len(chart_df) * height_per_row + 42))
    try:
        import altair as alt

        tooltip = [alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=value_format)]
        if text_label and text_label in chart_df.columns:
            tooltip.append(alt.Tooltip(f"{text_label}:N", title="标签"))
        bar = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=4, height=18)
            .encode(
                y=alt.Y(f"{category}:N", sort="-x", title="", axis=alt.Axis(labelLimit=260)),
                x=alt.X(f"{value}:Q", title="", axis=alt.Axis(grid=True)),
                color=alt.value(color),
                tooltip=tooltip,
            )
        )
        label = (
            alt.Chart(chart_df)
            .mark_text(align="left", dx=6, color="#5b677a", fontSize=12)
            .encode(
                y=alt.Y(f"{category}:N", sort="-x", title=""),
                x=alt.X(f"{value}:Q", title=""),
                text=alt.Text(f"{text_label}:N") if text_label and text_label in chart_df.columns else alt.Text(f"{value}:Q", format=value_format),
            )
        )
        _show_altair((bar + label).properties(height=height), use_container_width=True)
    except Exception:
        _show_bar_chart(chart_df.set_index(category)[[value]])


def _show_grouped_horizontal_bar_chart(
    df: pd.DataFrame,
    *,
    category: str,
    value_columns: list[str],
    height_per_row: int = 42,
) -> None:
    """Render comparison metrics as grouped horizontal bars instead of hard-to-read vertical labels."""
    available = [col for col in value_columns if col in df.columns]
    if df.empty or category not in df.columns or not available:
        _show_df(df)
        return
    chart_df = df[[category, *available]].copy()
    for col in available:
        chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce").fillna(0)
    long_df = chart_df.melt(category, var_name="指标", value_name="数值")
    height = max(220, min(560, chart_df[category].nunique() * len(available) * height_per_row + 40))
    try:
        import altair as alt

        chart = (
            alt.Chart(long_df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y(f"{category}:N", sort="-x", title="", axis=alt.Axis(labelLimit=320)),
                x=alt.X("数值:Q", title=""),
                color=alt.Color(
                    "指标:N",
                    scale=alt.Scale(range=["#0f6f68", "#6aaed6", "#d97706"]),
                    legend=alt.Legend(title=""),
                ),
                yOffset=alt.YOffset("指标:N"),
                tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip("指标:N"), alt.Tooltip("数值:Q", format=".3f")],
            )
            .properties(height=height)
        )
        _show_altair(chart, use_container_width=True)
    except Exception:
        _show_bar_chart(chart_df.set_index(category)[available])


@st.cache_resource
def get_runtime() -> tuple:
    config = load_config()
    setup_logging(config)
    store = SQLiteStore(config.db_path)
    ensure_seed_data(store)
    return config, store


config, store = get_runtime()


def normalize_question(text: str) -> str:
    return "".join(str(text).lower().split())


def samples_to_dataframe(samples: list[EvalSample]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        data = sample.to_dict()
        data["tags"] = ",".join(sample.tags)
        data["source_context_refs"] = ",".join(sample.source_context_refs)
        data["delete"] = False
        rows.append(data)
    return pd.DataFrame(rows)


def dataframe_to_test_samples(df: pd.DataFrame) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for _, row in df.fillna("").iterrows():
        data = row.to_dict()
        data.pop("delete", None)
        if not data.get("question"):
            continue
        if not data.get("question_id"):
            data.pop("question_id", None)
        samples.append(EvalSample.from_dict(data))
    return samples


def results_dataframe(results) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "result_id": result.result_id,
            "question_id": result.question_id,
            "response_id": result.response_id,
            "normalized_score": result.normalized_score,
            "judge_model": result.judge_model,
            "failure_labels": ",".join(result.failure_labels),
            "judge_reason": result.judge_reason,
        }
        for name, score in result.scores.items():
            row[name] = score.normalized_score
        rows.append(row)
    return pd.DataFrame(rows)


def response_review_dataframe(
    responses: list[SystemResponse],
    samples: list[EvalSample],
    results: list | None = None,
) -> pd.DataFrame:
    sample_map = {s.question_id: s for s in samples}
    result_map = {r.response_id: r for r in (results or [])}
    rows = []
    for idx, response in enumerate(responses, start=1):
        sample = sample_map.get(response.question_id)
        result = result_map.get(response.response_id)
        row = {
            "序号": idx,
            "原问题": response.question or (sample.question if sample else ""),
            "RAG回答": response.answer,
            "期望回答": response.reference_answer or (sample.reference_answer if sample else ""),
            "题型": sample.question_type if sample else "",
            "难度": sample.difficulty if sample else "",
            "是否成功": "成功" if response.success else "失败",
            "错误信息": response.error,
            "延迟(ms)": response.latency_ms,
            "Token用量": response.token_usage,
            "检索上下文": "\n".join(response.retrieved_contexts[:3]),
            "引用": "\n".join(response.citations[:5]),
        }
        if result:
            row.update(
                {
                    "综合得分": result.normalized_score,
                    "正确性": result.scores.get("correctness").normalized_score if result.scores.get("correctness") else "",
                    "相关性": result.scores.get("relevance").normalized_score if result.scores.get("relevance") else "",
                    "忠实性": result.scores.get("faithfulness").normalized_score if result.scores.get("faithfulness") else "",
                    "完整性": result.scores.get("completeness").normalized_score if result.scores.get("completeness") else "",
                    "幻觉风险": result.scores.get("hallucination_risk").normalized_score if result.scores.get("hallucination_risk") else "",
                    "检索命中": result.scores.get("hit_rate").normalized_score if result.scores.get("hit_rate") else "",
                    "证据覆盖": result.scores.get("evidence_coverage").normalized_score if result.scores.get("evidence_coverage") else "",
                    "失败标签": ",".join(result.failure_labels),
                    "裁判理由": result.judge_reason,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def overall_judge_summary(summary: dict, results: list) -> str:
    if not results:
        return "当前运行还没有评分结果。请先执行评估。"
    score = summary.get("avg_score", 0)
    success_rate = summary.get("success_rate", 0)
    failures = summary.get("failure_distribution", {})
    metrics = summary.get("metric_summary", {})
    top_failures = sorted(failures.items(), key=lambda item: item[1], reverse=True)[:3]
    weak_metrics = sorted(metrics.items(), key=lambda item: item[1])[:3]
    failure_text = "、".join([f"{k} {v} 次" for k, v in top_failures]) or "暂无明显失败标签"
    weak_text = "、".join([f"{k}={v}" for k, v in weak_metrics]) or "暂无指标"
    if score >= 0.8:
        level = "整体表现较好，可以重点复查低分样本和少数失败标签。"
    elif score >= 0.55:
        level = "整体表现中等，建议优先分析低分题型、证据覆盖和不完整回答。"
    else:
        level = "整体表现偏弱，建议先检查外部 RAG 返回字段映射、检索上下文质量、参考答案匹配和裁判配置。"
    return (
        f"本次运行平均综合得分为 {score}，API 成功率为 {success_rate}。{level} "
        f"主要失败集中在：{failure_text}。相对薄弱的指标是：{weak_text}。"
    )


def offer_dataframe_download(df: pd.DataFrame, file_prefix: str) -> None:
    if df.empty:
        return
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载当前表格 CSV",
        data=csv_data,
        file_name=f"{file_prefix}.csv",
        mime="text/csv",
    )


def metric_combo_findings(metric_summary: dict[str, float]) -> list[str]:
    findings: list[str] = []
    correctness = float(metric_summary.get("correctness", 0))
    faithfulness = float(metric_summary.get("faithfulness", 0))
    relevance = float(metric_summary.get("relevance", 0))
    completeness = float(metric_summary.get("completeness", 0))
    hallucination = float(metric_summary.get("hallucination_risk", 0))
    hit_rate = float(metric_summary.get("hit_rate", 0))
    ctx_precision = float(metric_summary.get("context_precision", 0))
    ctx_recall = float(metric_summary.get("context_recall", 0))
    evidence = float(metric_summary.get("evidence_coverage", 0))

    if correctness < 0.45 and faithfulness < 0.45:
        findings.append("正确性低 + 忠实性低：检索与生成链路同时偏弱，建议先修检索命中与证据约束。")
    if correctness < 0.45 and faithfulness >= 0.6:
        findings.append("正确性低 + 忠实性较高：证据方向可能对，但推理/抽取环节出错。")
    if relevance >= 0.6 and completeness < 0.45:
        findings.append("相关性高 + 完整性低：回答切题但不完整，建议增加必答要点检查。")
    if hit_rate < 0.4 and ctx_recall < 0.4:
        findings.append("检索命中低 + 召回低：关键证据未被拉回，优先优化召回策略。")
    if hit_rate >= 0.6 and ctx_precision < 0.45:
        findings.append("检索命中尚可 + 精确率低：噪声偏高，建议加强重排与去噪。")
    if hallucination > 0.55 and evidence < 0.45:
        findings.append("幻觉风险高 + 证据覆盖低：回答与证据脱节，建议增加拒答和证据门控。")
    if not findings:
        findings.append("未出现明显的高风险指标叠加，建议继续按低分样本做针对性优化。")
    return findings


def generate_llm_guidance(
    summary: dict,
    combo_findings: list[str],
    api_base: str,
    api_key: str,
    model_name: str,
) -> str:
    """在 LLM 评估模式下，基于本次评估结果生成可执行建议。"""
    fallback = (
        "系统建议（规则回退）：优先处理失败最多的标签，并围绕低分指标分阶段优化。"
        + " 指标叠加提示：" + "；".join(combo_findings[:3])
    )
    if not (api_base and api_key and model_name):
        return fallback
    try:
        llm = OpenAICompatibleClient(config, api_base=api_base, api_key=api_key, model=model_name)
        text = llm.chat(
            system_prompt="你是严谨的评测分析师，输出简明、可执行、可落地的建议。",
            user_prompt=f"""
请根据以下评估结果输出中文建议：
1. 先给出总体判断（1-2句）
2. 再给出3-5条修改方向
3. 必须指出指标或指标叠加意味着什么问题
4. 控制在260字以内

评估摘要：
{json.dumps(summary, ensure_ascii=False)}

指标叠加线索：
{json.dumps(combo_findings, ensure_ascii=False)}
""",
            temperature=0.2,
            timeout=90,
        )
        return (text or "").strip()[:1200] or fallback
    except Exception:
        return fallback


def attach_responses_to_samples(
    responses: list[SystemResponse], samples: list[EvalSample]
) -> list[SystemResponse]:
    by_question = {normalize_question(s.question): s for s in samples}
    by_id = {s.question_id: s for s in samples}
    attached: list[SystemResponse] = []
    for response in responses:
        sample = by_id.get(response.question_id) or by_question.get(normalize_question(response.question))
        if sample:
            response.question_id = sample.question_id
            response.question = response.question or sample.question
            response.reference_answer = response.reference_answer or sample.reference_answer
        attached.append(response)
    return attached


def parse_mapping(text: str, fallback: dict[str, str]) -> dict[str, str]:
    try:
        parsed = json.loads(text or "{}")
        return parsed if isinstance(parsed, dict) else fallback
    except json.JSONDecodeError:
        st.warning("字段映射 JSON 无效，已使用默认映射。")
        return fallback


def ui_escape(value) -> str:
    return escape(str(value if value is not None else ""))


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rag-primary: #14b8a6;
            --rag-primary-dark: #0f766e;
            --rag-primary-soft: #dff8f4;
            --rag-bg: #f5f7fa;
            --rag-surface: #ffffff;
            --rag-border: #dde3ea;
            --rag-border-soft: #e8edf3;
            --rag-text: #111827;
            --rag-sub: #5b677a;
            --rag-muted: #8a96a8;
            --rag-green: #16a34a;
            --rag-green-soft: #eaf8ef;
            --rag-amber: #d97706;
            --rag-amber-soft: #fff4de;
            --rag-red: #dc2626;
            --rag-red-soft: #feecec;
            --rag-blue: #2563eb;
            --rag-blue-soft: #eaf1ff;
            --rag-sidebar: #101827;
        }
        html, body, [class*="css"] {
            font-family: Inter, "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
            font-size: 14px;
            line-height: 1.55;
        }
        .stApp {
            background: var(--rag-bg);
            color: var(--rag-text);
        }
        .block-container {
            padding-top: 0.9rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        header[data-testid="stHeader"] {
            background: transparent;
            height: 48px;
            pointer-events: none;
            z-index: 999998;
        }
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] [role="button"] {
            pointer-events: auto;
        }
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0;
        }
        #MainMenu,
        footer {
            visibility: hidden;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            z-index: 999999 !important;
        }
        [data-testid="collapsedControl"] {
            height: 32px !important;
            position: fixed !important;
            top: 16px !important;
            left: 18px !important;
            width: 48px !important;
        }
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {
            align-items: center !important;
            background: transparent !important;
            border: 0 !important;
            border-radius: 6px !important;
            box-shadow: none !important;
            color: inherit !important;
            display: inline-flex !important;
            font-family: Inter, "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif !important;
            font-size: 0 !important;
            font-weight: 900 !important;
            height: 32px !important;
            justify-content: center !important;
            letter-spacing: -0.04em !important;
            min-height: 32px !important;
            padding: 0 !important;
            width: 48px !important;
        }
        [data-testid="collapsedControl"] button:hover,
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: rgba(20, 184, 166, 0.12) !important;
        }
        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapseButton"] svg {
            display: none !important;
        }
        [data-testid="collapsedControl"] button > *,
        [data-testid="stSidebarCollapseButton"] button > * {
            display: none !important;
        }
        [data-testid="collapsedControl"] button::before {
            color: #0f172a;
            content: ">>";
            font-size: 18px;
        }
        [data-testid="stSidebarCollapseButton"] button::before {
            color: #ffffff;
            content: "<<";
            font-size: 18px;
        }
        [data-testid="stSidebar"] {
            background: var(--rag-sidebar);
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: #e5edf7;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stCaptionContainer {
            color: #94a3b8 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #111c2c;
            border-color: #27364a;
            border-radius: 8px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--rag-text);
            font-family: Inter, "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
        }
        h1 {
            font-size: 1.85rem !important;
            line-height: 1.25 !important;
        }
        h2 {
            font-size: 1.45rem !important;
            line-height: 1.3 !important;
        }
        h3 {
            font-size: 1.12rem !important;
            line-height: 1.35 !important;
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px;
            padding: 8px 14px;
            color: var(--rag-sub);
            font-size: 13px;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--rag-primary-soft);
            color: var(--rag-primary-dark);
            font-weight: 700;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--rag-primary) !important;
            height: 2px;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 6px;
            border-bottom: 1px solid var(--rag-border);
        }
        div[data-testid="stMetric"] {
            background: var(--rag-surface);
            border: 1px solid var(--rag-border-soft);
            border-radius: 10px;
            padding: 18px 18px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--rag-sub);
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            color: var(--rag-text);
        }
        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            border: 1px solid var(--rag-border);
            font-weight: 700;
            min-height: 38px;
            color: var(--rag-text) !important;
            background: #ffffff;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: var(--rag-primary) !important;
            border-color: var(--rag-primary) !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] *,
        .stDownloadButton > button[kind="primary"] *,
        div[data-testid="stFormSubmitButton"] button[kind="primary"] * {
            color: #ffffff !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--rag-primary);
            color: var(--rag-primary-dark) !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: var(--rag-primary-dark) !important;
            border-color: var(--rag-primary-dark) !important;
            color: #ffffff !important;
        }
        .stButton > button:disabled,
        .stDownloadButton > button:disabled,
        div[data-testid="stFormSubmitButton"] button:disabled {
            background: #e8edf3 !important;
            border-color: #d7dee8 !important;
            color: #7a8698 !important;
            opacity: 1 !important;
        }
        .stButton > button:disabled *,
        .stDownloadButton > button:disabled *,
        div[data-testid="stFormSubmitButton"] button:disabled * {
            color: #7a8698 !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            font-size: 14px !important;
            color: var(--rag-text) !important;
        }
        label, [data-testid="stWidgetLabel"] {
            color: var(--rag-text) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"] {
            border: 1px solid var(--rag-border-soft);
            border-radius: 10px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            overflow: hidden;
        }
        .rag-brand {
            padding: 18px 14px 8px;
            margin-bottom: 10px;
        }
        .rag-brand-title {
            color: #ffffff;
            font-size: 30px;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .rag-brand-subtitle {
            color: #94a3b8;
            font-size: 12px;
            margin-top: 6px;
        }
        .rag-sidebar-card {
            background: #111c2c;
            border: 1px solid #27364a;
            border-radius: 10px;
            padding: 14px;
            margin: 12px 0 18px;
        }
        .rag-sidebar-label {
            color: #94a3b8;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: 6px;
        }
        .rag-sidebar-value {
            color: #ffffff;
            font-size: 14px;
            font-weight: 700;
        }
        .rag-sidebar-meta {
            color: #94a3b8;
            font-size: 12px;
            margin-top: 4px;
        }
        .rag-page-hero {
            background: #ffffff;
            border: 1px solid var(--rag-border-soft);
            border-radius: 14px;
            padding: 20px 24px;
            margin: 4px 0 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .rag-eyebrow {
            color: var(--rag-primary-dark);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .rag-page-hero h1.rag-page-title {
            color: var(--rag-text);
            font-size: 30px !important;
            line-height: 1.2;
            font-weight: 800;
            margin: 0;
        }
        .rag-page-desc {
            color: var(--rag-sub);
            font-size: 14px;
            line-height: 1.65;
            margin-top: 8px;
            max-width: 860px;
        }
        .rag-workflow {
            background: #ffffff;
            border: 1px solid var(--rag-border);
            border-radius: 12px;
            padding: 18px 18px 14px;
            margin: 18px 0 0;
        }
        .rag-workflow-title {
            color: var(--rag-text);
            font-size: 15px;
            font-weight: 800;
            margin-bottom: 18px;
        }
        .rag-stepper {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            align-items: start;
            column-gap: 12px;
        }
        .rag-step {
            position: relative;
            text-align: center;
            color: var(--rag-sub);
            font-size: 13px;
            font-weight: 600;
        }
        .rag-step:not(:last-child)::after {
            content: "";
            position: absolute;
            top: 20px;
            left: calc(50% + 30px);
            right: calc(-50% + 30px);
            height: 2px;
            background: #d9e0e8;
        }
        .rag-step.done:not(:last-child)::after,
        .rag-step.active:not(:last-child)::after {
            background: #18c08f;
        }
        .rag-step-dot {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            border-radius: 999px;
            border: 3px solid #d9e0e8;
            background: #f7f9fb;
            color: #5b677a;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }
        .rag-step.done .rag-step-dot {
            background: #18c08f;
            border-color: #18c08f;
            color: #ffffff;
        }
        .rag-step.active .rag-step-dot {
            background: var(--rag-primary-dark);
            border-color: var(--rag-primary-dark);
            color: #ffffff;
        }
        .rag-step.active .rag-step-label {
            color: var(--rag-text);
            font-weight: 800;
        }
        .rag-section-card {
            background: var(--rag-surface);
            border: 1px solid var(--rag-border-soft);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin: 12px 0 18px;
        }
        .rag-section-title {
            color: var(--rag-text);
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .rag-section-desc {
            color: var(--rag-sub);
            font-size: 13px;
            line-height: 1.55;
        }
        .rag-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin: 10px 0 18px;
        }
        .rag-kpi {
            background: var(--rag-surface);
            border: 1px solid var(--rag-border-soft);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .rag-kpi-label {
            color: var(--rag-sub);
            font-size: 12px;
            font-weight: 800;
        }
        .rag-kpi-value {
            color: var(--rag-text);
            font-size: 28px;
            line-height: 1.15;
            font-weight: 800;
            margin-top: 8px;
        }
        .rag-kpi-note {
            color: var(--rag-muted);
            font-size: 12px;
            margin-top: 8px;
        }
        .rag-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 4px 9px;
            font-size: 12px;
            font-weight: 800;
            margin-right: 6px;
            white-space: nowrap;
        }
        .rag-badge.green { background: var(--rag-green-soft); color: var(--rag-green); }
        .rag-badge.amber { background: var(--rag-amber-soft); color: var(--rag-amber); }
        .rag-badge.red { background: var(--rag-red-soft); color: var(--rag-red); }
        .rag-badge.blue { background: var(--rag-blue-soft); color: var(--rag-blue); }
        .rag-actions {
            color: var(--rag-sub);
            font-size: 13px;
            line-height: 1.7;
        }
        .rag-option-card {
            background: #ffffff;
            border: 2px solid var(--rag-border);
            border-radius: 10px;
            padding: 16px 18px;
            margin: 10px 0;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        }
        .rag-option-card.active {
            background: #f0faf8;
            border-color: var(--rag-primary-dark);
        }
        .rag-option-card-link {
            color: inherit !important;
            display: block;
            text-decoration: none !important;
        }
        .rag-option-card-link:hover .rag-option-card {
            background: #f7fffd;
            border-color: var(--rag-primary);
            box-shadow: 0 12px 28px rgba(20, 184, 166, 0.12);
        }
        .rag-option-title {
            color: var(--rag-text);
            font-size: 15px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .rag-option-desc {
            color: var(--rag-sub);
            font-size: 13px;
            line-height: 1.55;
        }
        .st-key-eval_mode_choice div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 12px;
            width: 100%;
        }
        .st-key-eval_mode_choice label {
            background: #ffffff;
            border: 2px solid var(--rag-border);
            border-radius: 10px;
            box-sizing: border-box;
            min-height: 92px;
            margin: 0 !important;
            padding: 16px 18px !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
            width: 100% !important;
            max-width: none !important;
        }
        .st-key-eval_mode_choice,
        .st-key-eval_mode_choice > div,
        .st-key-eval_mode_choice [data-testid="stRadio"] {
            width: 100% !important;
            max-width: none !important;
        }
        .st-key-eval_mode_choice label:hover {
            background: #f7fffd;
            border-color: var(--rag-primary);
            box-shadow: 0 12px 28px rgba(20, 184, 166, 0.10);
        }
        .st-key-eval_mode_choice label:has(input:checked) {
            background: #f0faf8;
            border-color: var(--rag-primary-dark);
        }
        .st-key-eval_mode_choice label p {
            color: var(--rag-text) !important;
            font-size: 15px !important;
            font-weight: 800 !important;
        }
        .st-key-eval_mode_choice label small {
            color: var(--rag-sub) !important;
            font-size: 13px !important;
            line-height: 1.55 !important;
        }
        .rag-summary-card {
            background: linear-gradient(135deg, #ecfdf8 0%, #f8fafc 100%);
            border: 1px solid #b7e8dc;
            border-radius: 12px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            margin: 18px 0 22px;
            padding: 20px 22px;
        }
        .rag-summary-title {
            align-items: center;
            color: var(--rag-text);
            display: flex;
            font-size: 18px;
            font-weight: 850;
            gap: 8px;
            margin-bottom: 8px;
        }
        .rag-summary-title::before {
            background: var(--rag-primary);
            border-radius: 999px;
            content: "";
            display: inline-block;
            height: 10px;
            width: 10px;
        }
        .rag-summary-body {
            color: #475569;
            font-size: 14px;
            line-height: 1.8;
        }
        .rag-export-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 24px;
            margin: 18px 0;
        }
        .rag-export-card {
            background: #ffffff;
            border: 2px solid var(--rag-border);
            border-radius: 12px;
            padding: 28px;
            min-height: 270px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .rag-export-card.featured {
            border-color: #18c08f;
        }
        .rag-export-head {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 14px;
        }
        .rag-export-icon {
            width: 54px;
            height: 54px;
            border-radius: 10px;
            background: #e0f2ef;
            color: var(--rag-primary-dark);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 25px;
            font-weight: 800;
        }
        .rag-export-title {
            color: var(--rag-text);
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .rag-export-desc {
            color: var(--rag-sub);
            font-size: 14px;
            line-height: 1.55;
        }
        .rag-export-list {
            background: #f8fafc;
            border-radius: 8px;
            padding: 14px 16px;
            color: var(--rag-sub);
            font-size: 13px;
            line-height: 1.65;
            margin: 16px 0;
        }
        .rag-button-row {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            align-items: center;
        }
        @media (max-width: 900px) {
            .rag-export-grid { grid-template-columns: 1fr; }
            .rag-stepper { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 18px; }
            .rag-step::after { display: none; }
        }
        .rag-sidebar-about {
            color: #b6c4d6;
            font-size: 12px;
            line-height: 1.65;
            margin: 8px 14px 16px;
        }
        .rag-sidebar-footer {
            border-top: 1px solid #27364a;
            color: #a8b3c3;
            font-size: 10.5px;
            line-height: 1.55;
            margin: 0 14px;
            padding-top: 10px;
            position: fixed;
            bottom: 14px;
            left: 14px;
            width: 210px;
            opacity: 0.78;
        }
        .rag-sidebar-footer div {
            color: #a8b3c3;
            font-size: 10.5px;
            font-weight: 500;
        }
        @media (max-width: 1100px) {
            .rag-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
            .rag-kpi-grid { grid-template-columns: 1fr; }
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str, step: int | None = None) -> None:
    workflow = [
        "项目设置",
        "测试问题集",
        "运行测试",
        "评估分析",
        "实验对比",
        "导出中心",
    ]
    step_html = ""
    if step is not None:
        parts = []
        for index, label in enumerate(workflow, start=1):
            state = "active" if index == step else "done" if index < step else ""
            symbol = "✓" if index < step else "○"
            parts.append(
                f'<div class="rag-step {state}">'
                f'<div class="rag-step-dot">{symbol}</div>'
                f'<div class="rag-step-label">{ui_escape(label)}</div>'
                "</div>"
            )
        step_html = (
            '<div class="rag-workflow"><div class="rag-workflow-title">评测流程</div>'
            f'<div class="rag-stepper">{"".join(parts)}</div></div>'
        )
    html = (
        '<div class="rag-page-hero">'
        '<div class="rag-eyebrow">RAG_Eval 工作区</div>'
        f'<h1 class="rag-page-title">{ui_escape(title)}</h1>'
        f'<div class="rag-page-desc">{ui_escape(description)}</div>'
        f"{step_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_intro(title: str, description: str) -> None:
    html = (
        '<div class="rag-section-card">'
        f'<div class="rag-section-title">{ui_escape(title)}</div>'
        f'<div class="rag-section-desc">{ui_escape(description)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    card_html = "".join(
        (
            '<div class="rag-kpi">'
            f'<div class="rag-kpi-label">{ui_escape(label)}</div>'
            f'<div class="rag-kpi-value">{ui_escape(value)}</div>'
            f'<div class="rag-kpi-note">{ui_escape(note)}</div>'
            "</div>"
        )
        for label, value, note in cards
    )
    st.markdown(f'<div class="rag-kpi-grid">{card_html}</div>', unsafe_allow_html=True)


def render_eval_mode_cards(eval_mode: str) -> str:
    items = [
        (
            "rule",
            "相似度规则评分 (基础)",
            "快速，基于 Jaccard 字符/bigram 重合度算法，将文本拆成字符集合与二元字符组（bigram）的并集后计算交集比，准确性有限。",
        ),
        (
            "embedding",
            "语义嵌入相似度（本地）",
            "本地运行，无需 API，只基于 embedding 模型计算语义相似度，准确性较好。",
        ),
        (
            "llm_judge",
            "LLM 裁判评分 (标准)",
            "每条样本约 1 次 LLM 调用，理解语义与证据关系，Token即时间等成本低于 RAGAS。",
        ),
        (
            "ragas",
            "RAGAS 框架评估 (精确)",
            "最准确，参考Ragas算法，覆盖 faithfulness、relevancy 等多维度，每条样本约 8 次 LLM 调用，若测试集庞大，成本会大幅上升。",
        ),
    ]
    keys = [key for key, _, _ in items]
    title_map = {key: title for key, title, _ in items}
    desc_map = {key: desc for key, _, desc in items}
    if st.session_state.get("eval_mode_choice") not in keys:
        st.session_state["eval_mode_choice"] = eval_mode if eval_mode in keys else "embedding"
    return st.radio(
        "评估方式",
        options=keys,
        format_func=lambda key: title_map[key],
        captions=[desc_map[key] for key in keys],
        key="eval_mode_choice",
        label_visibility="collapsed",
    )


def render_summary_card(summary_text: str) -> None:
    html = (
        '<div class="rag-summary-card">'
        '<div class="rag-summary-title">综合评论</div>'
        f'<div class="rag-summary-body">{ui_escape(summary_text)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_export_cards() -> None:
    st.markdown(
        """
        <div class="rag-export-grid">
          <div class="rag-export-card featured">
            <div class="rag-export-head">
              <div class="rag-export-icon">▦</div>
              <div>
                <div class="rag-export-title">导出 Excel 数据</div>
                <div class="rag-export-desc">包含多个工作表，适合工程复盘、样本排查和二次分析。</div>
              </div>
            </div>
            <div class="rag-export-list">
              包含工作表：<br>
              • experiment_overview - 实验概览<br>
              • sample_details - 样本明细<br>
              • metric_summary - 指标汇总<br>
              • failure_cases - 失败案例<br>
              • question_type_distribution - 题型分布
            </div>
          </div>
          <div class="rag-export-card">
            <div class="rag-export-head">
              <div class="rag-export-icon">▤</div>
              <div>
                <div class="rag-export-title">生成 Markdown 报告</div>
                <div class="rag-export-desc">生成专业评测报告，适合项目汇报、评审和阶段性归档。</div>
              </div>
            </div>
            <div class="rag-export-list">
              报告结构：<br>
              • 项目摘要<br>
              • 实验概况与总体指标<br>
              • 题型分布与得分分析<br>
              • 代表性失败案例<br>
              • 实验对比<br>
              • 改进建议
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="rag-brand"><div class="rag-brand-title">RAG_Eval</div>'
        '<div class="rag-brand-subtitle">RAG / 检索生成评测平台</div></div>'
        '<div class="rag-sidebar-about">'
        '用于评估 Agent 的 RAG、检索生成与问答系统表现，支持测试集管理、批量运行、评分分析和报告导出。'
        '</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    st.sidebar.markdown(
        '<div class="rag-sidebar-footer">'
        '<div>版本：1.0</div>'
        '<div>开发人员：JonasLu.com</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_context(project: ProjectContext | None) -> None:
    if project:
        samples = store.list_test_cases(project.project_id)
        approved = store.list_test_cases(project.project_id, approved_only=True)
        runs = store.list_experiments(project.project_id)
        st.sidebar.markdown(
            '<div class="rag-sidebar-card">'
            '<div class="rag-sidebar-label">当前项目</div>'
            f'<div class="rag-sidebar-value">{ui_escape(project.name)}</div>'
            f'<div class="rag-sidebar-meta">{len(samples)} 个测试用例 · {len(approved)} 个已通过 · {len(runs)} 次运行</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="rag-sidebar-card"><div class="rag-sidebar-label">当前项目</div>'
            '<div class="rag-sidebar-value">尚未创建</div>'
            '<div class="rag-sidebar-meta">先完成项目设置，再生成测试集。</div></div>',
            unsafe_allow_html=True,
        )


def render_dashboard(project: ProjectContext | None) -> None:
    render_page_header(
        "评测总览",
        "集中查看项目状态、测试集准备度、测试运行进展和下一步动作。",
        None,
    )
    if not project:
        st.info("暂无项目。请先进入“项目设置”创建项目，并补充背景、业务规则和评测目标。")
        render_kpi_cards([
            ("项目状态", "未创建", "需要先建立评测上下文"),
            ("测试用例", "0", "生成或上传后进入审核"),
            ("测试运行", "0", "暂无历史记录"),
            ("评估结果", "未评分", "等待系统输出"),
        ])
        return

    samples = store.list_test_cases(project.project_id)
    approved = store.list_test_cases(project.project_id, approved_only=True)
    runs = store.list_experiments(project.project_id)
    latest_run = runs[0] if runs else None
    latest_responses = store.list_system_responses(latest_run.run_id) if latest_run else []
    latest_results = store.list_eval_results(latest_run.run_id) if latest_run else []
    latest_summary = summarize_run(latest_run, latest_responses, latest_results) if latest_run else {}
    avg_score = latest_summary.get("avg_score", 0) if latest_results else "未评分"
    success_rate = latest_summary.get("success_rate", 0) if latest_responses else "暂无"

    render_kpi_cards([
        ("测试用例", str(len(samples)), f"{len(approved)} 个已通过，可运行测试"),
        ("测试运行", str(len(runs)), latest_run.name if latest_run else "尚未创建运行记录"),
        ("平均综合得分", str(avg_score), "最近一次已评分运行" if latest_results else "等待评估结果"),
        ("API 成功率", str(success_rate), "最近一次系统输出统计" if latest_responses else "暂无系统输出"),
    ])

    next_actions = []
    if not samples:
        next_actions.append("生成或上传测试问题集，并进入审核表。")
    elif not approved:
        next_actions.append("将可用测试用例标记为“已通过”。")
    elif not runs:
        next_actions.append("创建历史结果导入或外部 API 运行测试。")
    elif latest_responses and not latest_results:
        next_actions.append("进入“评估与失败分析”，选择评估引擎并开始评分。")
    else:
        next_actions.append("查看失败标签和低分样本，导出阶段报告。")
    if project.uploaded_assets:
        next_actions.append(f"已保存 {len(project.uploaded_assets)} 份项目材料，可继续补充业务规则。")
    else:
        next_actions.append("建议上传小型业务材料，让测试集生成更贴近真实场景。")

    st.markdown(
        '<div class="rag-section-card"><div class="rag-section-title">下一步建议</div>'
        f'<div class="rag-actions">{"<br>".join(ui_escape(item) for item in next_actions)}</div></div>',
        unsafe_allow_html=True,
    )

    if runs:
        rows = []
        for run in runs[:5]:
            responses = store.list_system_responses(run.run_id)
            results = store.list_eval_results(run.run_id)
            summary = summarize_run(run, responses, results)
            rows.append(
                {
                    "运行名称": run.name,
                    "模式": run.mode,
                    "样本数": summary.get("samples", 0),
                    "平均总分": summary.get("avg_score", 0) if results else "未评分",
                    "成功率": summary.get("success_rate", 0) if responses else "暂无",
                    "平均延迟(ms)": summary.get("avg_latency_ms", 0),
                }
            )
        render_section_intro("最近运行", "用于快速判断当前项目是否已经具备可分析结果。")
        _show_df(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("暂无运行记录。完成测试用例审核后，可在“运行测试”中导入历史结果或调用外部 RAG API。")


def current_project_selector() -> ProjectContext | None:
    projects = store.list_project_contexts()
    if not projects:
        return None
    options = {f"{p.name} ({p.project_id})": p.project_id for p in projects}
    selected = st.sidebar.selectbox("当前项目", list(options.keys()))
    return store.get_project_context(options[selected])


def run_with_waiting_ui(task, timeout_seconds: int, title: str):
    """单次 LLM 调用没有真实进度，这里用轮询动画告诉用户当前仍在等待网络返回。"""
    progress = st.progress(0)
    status_text = st.empty()
    started_at = time.monotonic()
    steps = [
        "正在整理项目背景和上传材料...",
        "正在连接 LLM API...",
        "请求已发送，等待模型生成测试问题...",
        "仍在等待返回，办公网络或代理较慢时可能需要更久...",
        "返回后将解析 JSON 并写入测试集审核表...",
    ]
    with st.status(title, expanded=True) as status:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(task)
            tick = 0
            while not future.done():
                elapsed = int(time.monotonic() - started_at)
                step = steps[min(tick // 8, len(steps) - 1)]
                percent = min(95, 8 + int((elapsed / max(timeout_seconds, 1)) * 82))
                progress.progress(percent)
                status_text.info(f"{step} 已等待 {elapsed} 秒，设置超时 {timeout_seconds} 秒。")
                if tick % 10 == 0:
                    status.write(f"{step}（已等待 {elapsed}s）")
                time.sleep(1)
                tick += 1
            result = future.result()
        progress.progress(100)
        status_text.success("LLM 已返回，正在保存结果。")
        status.update(label="LLM 生成完成", state="complete", expanded=False)
        return result


def evaluate_run_now(
    run: ExperimentRun,
    responses: list[SystemResponse],
    samples: list[EvalSample],
    use_llm_judge: bool = False,
) -> tuple[list, dict]:
    """导入或 API 调用完成后可直接评分，减少用户在页面间来回跳转。"""
    samples_by_id = {s.question_id: s for s in samples}
    engine = EvaluationEngine(config, use_llm_judge=use_llm_judge)
    results = engine.evaluate_batch(responses, samples_by_id)
    store.save_eval_results(run.run_id, results)
    summary = summarize_run(run, responses, results)
    run.aggregate = summary
    run.config["score_version"] = config.score_version
    store.save_experiment(run)
    return results, summary


apply_app_theme()
render_sidebar_brand()
project = current_project_selector()
render_sidebar_context(project)
render_sidebar_footer()

tab_dashboard, tab_context, tab_testset, tab_experiment, tab_eval, tab_compare, tab_export = st.tabs(
    ["总览", "项目设置", "测试问题集", "运行测试实验", "评估与失败分析", "实验对比", "导出中心"]
)


with tab_dashboard:
    render_dashboard(project)


with tab_context:
    render_page_header(
        "项目设置",
        "用自然语言沉淀项目背景、被评测系统、评测目标和业务规则，为测试集生成和报告导出提供上下文。",
        1,
    )
    create_new = st.checkbox("新建项目", value=False)
    base = ProjectContext(name="新项目") if create_new or project is None else project

    with st.form("context_form"):
        name = st.text_input("项目名称", value=base.name)
        project_background = st.text_area(
            "项目背景",
            value=base.project_background,
            height=120,
            placeholder="例如：这是一个面向企业销售数据的中文问答AI系统，用户会询问销售额、客户、产品、时间范围等问题。",
        )
        system_description = st.text_area(
            "被评测系统说明",
            value=base.system_description,
            height=100,
            placeholder="例如：被评测对象是一个黑盒 RAG API，会根据数据库/文档检索结果生成中文答案。",
        )
        evaluation_goals = st.text_area(
            "评测目标",
            value=base.evaluation_goals,
            height=100,
            placeholder="例如：重点评估准确性、证据支持、不可回答拒答、业务规则遵循和延迟。",
        )
        business_rules = st.text_area(
            "关键业务规则（可选）",
            value=base.business_rules,
            height=120,
            placeholder="每行一条即可，例如：问题缺少时间范围时不能擅自假设月份。",
        )
        question_type_instructions = st.text_area(
            "期望题型（可选）",
            value=base.question_type_instructions,
            height=90,
            placeholder="例如：事实核对、多跳推理、边界条件、业务规则、不可回答。",
        )
        uploaded_files = st.file_uploader(
            "上传项目材料（CSV / Excel / Markdown / TXT / JSON / SQL，单文件最大 5MB）",
            type=["csv", "xlsx", "xls", "md", "txt", "json", "sql"],
            accept_multiple_files=True,
        )
        replace_assets = st.checkbox("用本次上传材料替换已有材料", value=False)

        with st.expander("高级补充（可选，如不熟悉可不填）"):
            schema_text = st.text_area("Schema / 文档结构", value=base.schema_text, height=100)
            metadata_json = st.text_area("元数据 JSON", value=base.metadata_json, height=80)
            sample_rows = st.text_area("样例行", value=base.sample_rows, height=80)
            few_shot_examples = st.text_area("Few-shot 示例", value=base.few_shot_examples, height=80)

        submitted = st.form_submit_button("保存项目设置")
        if submitted:
            assets = [] if replace_assets else list(base.uploaded_assets)
            parse_errors = []
            for file in uploaded_files or []:
                try:
                    assets.append(parse_source_file(file.name, file.getvalue()))
                except SourceFileTooLargeError as exc:
                    parse_errors.append(str(exc))
                except Exception as exc:  # noqa: BLE001
                    parse_errors.append(f"{file.name} 解析失败：{exc}")

            context = ProjectContext(
                project_id=base.project_id if not create_new else ProjectContext().project_id,
                name=name,
                project_background=project_background,
                system_description=system_description,
                evaluation_goals=evaluation_goals,
                business_rules=business_rules,
                question_type_instructions=question_type_instructions,
                uploaded_assets=assets,
                schema_text=schema_text,
                metadata_json=metadata_json,
                sample_rows=sample_rows,
                few_shot_examples=few_shot_examples,
                created_at=base.created_at,
            )
            store.save_project_context(context)
            if parse_errors:
                st.warning("项目已保存，但部分材料未解析成功：\n" + "\n".join(parse_errors))
            else:
                st.success("项目设置已保存。")
            st.rerun()

    st.subheader("已保存材料")
    if base.uploaded_assets:
        _show_df(
            pd.DataFrame(
                [
                    {
                        "文件名": x.get("file_name"),
                        "类型": x.get("file_type"),
                        "大小(bytes)": x.get("size_bytes"),
                        "行数": x.get("row_count", ""),
                    }
                    for x in base.uploaded_assets
                ]
            ),
            use_container_width=True,
        )
        with st.expander("查看材料抽样内容"):
            for asset in base.uploaded_assets:
                st.markdown(f"**{asset.get('file_name')}**")
                st.code(str(asset.get("excerpt", ""))[:3000])
    else:
        st.info("还没有上传项目材料。可以先保存背景说明，后续再上传。")


with tab_testset:
    render_page_header(
        "测试问题集",
        "通过 LLM 生成或上传已有题集，并在审核表中完成编辑、去重、删除和批准。",
        2,
    )
    if not project:
        st.info("请先创建项目设置。")
    else:
        render_section_intro(
            "题集来源",
            "两条路径会进入同一个审核表：生成适合探索，上传适合复用已有人工题集。",
        )
        generation_mode = st.radio(
            "测试问题集来源",
            ["使用 LLM 生成", "上传已有测试问题集"],
            horizontal=True,
        )

        approve_after_import = st.checkbox("导入/生成后自动标记为已通过", value=False)

        if generation_mode == "使用 LLM 生成":
            st.subheader("LLM 连接")
            # 初始化 session state，优先使用已保存的值
            if "llm_api_base" not in st.session_state:
                st.session_state.llm_api_base = config.llm_api_base
            if "llm_api_key" not in st.session_state:
                st.session_state.llm_api_key = config.llm_api_key
            if "llm_model_name" not in st.session_state:
                st.session_state.llm_model_name = config.llm_model
            if "llm_max_questions" not in st.session_state:
                st.session_state.llm_max_questions = min(30, config.default_max_generated_questions)
            if "llm_timeout_seconds" not in st.session_state:
                st.session_state.llm_timeout_seconds = 60

            col_a, col_b = st.columns(2)
            with col_a:
                api_base = st.text_input(
                    "API Base",
                    value=st.session_state.llm_api_base,
                    placeholder="例如：https://api.openai.com/v1",
                    key="llm_api_base_input",
                )
                model_name = st.text_input("模型名", value=st.session_state.llm_model_name, key="llm_model_name_input")
            with col_b:
                api_key_input = st.text_input(
                    "API Key",
                    type="password",
                    value=st.session_state.llm_api_key,
                    placeholder="不会保存到数据库",
                    key="llm_api_key_input",
                )
                max_questions = st.number_input(
                    "最大生成问题数",
                    min_value=1,
                    max_value=config.default_max_generated_questions,
                    value=st.session_state.llm_max_questions,
                    step=1,
                    key="llm_max_questions_input",
                )
            col_c, col_d, col_e = st.columns(3)
            with col_c:
                timeout_seconds = st.number_input(
                    "超时秒数",
                    min_value=10,
                    max_value=300,
                    value=st.session_state.llm_timeout_seconds,
                    step=5,
                    key="llm_timeout_input",
                )
            with col_d:
                include_ref = st.checkbox("生成参考答案", value=True)
            with col_e:
                include_evidence = st.checkbox("生成期望证据", value=True)
            proxy_url = st.text_input(
                "代理地址（可选）",
                placeholder="例如：http://127.0.0.1:7890；不填则使用系统环境变量代理",
            )
            st.caption(
                "如果网络需要代理，建议填写本机代理地址；如果经常超时，可以先把生成数量降到 5-10 条，并把超时调到 180-300 秒。"
            )

            # 清除按钮
            if st.button("清除本次 LLM 配置"):
                st.session_state.llm_api_base = config.llm_api_base
                st.session_state.llm_api_key = config.llm_api_key
                st.session_state.llm_model_name = config.llm_model
                st.session_state.llm_max_questions = min(30, config.default_max_generated_questions)
                st.session_state.llm_timeout_seconds = 60
                st.rerun()

            # 同步 session state
            st.session_state.llm_api_base = api_base
            st.session_state.llm_api_key = api_key_input
            st.session_state.llm_model_name = model_name
            st.session_state.llm_max_questions = int(max_questions)
            st.session_state.llm_timeout_seconds = int(timeout_seconds)

            question_type_text = st.text_area(
                "本次生成题型要求(推荐填写)",
                value=project.question_type_instructions,
                height=90,
            )
            _gen_spacer, _gen_col = st.columns([4, 1.4])
            with _gen_col:
                generate_cases = st.button("调用 LLM 生成测试问题集", type="primary", use_container_width=True)
            if generate_cases:
                api_key = api_key_input or config.llm_api_key
                if not api_base or not api_key or not model_name:
                    st.error("请填写 API Base、API Key 和模型名。")
                elif not (project.project_background or project.uploaded_assets or project.business_rules):
                    st.error("项目背景、业务规则或上传材料至少需要提供一项。")
                else:
                    try:
                        generator = TestsetGenerator(config)
                        settings = TestsetLLMSettings(
                            api_base=api_base,
                            api_key=api_key,
                            model=model_name,
                            timeout_seconds=int(timeout_seconds),
                            proxy_url=proxy_url.strip(),
                        )
                        question_types = [
                            x.strip(" -，,;；")
                            for x in question_type_text.splitlines()
                            if x.strip()
                        ]
                        generated = run_with_waiting_ui(
                            lambda: generator.generate_with_llm(
                                project,
                                settings,
                                int(max_questions),
                                question_types=question_types,
                                include_reference_answer=include_ref,
                                include_expected_evidence=include_evidence,
                            ),
                            int(timeout_seconds),
                            "正在调用 LLM 生成测试问题集",
                        )
                        if approve_after_import:
                            for item in generated:
                                item.review_status = "已通过"
                        store.save_test_cases(project.project_id, generated)
                        st.success(f"LLM 已生成并保存 {len(generated)} 条测试用例。")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"LLM 生成失败：{exc}")

            with st.expander("没有 API 时仅演示 UI 流程"):
                demo_count = st.number_input("演示样例数量", min_value=1, max_value=20, value=5)
                if st.button("生成演示样例（非正式测试集）"):
                    generator = TestsetGenerator(config)
                    generated = generator.generate_demo(project, int(demo_count))
                    store.save_test_cases(project.project_id, generated)
                    st.success("已生成演示样例。正式评测建议使用 LLM 生成或上传人工测试集。")
                    st.rerun()

        else:
            st.subheader("上传已有测试问题集")
            st.caption("选择这条路径会跳过 LLM 生成，直接进入审核、编辑、去重和批准。")
            uploaded_cases = st.file_uploader(
                "上传 CSV / Excel / JSON，建议包含 question、reference_answer、question_type、expected_evidence",
                type=["csv", "xlsx", "xls", "json"],
                key="case_import",
            )
            if uploaded_cases and st.button("保存上传的测试问题集", type="primary"):
                df = read_uploaded_table(uploaded_cases.name, uploaded_cases.getvalue())
                imported = dataframe_to_samples(df)
                if approve_after_import:
                    for item in imported:
                        item.review_status = "已通过"
                store.save_test_cases(project.project_id, imported)
                st.success(f"已导入 {len(imported)} 条测试用例。")
                st.rerun()

        samples = store.list_test_cases(project.project_id)
        st.subheader(f"测试用例审核表：{len(samples)} 条")
        if samples:
            edited_df = _edit_df(
                samples_to_dataframe(samples),
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "review_status": st.column_config.SelectboxColumn(
                        "review_status",
                        options=["待审核", "已通过", "已拒绝"],
                    ),
                    "difficulty": st.column_config.SelectboxColumn("difficulty", options=["低", "中", "高"]),
                    "delete": st.column_config.CheckboxColumn("删除"),
                },
                column_order=[
                    "delete",
                    "review_status",
                    "question",
                    "question_type",
                    "difficulty",
                    "expected_scope",
                    "reference_answer",
                    "expected_evidence",
                    "tags",
                    "source_context_refs",
                    "generation_method",
                    "question_id",
                ],
                hide_index=True,
            )
            col_save, col_delete, col_approve, col_dedup = st.columns(4)
            with col_save:
                if st.button("保存表格修改"):
                    for _, row in edited_df.iterrows():
                        if bool(row.get("delete")) and row.get("question_id"):
                            store.delete_test_case(row["question_id"])
                    remain = dataframe_to_test_samples(edited_df[edited_df["delete"] == False])  # noqa: E712
                    store.save_test_cases(project.project_id, remain)
                    st.success("测试集修改已保存。")
                    st.rerun()
            with col_delete:
                if st.button("删除选中数据", type="primary"):
                    delete_count = 0
                    for _, row in edited_df.iterrows():
                        if bool(row.get("delete")) and row.get("question_id"):
                            store.delete_test_case(row["question_id"])
                            delete_count += 1
                    if delete_count > 0:
                        st.success(f"已删除 {delete_count} 条数据。")
                        st.rerun()
                    else:
                        st.warning("请先勾选要删除的数据。")
            with col_approve:
                if st.button("全部标记为已通过"):
                    for sample in samples:
                        sample.review_status = "已通过"
                    store.save_test_cases(project.project_id, samples)
                    st.success("已全部标记为已通过。")
                    st.rerun()
            with col_dedup:
                if st.button("按问题文本去重"):
                    generator = TestsetGenerator(config)
                    unique = generator.deduplicate(samples)
                    keep = {s.question_id for s in unique}
                    for sample in samples:
                        if sample.question_id not in keep:
                            store.delete_test_case(sample.question_id)
                    st.success(f"去重完成，保留 {len(unique)} 条。")
                    st.rerun()
        else:
            st.info("还没有测试用例。请使用 LLM 生成，或上传已有测试问题集。")


with tab_experiment:
    render_page_header(
        "运行测试与结果导入",
        "导入已有 RAG 回答，或批量调用外部 RAG API，形成可评分的测试运行记录。",
        3,
    )
    if not project:
        st.info("请先创建项目。")
    else:
        approved_samples = store.list_test_cases(project.project_id, approved_only=True)
        render_kpi_cards([
            ("已通过测试用例", str(len(approved_samples)), "外部 API 模式将使用这些题目"),
            ("单次评估上限", str(config.default_max_eval_questions), "用于控制成本和运行时间"),
            ("最大生成题数", str(config.default_max_generated_questions), "测试集生成配置上限"),
        ])
        st.info("测试问题集已自动保存。外部 API 模式会直接使用上一步审核通过的测试题，无需再次上传。")
        run_name = st.text_input("运行名称", value="测试运行")
        run_limit = st.number_input(
            "本次运行/导入样本上限",
            min_value=1,
            max_value=config.default_max_eval_questions,
            value=min(len(approved_samples) or 20, config.default_max_eval_questions),
        )
        cost_per_1k = st.number_input("估算成本：每 1000 tokens 单价", min_value=0.0, value=0.0, step=0.001, format="%.4f")

        mode = st.radio("输入模式", ["历史结果导入", "外部 API 模式"], horizontal=True)

        if mode == "历史结果导入":
            auto_eval_import = st.checkbox("导入后立即评估", value=True, key="auto_eval_import")
            st.caption("如果上传的数据已经包含 RAG 回答 answer，导入后可以直接评分；之后仍可在“评估与失败分析”中查看和重评。")
            uploaded_results = st.file_uploader(
                "上传历史结果（CSV / Excel / JSON）",
                type=["csv", "xlsx", "xls", "json"],
                key="result_import",
            )
            pasted = st.text_area("或粘贴 CSV / JSON", height=160, placeholder="question,answer,retrieved_contexts...")
            with st.expander("字段映射 JSON"):
                mapping_text = st.text_area(
                    "左侧为内部字段，右侧为导入文件列名",
                    value=json.dumps(
                        {
                            "question_id": "question_id",
                            "question": "question",
                            "reference_answer": "reference_answer",
                            "answer": "answer",
                            "retrieved_contexts": "retrieved_contexts",
                            "citations": "citations",
                            "latency_ms": "latency_ms",
                            "token_usage": "token_usage",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            _import_spacer, _import_col = st.columns([4, 1.4])
            with _import_col:
                create_import_run = st.button("创建运行并保存历史结果", type="primary", use_container_width=True)
            if create_import_run:
                if uploaded_results:
                    raw_df = read_uploaded_table(uploaded_results.name, uploaded_results.getvalue())
                elif pasted.strip().startswith(("[", "{")):
                    raw_df = pd.json_normalize(json.loads(pasted))
                elif pasted.strip():
                    raw_df = pd.read_csv(StringIO(pasted))
                else:
                    st.error("请上传文件或粘贴数据。")
                    raw_df = pd.DataFrame()
                if not raw_df.empty:
                    mapping = parse_mapping(mapping_text, {})
                    responses = dataframe_to_responses(raw_df.head(int(run_limit)), mapping)
                    responses = attach_responses_to_samples(responses, approved_samples)
                    run = ExperimentRun(
                        project_id=project.project_id,
                        name=run_name,
                        mode="historical_import",
                        config={
                            "cost_per_1k_tokens": cost_per_1k,
                            "score_version": config.score_version,
                            "import_columns": list(raw_df.columns),
                        },
                    )
                    store.save_experiment(run)
                    store.save_system_responses(run.run_id, responses)
                    badge = _answer_only_badge(responses)
                    if badge:
                        st.info(badge)
                    if auto_eval_import:
                        with st.spinner("正在根据导入的 RAG 回答执行评估..."):
                            results, summary = evaluate_run_now(run, responses, approved_samples)
                        st.success(
                            f"运行记录已创建，保存 {len(responses)} 条系统输出，并完成 {len(results)} 条评分。请到“评估与失败分析”查看详情。"
                        )
                        st.info(overall_judge_summary(summary, results))
                    else:
                        st.success(f"运行记录已创建，保存 {len(responses)} 条系统输出。下一步到“评估与失败分析”执行评分。")
                    preview_df = response_review_dataframe(
                        responses,
                        approved_samples,
                        results if auto_eval_import else None,
                    )
                    st.subheader("本次导入的 RAG 回答预览")
                    _show_df(preview_df)
                    offer_dataframe_download(preview_df, f"{run.name}_rag_responses")

        else:
            if not approved_samples:
                st.warning("外部 API 模式需要先审核通过测试用例。")
            else:
                st.success(f"将自动使用 {min(len(approved_samples), int(run_limit))} 条已通过测试题调用外部 RAG API。")
            auto_eval_api = st.checkbox("API 调用完成后立即评估", value=True, key="auto_eval_api")
            endpoint = st.text_input("外部 RAG / 生成 API 地址", placeholder="https://example.com/query")
            method = st.selectbox("HTTP 方法", ["POST", "GET"])
            headers_json = st.text_area("请求头 JSON", value='{"Content-Type":"application/json"}', height=80)
            with st.expander("请求体常量字段（可选，适配 Dify 等需要固定参数的接口）", expanded=False):
                st.caption(
                    "这里写**所有样本共用**的固定字段，会作为 payload 基底；下方映射的字段在其上层覆盖。"
                    "outbound key 支持点号路径（如 `inputs.query`）自动展开为嵌套对象。"
                )
                static_payload_text = st.text_area(
                    "常量 JSON",
                    value="{}",
                    height=160,
                    help=(
                        "【这一栏要不要填】**完全可选**——本系统对常量 JSON 没有任何字段要求，留 `{}` 也能跑。\n"
                        "只在目标 API 要求一些「与具体问题无关、所有请求共享」的字段时才用得上（典型：Dify、Coze 这类带租户/会话/用户标识的接口）。\n"
                        "\n"
                        "【与下方映射的关系】这里的字段是请求体**基底**；下方「请求字段映射」里的字段会**覆盖**同名 key。\n"
                        "\n"
                        "─── Dify /chat-messages 参考模板（按需挑选，不是都得填）───\n"
                        '{\n'
                        '  "inputs": {},\n'
                        '  "response_mode": "blocking",\n'
                        '  "conversation_id": "",\n'
                        '  "user": "<你自己起一个标识，例如 eval-bot-001>"\n'
                        '}\n'
                        "\n"
                        "【字段说明（哪些 Dify 真的要、哪些可以省）】\n"
                        "• user：**Dify 一般会要**——它用来做用量统计和限流。建议自己起一个字符串（工号、邮箱前缀、`eval-bot-001` 都行），同一批评估保持一致便于在 Dify 后台过滤这批调用。**不要照抄占位符**。如果你的 Dify 实例没有强制要求，也可以不传。\n"
                        "• inputs：仅当你的 Dify 应用里**定义了自定义变量**才需要填（如 `{\"lang\": \"zh\"}`）；没有自定义变量传 `{}` 即可，多数版本下整个 key 不写也能过。\n"
                        "• response_mode：可选，省略时 Dify 通常按服务端默认走。本系统是同步阻塞调用，不要写 `streaming`（流式 SSE 解析不了）；如果你的 Dify 默认就是 `blocking` 可以不写这条。\n"
                        "• conversation_id：可选。建议**不传或写空字符串**，让每条评估样本独立；只有想测多轮场景时才填上一轮返回的会话 ID。\n"
                        "\n"
                        "【最小可跑配置】如果不确定，先只放 `{\"user\": \"你的标识\"}` 试一次；报缺字段再按报错回填。\n"
                        "\n"
                        "【其他几栏怎么配（Dify 场景）】\n"
                        "1. 「请求头 JSON」必须加 `\"Authorization\": \"Bearer app-你的APIKey\"`，这是 Dify 鉴权，省不掉。\n"
                        "2. 「请求字段映射」改成 `{\"question\": \"query\"}`（Dify 用 `query` 接收问题）。\n"
                        "3. 「响应字段映射」保留 `answer: answer` 即可；Dify 不返回结构化 contexts/citations，本系统会自动进入「仅答案模式」（只算 correctness/relevance/completeness 三项），结果页有蓝色徽章提示。"
                    ),
                )
            col_req, col_resp = st.columns(2)
            with col_req:
                request_mapping_text = st.text_area(
                    "请求字段映射：内部字段 -> API 字段",
                    value=json.dumps({"question": "question", "question_id": "question_id"}, ensure_ascii=False, indent=2),
                    height=150,
                )
            with col_resp:
                response_mapping_text = st.text_area(
                    "响应字段映射：内部字段 -> API JSON 路径",
                    value=json.dumps(
                        {
                            "answer": "answer",
                            "retrieved_contexts": "retrieved_contexts",
                            "citations": "citations",
                            "latency_ms": "latency_ms",
                            "token_usage": "token_usage",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    height=150,
                )
            _api_spacer, _api_col = st.columns([4, 1.4])
            with _api_col:
                run_external_api = st.button("验证并批量调用外部 API", type="primary", use_container_width=True)
            if run_external_api:
                if not approved_samples:
                    st.error("没有已通过测试题。请先在“测试问题集”中审核并标记为已通过。")
                    st.stop()
                connector_config = ConnectorConfig(
                    endpoint=endpoint,
                    method=method,
                    headers_json=headers_json,
                    request_mapping=parse_mapping(request_mapping_text, {"question": "question"}),
                    response_mapping=parse_mapping(response_mapping_text, {"answer": "answer"}),
                    static_payload_json=static_payload_text,
                )
                connector = ExternalAPIConnector(config, connector_config)
                ok, message = connector.validate()
                if not ok:
                    st.error(message)
                else:
                    progress_bar = st.progress(0)
                    status = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress.ratio)
                        status.write(f"已完成 {progress.completed}/{progress.total}，失败 {progress.failed}")

                    selected_samples = approved_samples[: int(run_limit)]
                    responses = connector.run_batch(selected_samples, update_progress)
                    run = ExperimentRun(
                        project_id=project.project_id,
                        name=run_name,
                        mode="external_api",
                        config={
                            "endpoint": endpoint,
                            "method": method,
                            "request_mapping": connector_config.request_mapping,
                            "response_mapping": connector_config.response_mapping,
                            "cost_per_1k_tokens": cost_per_1k,
                            "score_version": config.score_version,
                        },
                    )
                    store.save_experiment(run)
                    store.save_system_responses(run.run_id, responses)
                    badge = _answer_only_badge(responses)
                    if badge:
                        st.info(badge)
                    if auto_eval_api:
                        with st.spinner("API 响应已保存，正在执行评估..."):
                            results, summary = evaluate_run_now(run, responses, selected_samples)
                        st.success(
                            f"外部 API 运行已保存，获得 {len(responses)} 条响应，并完成 {len(results)} 条评分。请到“评估与失败分析”查看详情。"
                        )
                        st.info(overall_judge_summary(summary, results))
                    else:
                        st.success(f"外部 API 运行已保存，获得 {len(responses)} 条响应。下一步到“评估与失败分析”执行评分。")
                    preview_df = response_review_dataframe(
                        responses,
                        selected_samples,
                        results if auto_eval_api else None,
                    )
                    st.subheader("本次 API 返回结果预览")
                    _show_df(preview_df)
                    offer_dataframe_download(preview_df, f"{run.name}_rag_responses")


with tab_eval:
    render_page_header(
        "评估与失败分析",
        "把系统输出转成可执行结论：综合得分、指标短板、失败标签、低分样本和人工修正。",
        4,
    )
    st.warning(
        "LLM 裁判可能产生较高费用。1000 条样本会消耗大量 tokens，因为裁判 prompt 往往包含问题、答案、参考答案和长上下文。建议先小批量验证。"
    )
    runs = store.list_experiments(project.project_id if project else None)
    if not runs:
        st.info("暂无运行记录，请先在“运行测试”中创建。")
    else:
        run_options = {f"{r.name} ({r.run_id})": r.run_id for r in runs}
        run_id = st.selectbox("选择实验", list(run_options.keys()), key="eval_run")
        run = store.get_experiment(run_options[run_id])
        responses = store.list_system_responses(run.run_id)
        samples = store.list_test_cases(run.project_id)
        samples_by_id = {s.question_id: s for s in samples}
        existing_results = store.list_eval_results(run.run_id)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("系统输出数", len(responses))
        col_m2.metric("已评分样本数", len(existing_results))

        with st.expander("删除操作"):
            st.caption("以下操作不可撤销，请确认后再执行！")
            _dcol1, _dcol2 = st.columns(2)
            with _dcol1:
                _confirm_clear = st.checkbox("确认清除评分结果", key="confirm_clear_eval")
                if st.button("清除本次评分结果", disabled=not _confirm_clear, key="btn_clear_eval"):
                    store.delete_eval_results(run.run_id)
                    run.aggregate = {}
                    store.save_experiment(run)
                    st.success("评分结果已清除，实验数据保留。")
                    st.rerun()
            with _dcol2:
                _confirm_del = st.checkbox("确认删除整个实验", key="confirm_delete_run")
                if st.button("删除整个实验", disabled=not _confirm_del, key="btn_delete_run"):
                    store.delete_experiment(run.run_id)
                    st.success("实验已删除（含响应和评分）。")
                    st.rerun()

        with st.expander("各指标说明(通用)"):
            detail_rows = []
            for metric_name, detail in METRIC_DETAILED_INFO.items():
                detail_rows.append(
                    {
                        "指标": detail["label"],
                        "方向": "越高越好" if detail["high_is_good"] else "越低越好",
                        "含义": detail["meaning"],
                        "低分/异常风险": detail["low_risk"],
                        "建议优化方向": detail["improve"],
                    }
                )
            st.caption("单指标说明：用于解释每个分数意味着什么，以及低分时优先该改哪里。")
            _show_df(pd.DataFrame(detail_rows))
            st.caption("指标叠加说明：当多个指标同时异常时，通常指向更具体的问题根因。")
            _show_df(pd.DataFrame(METRIC_COMBINATION_GUIDE))

        st.subheader("评估方式")
        eval_mode = render_eval_mode_cards(st.session_state.get("eval_mode_choice", "embedding"))

        _mode_label = {"rule": "规则", "llm_judge": "LLM 裁判", "ragas": "RAGAS", "embedding": "嵌入相似度"}.get(eval_mode, eval_mode)
        with st.expander(f"当前模式（{_mode_label}）下各指标如何计算", expanded=False):
            _mode_rows = [
                {"指标": METRIC_DETAILED_INFO[name]["label"], "该模式下如何计算": how}
                for name, how in METRIC_MODE_EXPLANATIONS.get(eval_mode, {}).items()
            ]
            _show_df(pd.DataFrame(_mode_rows))

        eval_embedding_model = None
        _ragas_base = _ragas_key = _ragas_model = ""
        _eval_max_contexts: int | None = None
        _eval_context_max_chars: int | None = None

        if eval_mode == "embedding":
            from app.services.embedding_evaluator import EMBEDDING_MODELS
            eval_embedding_model = st.selectbox(
                "嵌入模型（首次使用会自动下载到本地）",
                options=list(EMBEDDING_MODELS.keys()),
                format_func=lambda x: f"{x}  ——  {EMBEDDING_MODELS[x]}",
                key="embedding_model_select",
            )
            st.caption("模型下载后缓存在本机，评估无需网络连接和 API 费用。评估为单线程顺序执行，较慢但更稳定。")
            st.info(
                "Windows 用户注意：首次运行时 Windows 智能应用控制（SAC）可能弹出拦截提示，"
                "请点击【仍然运行】，或在 Windows 安全中心 → 应用和浏览器控制 → 智能应用控制中设为评估模式，"
                "之后无需重复操作。"
            )

        elif eval_mode == "ragas":
            st.warning("RAGAS 每条样本约需 LLM 调用 8 次，建议先用 10-20 条样本估算费用后再扩大规模。")
            _existing_base = st.session_state.get("llm_api_base", "") or config.llm_api_base
            _existing_key = st.session_state.get("llm_api_key", "") or config.llm_api_key
            _existing_model = st.session_state.get("llm_model_name", "") or config.llm_model
            _use_existing = st.checkbox(
                f"使用测试集生成时的 LLM 配置（{_existing_base or '未配置'}）",
                value=bool(_existing_base and _existing_key),
                key="ragas_use_existing",
            )
            if _use_existing and _existing_base and _existing_key:
                _ragas_base, _ragas_key, _ragas_model = _existing_base, _existing_key, _existing_model
                st.caption(f"模型：{_ragas_model}")
            else:
                _rcol1, _rcol2 = st.columns(2)
                with _rcol1:
                    _ragas_base = st.text_input(
                        "API Base", value=config.llm_api_base, key="ragas_api_base_input",
                        placeholder="https://api.openai.com/v1",
                    )
                    _ragas_model = st.text_input("模型名", value=config.llm_model, key="ragas_model_input")
                with _rcol2:
                    _ragas_key = st.text_input(
                        "API Key", type="password", value=config.llm_api_key, key="ragas_api_key_input",
                    )

        elif eval_mode == "llm_judge":
            st.warning("LLM 裁判每条样本通常调用 1 次 LLM，建议先用小批量确认费用和评分稳定性。")
            _existing_base = st.session_state.get("llm_api_base", "") or config.llm_api_base
            _existing_key = st.session_state.get("llm_api_key", "") or config.llm_api_key
            _existing_model = config.judge_model or config.llm_model
            _use_existing = st.checkbox(
                f"使用当前裁判模型配置（{_existing_base or '未配置'}）",
                value=bool(_existing_base and _existing_key),
                key="llm_judge_use_existing",
            )
            if _use_existing and _existing_base and _existing_key:
                _ragas_base, _ragas_key, _ragas_model = _existing_base, _existing_key, _existing_model
                st.caption(f"裁判模型：{_ragas_model}")
            else:
                _jcol1, _jcol2 = st.columns(2)
                with _jcol1:
                    _ragas_base = st.text_input(
                        "API Base", value=config.llm_api_base, key="llm_judge_api_base_input",
                        placeholder="https://api.openai.com/v1",
                    )
                    _ragas_model = st.text_input("裁判模型ID", value=config.judge_model, key="llm_judge_model_input")
                with _jcol2:
                    _ragas_key = st.text_input(
                        "API Key", type="password", value=config.llm_api_key, key="llm_judge_api_key_input",
                    )

        elif eval_mode == "rule":
            st.info(
                "规则评分基于字符/二元组重叠度，速度快但不理解语义，容易高估相似但措辞不同的答案。"
                "建议仅用于快速初筛，正式评估请选择语义嵌入或 RAGAS。"
            )

        if eval_mode in {"ragas", "llm_judge"}:
            with st.expander("RAG 系统上下文参数（不填则使用通用默认值）", expanded=False):
                st.caption(
                    "以下参数控制评估时传入裁判 LLM 的上下文内容量。"
                    "默认值（最多 5 段 / 每段 500 字）适配大多数 RAG 系统，"
                    "**但若你的系统 chunk 较长或 top-k 较大，建议根据实际情况调整，否则评分可能偏低或信息截断。**"
                )
                _ctx_col1, _ctx_col2 = st.columns(2)
                with _ctx_col1:
                    _max_ctx_input = st.number_input(
                        "最多取几段上下文（top-k）",
                        min_value=1, max_value=20,
                        value=config.eval_max_contexts,
                        step=1,
                        key="eval_max_contexts",
                        help="你的 RAG 系统每次检索返回几段，建议与实际 top-k 保持一致。默认 5。",
                    )
                with _ctx_col2:
                    _max_chars_input = st.number_input(
                        "每段上下文最多字符数",
                        min_value=100, max_value=3000,
                        value=config.eval_context_max_chars,
                        step=100,
                        key="eval_context_max_chars",
                        help="你的 RAG 系统 chunk 大小（字符数）。默认 500，较长 chunk 建议调高至 800-1500。",
                    )
                _using_defaults = (
                    _max_ctx_input == config.eval_max_contexts
                    and _max_chars_input == config.eval_context_max_chars
                )
                if _using_defaults:
                    st.warning(
                        "当前使用通用默认参数（5段 / 500字/段）。"
                        "若你的 RAG 系统 chunk 较大或 top-k 较多，评分精度可能受影响，建议填写实际参数。"
                    )
                _eval_max_contexts = int(_max_ctx_input)
                _eval_context_max_chars = int(_max_chars_input)

        _calls_per_sample = {"rule": 0, "embedding": 0, "llm_judge": 1, "ragas": 8}.get(eval_mode, 0)
        _planned_samples = min(len(responses), config.default_max_eval_questions)
        _planned_calls = _calls_per_sample * _planned_samples
        if _calls_per_sample > 0:
            st.caption(
                f"预计 LLM 调用次数：**{_planned_calls}** "
                f"（{eval_mode} 模式 × {_planned_samples} 条样本，每条 {_calls_per_sample} 次）。"
                f" 按 gpt-4o-mini 单价粗估约 $"
                f"{_planned_calls * 0.0008:.2f}（实际以你所用模型为准）。"
            )
            if _planned_calls > config.max_llm_calls_per_run:
                st.error(
                    f"超过单次评估硬上限（{config.max_llm_calls_per_run} 次）。"
                    f"请减少样本数或在环境变量 RAG_EVAL_MAX_LLM_CALLS_PER_RUN 中调整。"
                )

        _btn_spacer, _btn_col = st.columns([5, 1.4])
        with _btn_col:
            start_eval = st.button("开始 / 重新评估", type="primary", use_container_width=True)
        if start_eval:
            _ok = True
            if eval_mode in {"ragas", "llm_judge"} and not (_ragas_base and _ragas_key):
                st.error("请填写所选评估方式的 API Base 和 API Key。")
                _ok = False
            if _ok and _planned_calls > config.max_llm_calls_per_run:
                st.error(
                    f"调用次数 {_planned_calls} 超过硬上限 {config.max_llm_calls_per_run}，已阻止启动。"
                )
                _ok = False
            if _ok:
                limited = responses[: config.default_max_eval_questions]

                # 嵌入模式：首次需要下载模型（~470MB），在进度条启动前显式预加载
                if eval_mode == "embedding" and eval_embedding_model:
                    with st.spinner(
                        f"正在加载嵌入模型 {eval_embedding_model}……"
                        "首次使用需要从 HuggingFace 下载（约470MB），请耐心等待，之后会缓存到本机。"
                    ):
                        from app.services.embedding_evaluator import _load_model
                        try:
                            _load_model(eval_embedding_model)
                            st.success("模型加载完成，开始评估……")
                        except Exception as _e:
                            st.error(f"模型加载失败：{_e}")
                            _ok = False

            if _ok:
                limited = responses[: config.default_max_eval_questions]
                progress_bar = st.progress(0)
                status = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress.ratio)
                    status.write(f"已评分 {progress.completed}/{progress.total}，失败 {progress.failed}")

                engine = EvaluationEngine(
                    config,
                    eval_mode=eval_mode,
                    embedding_model=eval_embedding_model,
                    ragas_api_base=_ragas_base or None,
                    ragas_api_key=_ragas_key or None,
                    ragas_model=_ragas_model or None,
                    eval_max_contexts=_eval_max_contexts,
                    eval_context_max_chars=_eval_context_max_chars,
                )
                results = engine.evaluate_batch(limited, samples_by_id, update_progress)
                store.delete_eval_results(run.run_id)  # 先清旧结果，避免新旧混存
                store.save_eval_results(run.run_id, results)
                summary = summarize_run(run, limited, results)
                combo_findings = metric_combo_findings(summary.get("metric_summary", {}))
                summary["metric_combo_findings"] = combo_findings
                if eval_mode == "ragas":
                    _base = _ragas_base or config.llm_api_base
                    _key = _ragas_key or config.llm_api_key
                    _model = _ragas_model or config.llm_model
                    summary["llm_guidance"] = generate_llm_guidance(
                        summary,
                        combo_findings,
                        _base,
                        _key,
                        _model,
                    )
                else:
                    summary["llm_guidance"] = ""
                run.aggregate = summary
                run.config["score_version"] = config.score_version
                run.config["eval_mode"] = eval_mode
                store.save_experiment(run)
                st.success(f"评估完成，保存 {len(results)} 条评分结果。")
                st.rerun()

        results = store.list_eval_results(run.run_id)
        if results:
            summary = summarize_run(run, responses, results)
            if isinstance(run.aggregate, dict):
                if run.aggregate.get("metric_combo_findings"):
                    summary["metric_combo_findings"] = run.aggregate.get("metric_combo_findings")
                if run.aggregate.get("llm_guidance"):
                    summary["llm_guidance"] = run.aggregate.get("llm_guidance")
            render_summary_card(overall_judge_summary(summary, results))
            badge = _answer_only_badge(responses)
            if badge:
                st.info(badge)
            combo_findings = summary.get("metric_combo_findings") or metric_combo_findings(summary.get("metric_summary", {}))
            if combo_findings:
                st.caption("指标叠加诊断")
                for item in combo_findings:
                    st.write(f"- {item}")
            if run.config.get("eval_mode") == "ragas":
                st.subheader("LLM修改建议")
                st.info(summary.get("llm_guidance") or "本次未生成 LLM 建议。")

            metric_summary = summary.get("metric_summary", {})
            high_risk_count = sum(1 for result in results if result.normalized_score < 0.4)
            render_kpi_cards([
                ("平均综合得分", str(summary.get("avg_score", 0)), "最近一次评分结果"),
                ("成功率", str(summary.get("success_rate", 0)), "系统输出成功占比"),
                ("平均延迟(ms)", str(summary.get("avg_latency_ms", 0)), "来自系统响应字段"),
                ("高风险样本", str(high_risk_count), "综合得分低于 0.4"),
            ])

            metric_rows = []
            for name, value in metric_summary.items():
                label, desc = METRIC_USER_INFO.get(name, (name, ""))
                is_risk = name == "hallucination_risk"
                if is_risk:
                    status = "🔴 偏高" if value > 0.5 else "🟡 注意" if value > 0.3 else "🟢 正常"
                else:
                    status = "🟢 良好" if value >= 0.65 else "🟡 一般" if value >= 0.4 else "🔴 偏低"
                metric_rows.append({"指标": label, "分数": round(value, 3), "说明": desc, "状态": status})
            if metric_rows:
                metric_rows = sorted(metric_rows, key=lambda row: row["分数"], reverse=True)
                metric_df = pd.DataFrame(metric_rows)
                st.subheader("指标概览")
                _show_df(
                    metric_df,
                    column_config={
                        "分数": st.column_config.ProgressColumn("分数", min_value=0, max_value=1, format="%.3f"),
                    },
                )

            # ── 深度分析 ──────────────────────────────────────────────────
            st.subheader("深度分析")
            _vtab1, _vtab2, _vtab3 = st.tabs(
                ["失败标签分布", "综合得分分布", "各题型平均得分"]
            )

            _result_map_v = {r.response_id: r for r in results}

            with _vtab1:
                _fdist = summary.get("failure_distribution", {})
                if _fdist:
                    _LABEL_CN = {
                        "wrong_answer": "答案错误",
                        "incomplete_answer": "答案不完整",
                        "unsupported_answer": "无依据答案",
                        "missing_evidence": "缺少证据",
                        "retrieval_issue": "检索问题",
                        "should_abstain_but_answered": "本应拒答却作答",
                        "ambiguous_question_failure": "问题歧义失败",
                        "cannot_judge": "无法判断",
                    }
                    _fd_df = pd.DataFrame(
                        [
                            {"失败类型": _LABEL_CN.get(k, k), "说明": k, "出现次数": v}
                            for k, v in sorted(_fdist.items(), key=lambda x: x[1], reverse=True)
                        ]
                    )
                    try:
                        import altair as alt
                        _pie1 = (
                            alt.Chart(_fd_df)
                            .mark_arc(innerRadius=55, outerRadius=130)
                            .encode(
                                theta=alt.Theta("出现次数:Q"),
                                color=alt.Color(
                                    "失败类型:N",
                                    legend=alt.Legend(title="失败类型", orient="right"),
                                ),
                                tooltip=["失败类型:N", "出现次数:Q", "说明:N"],
                            )
                            .properties(height=300)
                        )
                        _show_altair(_pie1, use_container_width=True)
                    except Exception:
                        _show_bar_chart(_fd_df.set_index("失败类型")[["出现次数"]])
                    st.caption("各失败类型占比：面积越大说明该类问题越集中，是优先改进方向。")
                    _show_df(_fd_df[["失败类型", "出现次数", "说明"]])
                else:
                    st.success("本次评估无失败标签，系统表现良好。")

            with _vtab2:
                _scores_v = [r.normalized_score for r in results]
                if _scores_v:
                    _low = sum(1 for s in _scores_v if s < 0.4)
                    _mid = sum(1 for s in _scores_v if 0.4 <= s < 0.65)
                    _high = sum(1 for s in _scores_v if s >= 0.65)
                    try:
                        import altair as alt
                        _band_df = pd.DataFrame([
                            {"得分区间": "低分 < 0.4（需排查）", "样本数": _low},
                            {"得分区间": "中等 0.4-0.65（可改进）", "样本数": _mid},
                            {"得分区间": "良好 >= 0.65（达标）", "样本数": _high},
                        ])
                        _pie2 = (
                            alt.Chart(_band_df)
                            .mark_arc(innerRadius=55, outerRadius=130)
                            .encode(
                                theta=alt.Theta("样本数:Q"),
                                color=alt.Color(
                                    "得分区间:N",
                                    scale=alt.Scale(
                                        domain=["低分 < 0.4（需排查）", "中等 0.4-0.65（可改进）", "良好 >= 0.65（达标）"],
                                        range=["#ef4444", "#f59e0b", "#22c55e"],
                                    ),
                                    legend=alt.Legend(title="得分区间", orient="right"),
                                ),
                                tooltip=["得分区间:N", "样本数:Q"],
                            )
                            .properties(height=300)
                        )
                        _show_altair(_pie2, use_container_width=True)
                    except Exception:
                        _show_bar_chart(pd.DataFrame({"低分": [_low], "中等": [_mid], "良好": [_high]}))
                    _c1, _c2, _c3 = st.columns(3)
                    _c1.metric("🔴 低分（<0.4）", _low, help="需重点排查的样本")
                    _c2.metric("🟡 中等（0.4-0.65）", _mid, help="有改进空间的样本")
                    _c3.metric("🟢 良好（≥0.65）", _high, help="表现达标的样本")
                    st.caption(f"共 {len(_scores_v)} 条，平均综合得分 {round(sum(_scores_v)/len(_scores_v), 3)}")

            with _vtab3:
                _type_rows_v = []
                for _resp_v in responses:
                    _samp_v = samples_by_id.get(_resp_v.question_id)
                    _res_v = _result_map_v.get(_resp_v.response_id)
                    if _samp_v and _res_v:
                        _type_rows_v.append({
                            "题型": _samp_v.question_type or "未分类",
                            "综合得分": _res_v.normalized_score,
                        })
                if _type_rows_v:
                    _tdf = (
                        pd.DataFrame(_type_rows_v)
                        .groupby("题型")["综合得分"]
                        .agg(["mean", "count"])
                        .reset_index()
                        .rename(columns={"mean": "平均得分", "count": "样本数"})
                        .sort_values("平均得分")
                    )
                    _tdf["图表标签"] = _tdf.apply(
                        lambda row: f"{row['平均得分']:.3f}（样本数：{int(row['样本数'])}）",
                        axis=1,
                    )
                    _show_horizontal_bar_chart(_tdf, category="题型", value="平均得分", text_label="图表标签")
                    st.caption("红色=偏低，黄色=一般，绿色=良好。得分偏低的题型是系统薄弱点。")
                else:
                    st.info("测试用例需填写题型字段才能展示此图。")

            low_score_rows = []
            response_by_id = {response.response_id: response for response in responses}
            for result in sorted(results, key=lambda item: item.normalized_score)[:8]:
                response = response_by_id.get(result.response_id)
                if not response:
                    continue
                low_score_rows.append(
                    {
                        "原问题": response.question,
                        "综合得分": result.normalized_score,
                        "失败标签": ",".join(result.failure_labels),
                        "裁判理由": result.judge_reason,
                    }
                )
            if low_score_rows:
                render_section_intro(
                    "低分样本队列",
                    "优先复核这些样本：它们通常对应检索缺失、无依据回答、规则错误或参考答案不完整。",
                )
                _show_df(pd.DataFrame(low_score_rows))

            st.subheader("样本级评估明细")
            user_df = response_review_dataframe(responses, samples, results)
            score_filter = st.slider("只看综合得分低于", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
            display_df = user_df.copy()
            if "综合得分" in display_df.columns:
                display_df = display_df[pd.to_numeric(display_df["综合得分"], errors="coerce").fillna(0) <= score_filter]
            _show_df(display_df)
            offer_dataframe_download(user_df, f"{run.name}_evaluation_details")

            with st.expander("查看内部评分字段"):
                _show_df(results_dataframe(results))

            st.subheader("手动修正失败标签")
            label_df = pd.DataFrame(
                [
                    {
                        "result_id": result.result_id,
                        "原问题": next((r.question for r in responses if r.response_id == result.response_id), ""),
                        "failure_labels": ",".join(result.failure_labels),
                    }
                    for result in results
                ]
            )
            edited_labels = _edit_df(label_df, use_container_width=True, hide_index=True)
            if st.button("保存失败标签修正"):
                for _, row in edited_labels.iterrows():
                    labels = [x.strip() for x in str(row["failure_labels"]).split(",") if x.strip()]
                    labels = [x for x in labels if x in FAILURE_LABELS]
                    store.update_eval_failure_labels(row["result_id"], labels)
                st.success("失败标签已更新。")
                st.rerun()
        elif responses:
            st.subheader("尚未评分的 RAG 回答")
            preview_df = response_review_dataframe(responses, samples)
            _show_df(preview_df)
            offer_dataframe_download(preview_df, f"{run.name}_rag_responses")


with tab_compare:
    render_page_header(
        "实验对比",
        "对比不同实验的质量、成功率、延迟、成本和失败分布，辅助判断改动是否真的有效。",
        5,
    )
    runs = store.list_experiments(project.project_id if project else None)
    if len(runs) < 2:
        st.info("至少需要两个运行记录才能对比。")
    else:
        options = {f"{r.name} ({r.run_id})": r.run_id for r in runs}
        selected = st.multiselect("选择至少两个实验", list(options.keys()), default=list(options.keys())[:2])
        summaries = []
        for label in selected:
            run = store.get_experiment(options[label])
            responses = store.list_system_responses(run.run_id)
            results = store.list_eval_results(run.run_id)
            summaries.append(summarize_run(run, responses, results))
        if summaries:
            compare_df = comparison_dataframe(summaries)
            _show_df(compare_df, hide_index=False)
            render_section_intro("质量对比", "横向条形图更适合中文实验名称，便于快速比较平均总分和成功率。")
            _show_grouped_horizontal_bar_chart(compare_df, category="实验名称", value_columns=["平均总分", "成功率"])
            render_section_intro("成本与性能对比", "平均延迟和估算成本量纲不同，建议分别关注排序和异常值。")
            _show_grouped_horizontal_bar_chart(compare_df, category="实验名称", value_columns=["平均延迟(ms)", "估算成本"])
            st.subheader("失败分布")
            for summary in summaries:
                st.write(f"**{summary['name']}**")
                st.json(summary["failure_distribution"])


with tab_export:
    render_page_header(
        "导出中心",
        "将评测结果打包为工程复盘用 Excel，或生成面向 PM、导师和客户的 Markdown 报告。",
        6,
    )
    runs = store.list_experiments(project.project_id if project else None)
    if not project or not runs:
        st.info("请先完成项目和实验。")
    else:
        options = {f"{r.name} ({r.run_id})": r.run_id for r in runs}
        selected_run_label = st.selectbox("选择导出实验", list(options.keys()), key="export_run")
        run = store.get_experiment(options[selected_run_label])
        responses = store.list_system_responses(run.run_id)
        results = store.list_eval_results(run.run_id)
        samples = store.list_test_cases(run.project_id)
        summary = summarize_run(run, responses, results)
        if isinstance(run.aggregate, dict):
            if run.aggregate.get("metric_combo_findings"):
                summary["metric_combo_findings"] = run.aggregate.get("metric_combo_findings")
            if run.aggregate.get("llm_guidance"):
                summary["llm_guidance"] = run.aggregate.get("llm_guidance")
        exporter = ExportCenter(config)

        comparison_rows = []
        compare_labels = st.multiselect("可选：加入对比实验到 Markdown 报告", list(options.keys()))
        for label in compare_labels:
            compare_run = store.get_experiment(options[label])
            comparison_rows.append(
                summarize_run(
                    compare_run,
                    store.list_system_responses(compare_run.run_id),
                    store.list_eval_results(compare_run.run_id),
                )
            )

        render_export_cards()
        _exp_spacer, col_excel, col_md = st.columns([2.6, 1, 1])
        with col_excel:
            if st.button("一键导出 Excel", type="primary", use_container_width=True):
                path = exporter.export_excel(project, run, samples, responses, results, summary)
                st.success(f"Excel 已导出：{path}")
                with Path(path).open("rb") as file:
                    st.download_button("下载 Excel", data=file, file_name=Path(path).name, use_container_width=True)
        with col_md:
            if st.button("一键导出 Markdown", type="primary", use_container_width=True):
                path = exporter.export_markdown(project, run, samples, responses, results, summary, comparison_rows)
                st.success(f"Markdown 已导出：{path}")
                with Path(path).open("rb") as file:
                    st.download_button(
                        "下载 Markdown",
                        data=file,
                        file_name=Path(path).name,
                        mime="text/markdown",
                        use_container_width=True,
                    )
