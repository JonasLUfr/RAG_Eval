from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
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
from app.services.evaluator import EvaluationEngine, METRIC_USER_INFO, SCORING_DEFINITIONS
from app.services.exporter import ExportCenter
from app.services.importer import dataframe_to_responses, dataframe_to_samples, read_uploaded_table
from app.services.seed import ensure_seed_data
from app.services.source_loader import SourceFileTooLargeError, parse_source_file
from app.services.testset_generator import TestsetGenerator, TestsetLLMSettings
from app.storage import SQLiteStore


st.set_page_config(page_title="RAG 评测工作台 MVP", layout="wide")


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
        return "当前实验还没有评分结果。请先执行评估。"
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
        f"本次实验平均综合得分为 {score}，API 成功率为 {success_rate}。{level} "
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


st.sidebar.title("RAG 评测工作台")
st.sidebar.caption("本地中文 MVP，不内置检索引擎。")
project = current_project_selector()

tab_context, tab_testset, tab_experiment, tab_eval, tab_compare, tab_export = st.tabs(
    ["项目设置", "测试问题集", "实验运行", "评估与失败分析", "实验对比", "导出中心"]
)


with tab_context:
    st.header("项目设置")
    st.caption("这里只需要交代项目背景和上传少量材料，让 LLM 后续能理解要评测什么。")
    create_new = st.checkbox("新建项目", value=False)
    base = ProjectContext(name="新项目") if create_new or project is None else project

    with st.form("context_form"):
        name = st.text_input("项目名称", value=base.name)
        project_background = st.text_area(
            "项目背景",
            value=base.project_background,
            height=120,
            placeholder="例如：这是一个面向企业销售数据的中文问答系统，用户会询问销售额、客户、产品、时间范围等问题。",
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

        with st.expander("高级补充（可选，不建议一开始填写）"):
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
        st.dataframe(
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
    st.header("测试问题集")
    if not project:
        st.info("请先创建项目设置。")
    else:
        st.caption("你可以让 LLM 基于项目背景和上传材料生成测试问题，也可以直接上传已经做好的测试问题集。")
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
                "如果办公网络需要代理，建议填写本机代理地址；如果经常超时，可以先把生成数量降到 5-10 条，并把超时调到 180-300 秒。"
            )

            # 清除按钮
            if st.button("清除 LLM 配置"):
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
                "本次生成题型要求",
                value=project.question_type_instructions,
                height=90,
            )
            if st.button("调用 LLM 生成测试问题集", type="primary"):
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
            edited_df = st.data_editor(
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
    st.header("实验运行与历史结果导入")
    if not project:
        st.info("请先创建项目。")
    else:
        approved_samples = store.list_test_cases(project.project_id, approved_only=True)
        st.caption(f"当前已通过测试用例：{len(approved_samples)} 条；MVP 默认单次评估上限：{config.default_max_eval_questions} 条。")
        st.info("测试问题集已自动保存。外部 API 模式会直接使用上一步审核通过的测试题，无需再次上传。")
        run_name = st.text_input("实验名称", value="MVP 实验运行")
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
            if st.button("创建实验并保存历史结果", type="primary"):
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
                    if auto_eval_import:
                        with st.spinner("正在根据导入的 RAG 回答执行评估..."):
                            results, summary = evaluate_run_now(run, responses, approved_samples)
                        st.success(
                            f"实验已创建，保存 {len(responses)} 条系统输出，并完成 {len(results)} 条评分。请到“评估与失败分析”查看详情。"
                        )
                        st.info(overall_judge_summary(summary, results))
                    else:
                        st.success(f"实验已创建，保存 {len(responses)} 条系统输出。下一步到“评估与失败分析”执行评分。")
                    preview_df = response_review_dataframe(
                        responses,
                        approved_samples,
                        results if auto_eval_import else None,
                    )
                    st.subheader("本次导入的 RAG 回答预览")
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
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
            if st.button("验证并批量调用外部 API", type="primary"):
                if not approved_samples:
                    st.error("没有已通过测试题。请先在“测试问题集”中审核并标记为已通过。")
                    st.stop()
                connector_config = ConnectorConfig(
                    endpoint=endpoint,
                    method=method,
                    headers_json=headers_json,
                    request_mapping=parse_mapping(request_mapping_text, {"question": "question"}),
                    response_mapping=parse_mapping(response_mapping_text, {"answer": "answer"}),
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
                    if auto_eval_api:
                        with st.spinner("API 响应已保存，正在执行评估..."):
                            results, summary = evaluate_run_now(run, responses, selected_samples)
                        st.success(
                            f"外部 API 实验已保存，获得 {len(responses)} 条响应，并完成 {len(results)} 条评分。请到“评估与失败分析”查看详情。"
                        )
                        st.info(overall_judge_summary(summary, results))
                    else:
                        st.success(f"外部 API 实验已保存，获得 {len(responses)} 条响应。下一步到“评估与失败分析”执行评分。")
                    preview_df = response_review_dataframe(
                        responses,
                        selected_samples,
                        results if auto_eval_api else None,
                    )
                    st.subheader("本次 API 返回结果预览")
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    offer_dataframe_download(preview_df, f"{run.name}_rag_responses")


with tab_eval:
    st.header("评估引擎与失败分析")
    st.warning(
        "LLM 裁判可能产生较高费用。1000 条样本会消耗大量 tokens，因为裁判 prompt 往往包含问题、答案、参考答案和长上下文。建议先小批量验证。"
    )
    runs = store.list_experiments(project.project_id if project else None)
    if not runs:
        st.info("暂无实验，请先在“实验运行”中创建。")
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

        with st.expander("🗑️ 删除操作"):
            st.caption("以下操作不可撤销，请确认后再执行。")
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

        with st.expander("各指标说明"):
            _info_rows = [{"指标": label, "说明": desc} for _, (label, desc) in METRIC_USER_INFO.items()]
            st.dataframe(pd.DataFrame(_info_rows), use_container_width=True, hide_index=True)

        st.subheader("评估方式")
        eval_mode = st.radio(
            "选择评估引擎",
            options=["rule", "embedding", "ragas"],
            format_func=lambda x: {
                "rule": "⚡ 规则评分（快速，字符重叠，仅供初筛）",
                "embedding": "🧠 语义嵌入（本地运行，无需 API，推荐）",
                "ragas": "🎯 RAGAS 算法（最准确，每条样本约 8 次 LLM 调用）",
            }[x],
            key="eval_mode_radio",
            index=1,
        )

        eval_embedding_model = None
        _ragas_base = _ragas_key = _ragas_model = ""

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

        elif eval_mode == "rule":
            st.info(
                "规则评分基于字符/二元组重叠度，速度快但不理解语义，容易高估相似但措辞不同的答案。"
                "建议仅用于快速初筛，正式评估请选择语义嵌入或 RAGAS。"
            )

        if st.button("开始/重新评估", type="primary"):
            _ok = True
            if eval_mode == "ragas" and not (_ragas_base and _ragas_key):
                st.error("请填写 RAGAS 的 API Base 和 API Key。")
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
                )
                results = engine.evaluate_batch(limited, samples_by_id, update_progress)
                store.delete_eval_results(run.run_id)  # 先清旧结果，避免新旧混存
                store.save_eval_results(run.run_id, results)
                summary = summarize_run(run, limited, results)
                run.aggregate = summary
                run.config["score_version"] = config.score_version
                store.save_experiment(run)
                st.success(f"评估完成，保存 {len(results)} 条评分结果。")
                st.rerun()

        results = store.list_eval_results(run.run_id)
        if results:
            summary = summarize_run(run, responses, results)
            st.subheader("综合结论")
            st.info(overall_judge_summary(summary, results))

            metric_summary = summary.get("metric_summary", {})
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("平均综合得分", summary.get("avg_score", 0))
            col_s2.metric("成功率", summary.get("success_rate", 0))
            col_s3.metric("平均延迟(ms)", summary.get("avg_latency_ms", 0))
            col_s4.metric("估算成本", summary.get("estimated_cost", 0))

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
                metric_df = pd.DataFrame(metric_rows)
                st.subheader("指标概览")
                st.dataframe(
                    metric_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "分数": st.column_config.ProgressColumn("分数", min_value=0, max_value=1, format="%.3f"),
                    },
                )
                chart_df = pd.DataFrame(
                    [{"指标": r["指标"], "分数": r["分数"]} for r in metric_rows]
                ).set_index("指标")
                st.bar_chart(chart_df)

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
                        st.altair_chart(_pie1, use_container_width=True)
                    except Exception:
                        st.bar_chart(_fd_df.set_index("失败类型")[["出现次数"]])
                    st.caption("各失败类型占比：面积越大说明该类问题越集中，是优先改进方向。")
                    st.dataframe(_fd_df[["失败类型", "出现次数", "说明"]], use_container_width=True, hide_index=True)
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
                        st.altair_chart(_pie2, use_container_width=True)
                    except Exception:
                        st.bar_chart(pd.DataFrame({"低分": [_low], "中等": [_mid], "良好": [_high]}))
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
                    try:
                        import altair as alt
                        _bar_h = (
                            alt.Chart(_tdf)
                            .mark_bar()
                            .encode(
                                x=alt.X("平均得分:Q", scale=alt.Scale(domain=[0, 1]), title="平均综合得分"),
                                y=alt.Y("题型:N", sort="-x", title=""),
                                color=alt.condition(
                                    alt.datum["平均得分"] >= 0.65,
                                    alt.value("#22c55e"),
                                    alt.condition(
                                        alt.datum["平均得分"] >= 0.4,
                                        alt.value("#f59e0b"),
                                        alt.value("#ef4444"),
                                    ),
                                ),
                                tooltip=["题型:N", "平均得分:Q", "样本数:Q"],
                            )
                            .properties(height=max(200, len(_tdf) * 40))
                        )
                        st.altair_chart(_bar_h, use_container_width=True)
                    except Exception:
                        st.bar_chart(_tdf.set_index("题型")[["平均得分"]])
                    st.caption("红色=偏低，黄色=一般，绿色=良好。得分偏低的题型是系统薄弱点。")
                    st.dataframe(
                        _tdf,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "平均得分": st.column_config.ProgressColumn("平均得分", min_value=0, max_value=1, format="%.3f"),
                        },
                    )
                else:
                    st.info("测试用例需填写题型字段才能展示此图。")

            st.subheader("样本级评估明细")
            user_df = response_review_dataframe(responses, samples, results)
            score_filter = st.slider("只看综合得分低于", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
            display_df = user_df.copy()
            if "综合得分" in display_df.columns:
                display_df = display_df[pd.to_numeric(display_df["综合得分"], errors="coerce").fillna(0) <= score_filter]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            offer_dataframe_download(user_df, f"{run.name}_evaluation_details")

            with st.expander("查看内部评分字段"):
                st.dataframe(results_dataframe(results), use_container_width=True, hide_index=True)

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
            edited_labels = st.data_editor(label_df, use_container_width=True, hide_index=True)
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
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            offer_dataframe_download(preview_df, f"{run.name}_rag_responses")


with tab_compare:
    st.header("实验对比")
    runs = store.list_experiments(project.project_id if project else None)
    if len(runs) < 2:
        st.info("至少需要两个实验运行才能对比。")
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
            st.dataframe(compare_df, use_container_width=True)
            st.bar_chart(compare_df.set_index("实验名称")[["平均总分", "成功率"]])
            st.bar_chart(compare_df.set_index("实验名称")[["平均延迟(ms)", "估算成本"]])
            st.subheader("失败分布")
            for summary in summaries:
                st.write(f"**{summary['name']}**")
                st.json(summary["failure_distribution"])


with tab_export:
    st.header("导出中心")
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

        col_excel, col_md = st.columns(2)
        with col_excel:
            if st.button("一键导出 Excel", type="primary"):
                path = exporter.export_excel(project, run, samples, responses, results, summary)
                st.success(f"Excel 已导出：{path}")
                with Path(path).open("rb") as file:
                    st.download_button("下载 Excel", data=file, file_name=Path(path).name)
        with col_md:
            if st.button("一键导出 Markdown", type="primary"):
                path = exporter.export_markdown(project, run, samples, responses, results, summary, comparison_rows)
                st.success(f"Markdown 已导出：{path}")
                with Path(path).open("rb") as file:
                    st.download_button("下载 Markdown", data=file, file_name=Path(path).name, mime="text/markdown")
