from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.config import AppConfig
from app.models import EvalResult, EvalSample, ExperimentRun, ProjectContext, SystemResponse


class ExportCenter:
    def __init__(self, config: AppConfig):
        self.config = config

    def export_excel(
        self,
        context: ProjectContext,
        run: ExperimentRun,
        samples: list[EvalSample],
        responses: list[SystemResponse],
        results: list[EvalResult],
        summary: dict,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.export_dir / f"{run.name}_{timestamp}.xlsx"
        detail_df = self._detail_dataframe(samples, responses, results)
        metric_df = self._metric_summary_dataframe(results)
        failure_df = detail_df[detail_df["失败标签"].astype(str) != "[]"] if not detail_df.empty else detail_df
        qtype_df = self._question_type_distribution(samples)

        overview_df = pd.DataFrame(
            [
                {"项目": context.name, "值": run.name, "备注": f"样本数={summary.get('samples', 0)}"},
                {"项目": "平均综合得分", "值": summary.get("avg_score", 0), "备注": ""},
                {"项目": "成功率", "值": summary.get("success_rate", 0), "备注": ""},
                {"项目": "平均延迟(ms)", "值": summary.get("avg_latency_ms", 0), "备注": ""},
                {"项目": "估算成本", "值": summary.get("estimated_cost", 0), "备注": ""},
                {"项目": "LLM建议", "值": summary.get("llm_guidance", ""), "备注": ""},
                {
                    "项目": "指标叠加诊断",
                    "值": "；".join(summary.get("metric_combo_findings", [])[:8]),
                    "备注": "",
                },
            ]
        )

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            overview_df.to_excel(writer, sheet_name="experiment_overview", index=False)
            detail_df.to_excel(writer, sheet_name="sample_details", index=False)
            metric_df.to_excel(writer, sheet_name="metric_summary", index=False)
            failure_df.to_excel(writer, sheet_name="failure_cases", index=False)
            qtype_df.to_excel(writer, sheet_name="question_type_distribution", index=False)

            workbook = writer.book
            workbook.formats[0].set_font_name("Microsoft YaHei")
            header_format = workbook.add_format(
                {"bold": True, "bg_color": "#D9EAF7", "font_name": "Microsoft YaHei"}
            )
            for sheet in writer.sheets.values():
                sheet.set_row(0, None, header_format)
                sheet.freeze_panes(1, 0)
                sheet.set_column(0, 0, 22)
                sheet.set_column(1, 1, 42)
                sheet.set_column(2, 2, 60)
        return path

    def export_markdown(
        self,
        context: ProjectContext,
        run: ExperimentRun,
        samples: list[EvalSample],
        responses: list[SystemResponse],
        results: list[EvalResult],
        summary: dict,
        comparison_rows: list[dict] | None = None,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.export_dir / f"{run.name}_{timestamp}.md"
        detail_df = self._detail_dataframe(samples, responses, results)
        failures = detail_df[detail_df["失败标签"].astype(str) != "[]"].head(10) if not detail_df.empty else detail_df
        metric_df = self._metric_summary_dataframe(results)
        qtype_df = self._question_type_distribution(samples)

        lines: list[str] = []
        lines += [
            f"# {context.name} - 评测报告",
            f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> 实验：{run.name}　模式：{run.mode}",
            "",
            "## 项目摘要",
            "",
            "| 字段 | 内容 |",
            "|------|------|",
            f"| 项目 ID | {context.project_id} |",
            f"| 项目背景 | {_trunc(context.project_background, 300)} |",
            f"| 被评测系统 | {_trunc(context.system_description, 220)} |",
            f"| 评测目标 | {_trunc(context.evaluation_goals, 220)} |",
            f"| 业务规则 | {_trunc(context.business_rules, 220)} |",
            "",
            "## 实验概况",
            "",
            "| 指标 | 值 |",
            "|------|----|",
            f"| 样本数 | {summary.get('samples', 0)} |",
            f"| 成功率 | {summary.get('success_rate', 0)} |",
            f"| 平均综合得分 | {summary.get('avg_score', 0)} |",
            f"| 平均延迟(ms) | {summary.get('avg_latency_ms', 0)} |",
            f"| 估算成本 | {summary.get('estimated_cost', 0)} |",
            "",
            "## LLM建议与修改方向",
            "",
            summary.get("llm_guidance", "_本次未生成 LLM 建议（可能不是 LLM 评估模式）_"),
            "",
            "## 指标叠加诊断",
            "",
        ]

        combo_findings = summary.get("metric_combo_findings", [])
        if combo_findings:
            for item in combo_findings:
                lines.append(f"- {item}")
        else:
            lines.append("_本次未生成指标叠加诊断_")
        lines.append("")

        lines += ["## 总体指标", ""]
        if not metric_df.empty:
            lines.append("| 指标 | 平均分 |")
            lines.append("|------|--------|")
            for _, row in metric_df.iterrows():
                lines.append(f"| {row['指标']} | {row['平均分']} |")
        else:
            lines.append("_暂无评估结果_")
        lines.append("")

        lines += ["## 题型分布", ""]
        if not qtype_df.empty:
            lines.append("| 题型 | 数量 |")
            lines.append("|------|------|")
            for _, row in qtype_df.iterrows():
                lines.append(f"| {row['题型']} | {row['数量']} |")
        else:
            lines.append("_无题型数据_")
        lines.append("")

        lines += ["## 代表性失败案例（前10条）", ""]
        if not failures.empty:
            lines.append("| 问题 | 系统答案 | 总分 | 失败标签 | 裁判理由 |")
            lines.append("|------|----------|------|----------|----------|")
            for _, row in failures.iterrows():
                q = _trunc(str(row.get("问题", "")), 60)
                a = _trunc(str(row.get("系统答案", "")), 80)
                score = row.get("总分", "")
                labels = str(row.get("失败标签", ""))
                reason = _trunc(str(row.get("裁判理由", "")), 100)
                lines.append(f"| {q} | {a} | {score} | {labels} | {reason} |")
        else:
            lines.append("_无失败案例_")
        lines.append("")

        if comparison_rows:
            lines += ["## 实验对比", ""]
            lines.append("| 实验 | 样本数 | 成功率 | 平均总分 | 平均延迟(ms) | 估算成本 |")
            lines.append("|------|--------|--------|----------|-------------|----------|")
            for item in comparison_rows:
                lines.append(
                    f"| {item.get('name', '')} | {item.get('samples', '')} | {item.get('success_rate', '')}"
                    f" | {item.get('avg_score', '')} | {item.get('avg_latency_ms', '')} | {item.get('estimated_cost', '')} |"
                )
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _detail_dataframe(
        self,
        samples: list[EvalSample],
        responses: list[SystemResponse],
        results: list[EvalResult],
    ) -> pd.DataFrame:
        sample_map = {s.question_id: s for s in samples}
        result_map = {r.response_id: r for r in results}
        rows = []
        for response in responses:
            sample = sample_map.get(response.question_id)
            result = result_map.get(response.response_id)
            row = {
                "question_id": response.question_id,
                "问题": response.question,
                "题型": sample.question_type if sample else "",
                "难度": sample.difficulty if sample else "",
                "参考答案": response.reference_answer or (sample.reference_answer if sample else ""),
                "系统答案": response.answer,
                "成功": response.success,
                "错误": response.error,
                "延迟(ms)": response.latency_ms,
                "token_usage": response.token_usage,
                "总分": result.normalized_score if result else "",
                "失败标签": result.failure_labels if result else [],
                "裁判理由": result.judge_reason if result else "",
            }
            if result:
                for name, score in result.scores.items():
                    row[name] = score.normalized_score
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _metric_summary_dataframe(results: list[EvalResult]) -> pd.DataFrame:
        metric_names = sorted({name for result in results for name in result.scores})
        rows = []
        for name in metric_names:
            values = [result.scores[name].normalized_score for result in results if name in result.scores]
            rows.append({"指标": name, "平均分": round(sum(values) / len(values), 4) if values else 0})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("平均分", ascending=False)
        return df

    @staticmethod
    def _question_type_distribution(samples: list[EvalSample]) -> pd.DataFrame:
        if not samples:
            return pd.DataFrame(columns=["题型", "数量"])
        return (
            pd.DataFrame([{"题型": sample.question_type or "未分类"} for sample in samples])
            .value_counts("题型")
            .reset_index(name="数量")
        )


def _trunc(text: str, limit: int) -> str:
    text = text.replace("|", "¦").replace("\n", " ")
    return text[:limit] + "…" if len(text) > limit else text
