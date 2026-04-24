from __future__ import annotations

import logging
import os
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from xhtml2pdf import pisa

from app.core.config import AppConfig, ROOT_DIR
from app.models import EvalResult, EvalSample, ExperimentRun, ProjectContext, SystemResponse


logger = logging.getLogger(__name__)


def _find_cjk_font_path() -> str | None:
    """Return the absolute path of the first available CJK TTF font, or None."""
    if platform.system() == "Windows":
        fonts_dir = r"C:\Windows\Fonts"
        for fname in ("simhei.ttf", "simkai.ttf", "simfang.ttf", "simsunb.ttf"):
            p = os.path.join(fonts_dir, fname)
            if os.path.exists(p):
                return p
    return None


def _pisa_link_callback(uri: str, rel: str) -> str:
    """
    xhtml2pdf calls this to resolve every resource URL (fonts, images).
    We convert file:// URIs to absolute local paths so ReportLab can load the TTF.
    """
    if uri.startswith("file:///"):
        local = uri[8:].replace("/", os.sep)
        if os.path.exists(local):
            return local
    return uri


class ExportCenter:
    def __init__(self, config: AppConfig):
        self.config = config
        self.template_dir = ROOT_DIR / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

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
                {"项目": context.name, "实验": run.name, "样本数": summary.get("samples", 0)},
                {"项目": "平均总分", "实验": summary.get("avg_score", 0), "样本数": ""},
                {"项目": "成功率", "实验": summary.get("success_rate", 0), "样本数": ""},
                {"项目": "平均延迟(ms)", "实验": summary.get("avg_latency_ms", 0), "样本数": ""},
                {"项目": "估算成本", "实验": summary.get("estimated_cost", 0), "样本数": ""},
            ]
        )

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            overview_df.to_excel(writer, sheet_name="experiment_overview", index=False)
            detail_df.to_excel(writer, sheet_name="sample_details", index=False)
            metric_df.to_excel(writer, sheet_name="metric_summary", index=False)
            failure_df.to_excel(writer, sheet_name="failure_cases", index=False)
            qtype_df.to_excel(writer, sheet_name="question_type_distribution", index=False)

            workbook = writer.book
            # 设置默认字体支持中文（Microsoft YaHei 覆盖全部 CJK 字形）
            workbook.formats[0].set_font_name("Microsoft YaHei")
            header_format = workbook.add_format({
                "bold": True, "bg_color": "#D9EAF7", "font_name": "Microsoft YaHei",
            })
            for sheet in writer.sheets.values():
                sheet.set_row(0, None, header_format)
                sheet.freeze_panes(1, 0)
                sheet.set_column(0, 20, 22)
        return path

    def export_pdf(
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
        path = self.config.export_dir / f"{run.name}_{timestamp}.pdf"
        template = self.env.get_template("report.html")
        detail_df = self._detail_dataframe(samples, responses, results)
        failures = detail_df[detail_df["失败标签"].astype(str) != "[]"].head(10).to_dict("records") if not detail_df.empty else []
        # Build a file:// URI for the CJK font so xhtml2pdf can resolve it via link_callback
        _font_path = _find_cjk_font_path()
        cjk_font_uri = "file:///" + _font_path.replace(os.sep, "/") if _font_path else None
        html = template.render(
            context=context,
            run=run,
            samples=samples,
            summary=summary,
            metric_rows=self._metric_summary_dataframe(results).to_dict("records"),
            qtype_rows=self._question_type_distribution(samples).to_dict("records"),
            failures=failures,
            comparison_rows=comparison_rows or [],
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cjk_font_uri=cjk_font_uri,
        )
        with path.open("wb") as output:
            result = pisa.CreatePDF(
                html, dest=output, encoding="utf-8", link_callback=_pisa_link_callback
            )
        if result.err:
            logger.error("PDF 导出失败 err=%s", result.err)
            raise RuntimeError("PDF 导出失败，请检查 xhtml2pdf 依赖和 HTML 模板。")
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
        return pd.DataFrame(rows)

    @staticmethod
    def _question_type_distribution(samples: list[EvalSample]) -> pd.DataFrame:
        if not samples:
            return pd.DataFrame(columns=["题型", "数量"])
        return (
            pd.DataFrame([{"题型": s.question_type or "未分类"} for s in samples])
            .value_counts("题型")
            .reset_index(name="数量")
        )

