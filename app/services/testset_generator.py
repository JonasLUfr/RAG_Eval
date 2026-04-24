from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

from app.core.config import AppConfig
from app.models import EvalSample, ProjectContext
from app.services.llm_client import OpenAICompatibleClient
from app.services.source_loader import build_materials_prompt


logger = logging.getLogger(__name__)


DEFAULT_TYPES = ["事实核对", "多跳推理", "边界条件", "业务规则", "不可回答"]


@dataclass
class TestsetLLMSettings:
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    temperature: float = 0.4
    proxy_url: str = ""


class TestsetGenerator:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate_with_llm(
        self,
        context: ProjectContext,
        settings: TestsetLLMSettings,
        max_questions: int | None = None,
        question_types: list[str] | None = None,
        include_reference_answer: bool = True,
        include_expected_evidence: bool = True,
    ) -> list[EvalSample]:
        limit = min(
            max_questions or self.config.default_max_generated_questions,
            self.config.default_max_generated_questions,
        )
        if not settings.api_base or not settings.api_key or not settings.model:
            raise ValueError("请填写 API Base、API Key 和模型名后再生成测试集。")
        types = question_types or self._parse_types(context.question_type_instructions)
        llm = OpenAICompatibleClient(
            self.config,
            model=settings.model,
            api_base=settings.api_base,
            api_key=settings.api_key,
            proxy_url=settings.proxy_url,
        )
        raw = llm.chat_json(
            self._system_prompt(),
            self._user_prompt(
                context=context,
                limit=limit,
                question_types=types,
                include_reference_answer=include_reference_answer,
                include_expected_evidence=include_expected_evidence,
            ),
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
        )
        if not isinstance(raw, list):
            raise ValueError("LLM 生成结果不是 JSON 数组。")

        samples = []
        for item in raw[:limit]:
            sample = EvalSample.from_dict(
                {
                    **item,
                    "generation_method": f"llm:{settings.model}",
                    "review_status": "待审核",
                }
            )
            samples.append(sample)
        return self.deduplicate(samples)

    def generate_demo(self, context: ProjectContext, max_questions: int = 5) -> list[EvalSample]:
        """仅用于无 API 时演示 UI 流程，正式测试集应由 LLM 生成或人工上传。"""
        types = self._parse_types(context.question_type_instructions)
        background = self._first_line(context.project_background or context.system_description, "当前项目")
        rules = self._first_line(context.business_rules, "项目业务规则")
        examples = [
            ("事实核对", f"根据项目背景，{background} 中最关键的事实查询是什么？"),
            ("业务规则", f"系统回答涉及“{rules}”时，应如何给出依据？"),
            ("边界条件", "当用户问题缺少必要条件时，系统应该追问、拒答还是默认处理？"),
            ("不可回答", "如果上传材料中没有相关证据，系统是否应该明确说明无法判断？"),
            ("多跳推理", "哪些问题需要结合多个字段、表格或文档片段才能回答？"),
        ]
        samples = []
        for idx in range(max_questions):
            qtype = types[idx % len(types)] if types else examples[idx % len(examples)][0]
            samples.append(
                EvalSample(
                    question=f"[演示样例{idx + 1}] {examples[idx % len(examples)][1]}",
                    question_type=qtype,
                    difficulty=["低", "中", "高"][idx % 3],
                    expected_scope="应基于项目背景和上传材料作答；材料不足时应明确说明。",
                    reference_answer="演示样例不代表真实参考答案，请人工修改或使用 LLM 重新生成。",
                    expected_evidence=rules,
                    tags=["demo", qtype],
                    source_context_refs=["project_background", "uploaded_assets"],
                    generation_method="local-demo",
                    review_status="待审核",
                )
            )
        return samples

    def deduplicate(self, samples: list[EvalSample]) -> list[EvalSample]:
        """按问题文本归一化去重，保留第一次出现的样本。"""
        bucket: OrderedDict[str, EvalSample] = OrderedDict()
        for sample in samples:
            key = "".join(sample.question.lower().split())
            if key and key not in bucket:
                bucket[key] = sample
        return list(bucket.values())

    def _parse_types(self, text: str) -> list[str]:
        lines = [x.strip(" -，,;；") for x in text.splitlines() if x.strip()]
        return lines or DEFAULT_TYPES

    def _system_prompt(self) -> str:
        return (
            "你是中文 RAG/问答系统评测专家。你的任务是基于用户提供的项目背景和小型材料，"
            "生成可执行的评测测试问题集。只输出 JSON 数组，不要输出 Markdown 或解释。"
        )

    def _user_prompt(
        self,
        context: ProjectContext,
        limit: int,
        question_types: list[str],
        include_reference_answer: bool,
        include_expected_evidence: bool,
    ) -> str:
        materials = build_materials_prompt(context.uploaded_assets)
        reference_instruction = (
            "尽量生成 reference_answer；如果材料不足以给出标准答案，请留空并说明 expected_scope。"
            if include_reference_answer
            else "reference_answer 可以留空。"
        )
        evidence_instruction = (
            "尽量生成 expected_evidence，指出证据字段、文档片段或规则来源。"
            if include_expected_evidence
            else "expected_evidence 可以留空。"
        )
        return f"""
请生成最多 {limit} 条评测测试用例。

项目名称：
{context.name}

项目背景：
{context.project_background}

被评测系统说明：
{context.system_description}

评测目标：
{context.evaluation_goals}

关键业务规则：
{context.business_rules}

期望题型：
{question_types}

用户上传材料抽样：
{materials}

高级补充材料：
Schema / 文档结构：{context.schema_text}
元数据：{context.metadata_json}
样例行：{context.sample_rows}
Few-shot 示例：{context.few_shot_examples}

生成要求：
1. 测试问题必须适合评估黑盒 RAG / 检索生成系统，不要假设本工具自带检索。
2. 问题要覆盖不同难度和不同题型，包含不可回答/证据不足场景。
3. 不要生成过于宽泛、无法客观判断的问题。
4. {reference_instruction}
5. {evidence_instruction}

每条 JSON 对象必须包含这些字段：
question, question_type, difficulty, expected_scope, reference_answer,
expected_evidence, tags, source_context_refs。

字段约束：
- difficulty 只能是 “低”、“中”、“高”。
- tags 是字符串数组。
- source_context_refs 是字符串数组，指向项目背景、上传文件名、业务规则或高级材料。
- 如果材料不足，请生成考察拒答能力的问题，而不是编造事实。
"""

    @staticmethod
    def _first_line(text: str, fallback: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()[:80]
        return fallback
