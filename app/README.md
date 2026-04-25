# RAG 评测工作台 MVP 中文 README

## 1. 产品定位

这是一个本地运行的中文评测工作台，用于评估外部 RAG / 检索生成 / 问答系统。它不是完整 RAG 平台，不内置向量库、BM25、混合检索或生产流量监控。

核心流程：

```text
项目设置 -> 测试问题集 -> 实验运行 -> 评估与失败分析 -> 实验对比 -> 导出中心
```

## 2. 当前设计逻辑

### 项目设置

用户只需要简单交代项目背景，不需要一开始拆成复杂 schema 配置：

- 项目名称
- 项目背景
- 被评测系统说明
- 评测目标
- 关键业务规则
- 期望题型
- 上传少量项目材料

支持上传小型材料：

- CSV
- Excel
- Markdown
- TXT
- JSON
- SQL

MVP 限制：

- 单文件最大 5MB
- CSV / Excel 默认只抽样前 100 行
- 文本、Markdown、SQL、JSON 默认截取前 20,000 字符

这些材料会被整理成发给 LLM 的项目 context。

### 测试问题集

测试问题集有两种来源：

1. 使用 LLM 生成
2. 上传已有测试问题集

如果选择 LLM 生成，用户需要在页面中提供：

- API Base
- API Key
- 模型名
- 最大生成问题数
- 超时
- 是否生成参考答案
- 是否生成期望证据

系统会把项目背景、业务规则、期望题型和上传材料抽样发送给 LLM，要求 LLM 输出标准 JSON 测试集。

如果选择上传已有测试问题集，则会跳过 LLM 生成，直接进入审核、编辑、去重和批准。

## 3. 技术栈

- UI：Streamlit
- 语言：Python
- 存储：SQLite
- 数据处理：Pandas
- Excel：XlsxWriter / OpenPyXL
- PDF：HTML 模板 + xhtml2pdf
- 并发：`concurrent.futures.ThreadPoolExecutor`
- 外部系统调用：`requests`

## 4. 安装与启动

```bash
pip install -r requirements.lock.txt
streamlit run app/main.py
```

如果需要新增或升级依赖，先修改 `requirements.txt`，再用 `uv pip compile requirements.txt -o requirements.lock.txt` 重新生成锁文件。

Windows 下可以执行：

```powershell
.\run_app.ps1
```

默认地址：

```text
http://localhost:8501
```

## 5. 数据模型

### ProjectContext

- `project_id`
- `name`
- `project_background`
- `system_description`
- `evaluation_goals`
- `business_rules`
- `question_type_instructions`
- `uploaded_assets`
- `schema_text`
- `metadata_json`
- `sample_rows`
- `few_shot_examples`

其中 `schema_text`、`metadata_json`、`sample_rows`、`few_shot_examples` 是高级补充字段，不是主流程必填。

### EvalSample

- `question_id`
- `question`
- `question_type`
- `difficulty`
- `expected_scope`
- `reference_answer`
- `expected_evidence`
- `tags`
- `source_context_refs`
- `generation_method`
- `review_status`

### SystemResponse

- `response_id`
- `question_id`
- `question`
- `reference_answer`
- `answer`
- `retrieved_contexts`
- `citations`
- `latency_ms`
- `token_usage`
- `raw_response`
- `success`
- `error`

### EvalResult

- `result_id`
- `question_id`
- `response_id`
- `scores`
- `normalized_score`
- `judge_reason`
- `judge_model`
- `score_version`
- `failure_labels`

### ExperimentRun

- `run_id`
- `project_id`
- `name`
- `mode`
- `config`
- `aggregate`
- `created_at`

## 6. 实验运行

支持两种模式：

### 外部 API 模式

用于调用黑盒 RAG / 问答 API。应用会自动读取“测试问题集”中审核通过的测试题，不需要用户再次上传问题集。应用只负责批量发问题、解析响应、保存结果，不实现检索。

可配置：

- endpoint
- HTTP 方法
- headers JSON
- 请求字段映射
- 响应字段映射
- retry / timeout / 并发

### 历史结果导入

支持 CSV / Excel / JSON。推荐字段：

```text
question_id, question, reference_answer, answer,
retrieved_contexts, citations, latency_ms, token_usage
```

没有 `question_id` 时，应用会尝试按问题文本匹配已通过测试集。

如果上传文件已经包含 RAG 回答字段 `answer`，可以勾选“导入后立即评估”。应用会创建实验、保存系统回答，并立刻生成评分结果；之后仍可在“评估与失败分析”中查看或重新评估。

## 7. 评分指标

回答侧：

- `correctness`
- `relevance`
- `faithfulness`
- `completeness`
- `hallucination_risk`

检索侧：

- `hit_rate`
- `context_relevance`
- `context_precision`
- `context_recall`
- `evidence_coverage`

系统侧：

- `latency_ms`
- `estimated_cost`
- `success_rate`

MVP 默认规则评分会基于问题、参考答案、系统答案、检索上下文和预期证据的重合度进行估计。启用 LLM 裁判后，会使用配置的 judge model 生成分数、理由和失败标签。

## 8. 失败标签

- `wrong_answer`
- `incomplete_answer`
- `unsupported_answer`
- `missing_evidence`
- `retrieval_issue`
- `should_abstain_but_answered`
- `ambiguous_question_failure`
- `cannot_judge`

## 9. 导出

Excel 包含：

- experiment_overview
- sample_details
- metric_summary
- failure_cases
- question_type_distribution

PDF 包含：

- 项目摘要
- 实验配置
- 总体结果
- 指标图示
- 代表性失败案例
- 可选对比结论

## 10. 成本警告

LLM 生成测试集和 LLM 裁判都可能消耗大量 tokens。评估 1000 条样本时，每条样本可能携带问题、系统答案、参考答案、检索上下文和引用，因此成本可能很高。

建议：

- 先用 20-50 条样本验证流程
- 确认 prompt、字段映射和评分稳定后再扩大批量
- MVP 默认单次评估上限为 300 条
- 在大批量运行前确认模型价格、超时、重试和并发设置

## 11. 故障排查

### LLM 生成失败

检查 API Base、API Key、模型名、代理网络、超时设置，以及返回内容是否为 JSON 数组。

### 外部 API 全部失败

检查 endpoint、headers、鉴权、请求字段映射和响应字段映射。

### PDF 导出失败

确认安装 `xhtml2pdf`。如果中文字体显示异常，可以调整 `app/templates/report.html`。
