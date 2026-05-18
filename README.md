# RAG 评测工作台 (RAG Eval Workbench)

> 面向中文场景的本地 RAG / 问答系统评测工具。生成测试集 → 跑实验 → 多模式打分 → 失败分析 → 多实验对比 → 一键导出报告。

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B">
  <img alt="Storage" src="https://img.shields.io/badge/storage-SQLite-003B57">
  <img alt="Docker" src="https://img.shields.io/badge/deploy-Docker-2496ED">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="Status" src="https://img.shields.io/badge/status-MVP-orange">
</p>

仓库地址：<https://github.com/JonasLUfr/RAG_Eval>

视频介绍：[【JonasLu】集成 RAG 效果测试平台 —— RAG_Eval (Bilibili)](https://www.bilibili.com/video/BV1aHLP6BE5o/?share_source=copy_web&vd_source=fe67270f946c226d2114d5ba750b2a4c)

---

## 目录

- [它解决什么问题](#它解决什么问题)
- [它不做什么](#它不做什么)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [评估模式对比](#评估模式对比)
- [离线 / 内网部署：嵌入模型准备](#离线--内网部署嵌入模型准备)
- [配置参考](#配置参考)
- [项目结构](#项目结构)
- [数据与持久化](#数据与持久化)
- [常见问题](#常见问题)
- [开发](#开发)
- [路线图与当前限制](#路线图与当前限制)
- [文档索引](#文档索引)
- [致谢与许可](#致谢与许可)

---

## 它解决什么问题

做 RAG / 知识库问答的人会遇到三个反复出现的工程问题：

1. **没有稳定测试集**：每次改 prompt / chunk / top-k，凭感觉测两条就上线，无法回归。
2. **打分维度散乱**：到底是答案错、检索没召回、还是 LLM 编了？没有一致的可对比指标。
3. **结果难以归档对外**：实验跑完散落在 Notebook / Excel / 聊天截图里，无法对客户、对内部 reviewer 形成正式报告。

本项目把这三件事压到一个 Streamlit 工作台里，按 **项目 → 测试集 → 实验 → 评估 → 对比 → 导出** 的线性流程走完闭环。

## 它不做什么

为避免错误预期，明确划清边界：

- **不是 RAG 平台**：不内置向量库、BM25、混合检索、reranker。被测系统由你自己提供（外部 API 或离线产出的回答 CSV）。
- **不是生产监控**：没有线上流量看板，专注离线评测和回归。
- **不做多租户 / 鉴权**：默认单机使用；公网部署需自行加反向代理 + 鉴权（见 [DEPLOYMENT.md](DEPLOYMENT.md)）。

## 核心特性

- **两种测试集来源**
  - LLM 生成：把项目背景、业务规则、上传材料抽样喂给 LLM，自动产出测试题、参考答案、期望证据。
  - 直接上传：已有人工标注的测试集可直接进入审核 / 编辑 / 去重流程。

- **两种实验运行方式**
  - 外部 API 模式：自定义 endpoint / headers / 请求字段映射 / 响应字段映射，批量打到被测 RAG 服务。
  - 历史结果导入：已有回答的 CSV / Excel / JSON 直接导入，可选「导入后立即评估」。

- **四种评估模式**（同一指标体系，不同打分实现）
  - 规则评分：字符 / 二元组重叠度，免费、秒级。
  - 嵌入相似度：本地 sentence-transformers，免费、无网调用、单线程稳定。
  - LLM 裁判：单次 LLM 调用产出每条样本的分数、理由、失败标签。
  - RAGAS：基于 RAGAS 框架的多维度自动评估（每条样本约 8 次 LLM 调用）。

- **统一指标体系**：回答侧（correctness / relevance / faithfulness / completeness / hallucination_risk）+ 检索侧（hit_rate / context_relevance / context_precision / context_recall / evidence_coverage）+ 系统侧（latency / cost / success_rate）。

- **失败分析**：自动标注 `wrong_answer` / `incomplete_answer` / `unsupported_answer` / `missing_evidence` / `retrieval_issue` / `cannot_judge` 等标签，按标签聚合定位问题。

- **多实验对比**：选两个或多个 run，按指标侧对侧比对，定位回归来源。

- **报告导出**
  - Excel：experiment_overview / sample_details / metric_summary / failure_cases / question_type_distribution
  - PDF：项目摘要 + 实验配置 + 总体结果 + 指标图示 + 代表性失败案例 + 对比结论

- **生产可用细节**
  - SQLite 持久化，所有运行历史可回溯。
  - 启动 / 退出 / 异常都在 `app/data/logs/app.log` 留显眼标记，挂掉时直接看末尾几行定位死因。
  - 单次评估硬上限（默认 5000 次 LLM 调用），防止配错参数烧钱。
  - 嵌入模式提供离线下载指引，内网友好。

## 快速开始

### 方式一：Docker（推荐用于团队共享 / 服务器部署）

```bash
git clone https://github.com/JonasLUfr/RAG_Eval.git rag-eval && cd rag-eval
mkdir -p app/data app/exports

# 可选：填入 LLM 配置，也可以启动后在 UI 里填
cat > .env <<'EOF'
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini
EOF

docker compose up -d --build
# 打开 http://localhost:8501
```

完整生产部署、反代、systemd、HTTPS、备份见 **[DEPLOYMENT.md](DEPLOYMENT.md)**。

### 方式二：本地 Python（开发 / 个人使用）

要求 Python 3.11+。

```bash
git clone https://github.com/JonasLUfr/RAG_Eval.git rag-eval && cd rag-eval

# 用 venv（也可以用 conda / uv）
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.lock.txt
streamlit run app/main.py
# 打开 http://localhost:8501
```

> 开发期新增依赖：改 `requirements.txt`，再用 `uv pip compile requirements.txt -o requirements.lock.txt` 重新生成锁文件。

### 方式三：Windows 一键启动

仓库根目录提供：

```powershell
.\run_app.ps1        # PowerShell
.\run_app.bat        # 双击运行
```

脚本会自动激活 `.venv` 并启动 Streamlit。

### 首次启动后

- 系统会自动写入一个「电商订单问答评测」示例项目。
- 示例历史结果：`app/examples/demo_historical_results.csv`，可在「实验运行 → 历史结果导入」中用它走通全流程。
- 测试集既可以直接用示例生成的，也可以上传 CSV/JSON 自带的题库。

## 评估模式对比

| 模式 | 是否需要 LLM | 是否需要外网 | 单条耗时 | 单条成本 | 适用场景 |
|------|--------------|--------------|----------|----------|----------|
| 规则评分 | 否 | 否 | 毫秒级 | 0 | 快速冒烟 / 排查流程是否打通 |
| 嵌入相似度 | 否 | 仅首次下载模型 | 几十毫秒 | 0 | 内网 / 不想烧钱 / 中等准确度 |
| LLM 裁判 | 是（1 次/条） | 是 | 数秒 | 中 | 正式评测，需可解释理由和失败标签 |
| RAGAS | 是（约 8 次/条） | 是 | 十秒级 | 高 | 学术级多维评估，需 RAGAS 生态指标 |

> 建议路径：**规则跑通流程 → 嵌入快速规模化 → LLM 裁判正式归档 → 必要时再上 RAGAS**。

## 离线 / 内网部署：嵌入模型准备

嵌入模式首次使用会从 HuggingFace 下载模型（400 MB – 1 GB）。在内网或外网受限环境下，UI 会**先检测本机缓存**：

- 已缓存 → 绿色提示，直接评估。
- 未缓存 → 给出三种手动下载方式 + 缓存目录路径 + 强制勾选「允许联网下载」才会真的发起下载，**默认不会偷偷转圈**。

三种手动方式：

```bash
# 方式 A：在有外网的机器下载，整目录拷贝到内网
pip install huggingface_hub
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# 拷贝到目标机：~/.cache/huggingface/hub/models--sentence-transformers--<model-name>

# 方式 B：浏览器下载 https://huggingface.co/<repo>/tree/main 全部文件
# 放到：<HF cache>/hub/models--<org>--<name>/snapshots/<任意目录名>/

# 方式 C：使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
# 重启应用后自动生效
```

缓存目录解析顺序：`HUGGINGFACE_HUB_CACHE` → `HF_HOME/hub` → `~/.cache/huggingface/hub`。UI 显示的路径会按当前进程的环境变量实时计算。

## 配置参考

所有参数都有默认值，未配置也能跑（规则模式可用）。环境变量在容器中通过 `.env` 或 `docker-compose.yml` 注入，在本地 Python 中通过 shell `export` 即可。

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_API_BASE` | （空） | LLM 服务地址，如 `https://api.openai.com/v1`。也可在 UI 上即时填写。 |
| `LLM_API_KEY` | （空） | LLM API Key。 |
| `LLM_MODEL` | `gpt-4o-mini` | 测试集生成 / RAGAS 默认模型。 |
| `JUDGE_MODEL` | 同 `LLM_MODEL` | LLM 裁判模式默认模型。 |
| `RAG_EVAL_MAX_LLM_CALLS_PER_RUN` | `5000` | 单次评估硬上限，防止配错参数烧钱。 |
| `RAG_EVAL_MAX_EVAL` | `300` | 单次评估最多样本数。 |
| `RAG_EVAL_MAX_GENERATED` | `200` | 单次生成最多问题数。 |
| `RAG_EVAL_MAX_WORKERS` | `5` | LLM 调用并发数（嵌入模式强制为 1）。 |
| `RAG_EVAL_TIMEOUT` | `90` | 单次 LLM 请求超时（秒）。 |
| `RAG_EVAL_RETRY` | `2` | LLM 请求失败重试次数。 |
| `RAG_EVAL_DB_PATH` | `app/data/rag_eval.sqlite3` | SQLite 文件路径。 |
| `RAG_EVAL_EXPORT_DIR` | `app/exports` | 报告导出目录。 |
| `RAG_EVAL_LOG_DIR` | `app/data/logs` | 日志目录。 |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace 镜像，内网可设为 `https://hf-mirror.com`。 |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `~/.cache/huggingface` | 嵌入模型缓存位置。 |

## 项目结构

```
rag-eval/
├── app/
│   ├── main.py                  # Streamlit 入口
│   ├── core/                    # 配置 / 全局常量
│   ├── models/                  # Pydantic schema（EvalSample / Response / Result ...）
│   ├── services/                # 业务逻辑
│   │   ├── generator.py         #   LLM 生成测试集
│   │   ├── importer.py          #   历史结果 / 测试集导入
│   │   ├── evaluator.py         #   评估调度（rule / embedding / llm_judge / ragas）
│   │   ├── embedding_evaluator.py
│   │   ├── ragas_evaluator.py
│   │   └── external_api.py      #   外部 RAG 系统调用
│   ├── storage/                 # SQLite 仓储层
│   ├── ui/                      # Streamlit 各 Tab 组件
│   ├── templates/               # PDF 报告模板（Jinja2 + HTML）
│   ├── examples/                # 示例项目和数据
│   ├── data/                    # 运行时数据库 / 日志（gitignored）
│   └── exports/                 # 导出报告（gitignored）
├── tests/                       # pytest 用例
├── scripts/                     # 运维脚本
├── design/                      # UI 草图 / 设计稿
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # 顶层依赖声明
├── requirements.lock.txt        # 精确锁定版本（部署用）
├── run_app.ps1 / run_app.bat    # Windows 启动脚本
├── README.md                    # ← 你正在看的文件
├── DEPLOYMENT.md                # 部署 + 故障排查
└── app/USER_GUIDE.md            # 用户操作指南
```

## 数据与持久化

| 路径 | 内容 | 容器内挂载位置 |
|------|------|----------------|
| `app/data/rag_eval.sqlite3` | 全部业务数据：项目、测试集、实验、结果 | `/app/app/data/` |
| `app/data/logs/app.log` | 应用日志（含启动 / 异常 / 关闭标记） | `/app/app/data/logs/` |
| `app/exports/` | 导出的 Excel / PDF 报告 | `/app/app/exports/` |
| `~/.cache/huggingface/` | 嵌入模型缓存（可选挂载） | `/root/.cache/huggingface/` |

**备份**：日常只需备份 `app/data/`：

```bash
tar -czf rag-eval-backup-$(date +%F).tar.gz app/data/
```

## 常见问题

**Q: 嵌入模式一直转圈不动？**
A: 在公司内网/外网受限环境，HuggingFace 下载会卡。新版本会先检测缓存，未缓存时不会偷偷下载，请按 UI 提示手动下载或设置 `HF_ENDPOINT` 镜像。

**Q: LLM 生成测试集报「无法解析为 JSON」？**
A: 多数是模型输出非纯 JSON。换更稳定的模型（gpt-4o-mini / claude-haiku），或减小 `RAG_EVAL_MAX_GENERATED`、降低生成温度。

**Q: 外部 API 模式全部失败？**
A: 检查顺序：endpoint 可达 → headers 鉴权 → 请求字段映射 → 响应字段映射。`app/data/logs/app.log` 里会有完整 traceback。

**Q: PDF 导出中文乱码？**
A: 确认安装了 `xhtml2pdf`，必要时在 `app/templates/report.html` 里调整字体声明。

**Q: 想清空所有运行数据？**
A:
```bash
docker compose down
rm -rf app/data/* app/exports/*
docker compose up -d
```

**Q: 端口被占？**
A: 改 `docker-compose.yml` 的 `ports` 映射，例如 `"18501:8501"`。

更多排查见 [DEPLOYMENT.md#故障排查](DEPLOYMENT.md#故障排查网站突然不可访问)。

## 开发

```bash
# 1. 装开发依赖
pip install -r requirements.lock.txt -r requirements-dev.txt

# 2. 跑测试
pytest

# 3. 改完依赖后重新生成锁文件
uv pip compile requirements.txt -o requirements.lock.txt

# 4. 本地热重载
streamlit run app/main.py
```

模块速览：

- 新增评估模式 → 在 `app/services/` 加 evaluator，按 `evaluator.py` 的 dispatch 模式注册。
- 新增导出格式 → 在 `app/services/exporter.py`（或对应模块）加生成函数，UI 在导出中心挂入口。
- 新增 UI Tab → 在 `app/ui/` 加组件，在 `app/main.py` 挂 Tab。

提 Issue / PR 前请：

- 跑通 `pytest`。
- 如果改了用户可见行为，更新 `app/USER_GUIDE.md`。
- 如果改了部署相关参数 / 流程，更新 `DEPLOYMENT.md` 和本 README 的「配置参考」表。

## 路线图与当前限制

**已知边界**

- 无鉴权 / 多租户隔离，公网部署必须前置反代 + auth。
- 上传材料单文件 5MB；CSV/Excel 默认抽样前 100 行；文本默认截前 20,000 字符。
- 评估默认硬上限 300 条 / run、5000 次 LLM 调用 / run，可调但请慎重。
- 未经压测，不建议直接面向真实业务流量。

**Roadmap（按优先级）**

- [ ] 评测任务异步化 + 后台队列（当前长任务依赖浏览器保持）
- [ ] 多用户工作区 + 简单 RBAC
- [ ] 评估结果 diff 视图（按 question_id 维度的逐条变化）
- [ ] 更多内置 judge prompt 模板
- [ ] 一键导出最小可复现实验包

## 文档索引

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 项目总览（本文件） |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 服务器部署 + 反代 + systemd + HTTPS + 故障排查 |
| [app/README.md](app/README.md) | 产品定位 + 数据模型 + 流程细节 |
| [app/USER_GUIDE.md](app/USER_GUIDE.md) | 面向终端用户的操作指南 |

## 致谢与许可

- UI 框架：[Streamlit](https://streamlit.io/)
- 嵌入模型：[sentence-transformers](https://www.sbert.net/) 社区
- 自动评估：[RAGAS](https://docs.ragas.io/)
- PDF 渲染：[xhtml2pdf](https://github.com/xhtml2pdf/xhtml2pdf)

License: [Apache License 2.0](LICENSE)。
