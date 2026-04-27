# RAG 评测工作台 MVP

这是一个中文本地 Python 应用，用于评估检索增强生成系统或其他问答生成系统。它不是 RAG 平台，不包含向量库、BM25、混合检索或内部检索引擎；它只负责基于项目背景和小型上传材料生成/导入评测样本、运行或导入系统输出、评分、对比实验并导出报告。

详细说明见 [app/README.md](app/README.md)，用户操作指南见 [app/USER_GUIDE.md](app/USER_GUIDE.md)。

## 本地开发启动

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Windows 用户也可以直接运行：

```powershell
.\run_app.ps1
```

## 服务器部署（DEMO）

使用 Docker 部署内部 DEMO 站点，详见 [DEPLOYMENT.md](DEPLOYMENT.md)：

```bash
docker compose up -d --build
# 访问 http://<服务器 IP>:8501
```

⚠️ 部署用 `requirements.lock.txt`（精确锁定版本），如开发可继续用 `requirements.txt`。

## 其他文档

- [Eval_Design.md](Eval_Design.md) — 评估算法设计与实现
- [DEPLOYMENT.md](DEPLOYMENT.md) — 服务器部署指南
- [Delivery_Review.md](Delivery_Review.md) — DEMO 交付评估
- [app/USER_GUIDE.md](app/USER_GUIDE.md) — 用户操作指南
- [app/DEVELOPMENT_LOG.md](app/DEVELOPMENT_LOG.md) — 开发日志

首次启动会自动写入一个"电商订单问答评测示例"，并提供示例历史结果文件：`app/examples/demo_historical_results.csv`。

当前测试问题集支持两条路径：填写 LLM API 后生成，或上传已有测试问题集直接进入审核。
