# 部署指南

> 面向**内部 DEMO / 团队共享**部署。默认无鉴权、无多租户隔离 —— 任何能访问 URL 的人都能使用。
> **公网部署前必须**前置反向代理 + 鉴权（见 [反向代理与 HTTPS](#反向代理与-https)）。

本文档覆盖：

- [前置要求](#前置要求)
- [快速启动（Docker）](#快速启动docker)
- [裸机部署（systemd）](#裸机部署systemd)
- [反向代理与 HTTPS](#反向代理与-https)
- [离线/内网环境：嵌入模型预置](#离线内网环境嵌入模型预置)
- [环境变量](#环境变量)
- [数据持久化与备份](#数据持久化与备份)
- [日志](#日志)
- [升级与回滚](#升级与回滚)
- [安全建议](#安全建议)
- [故障排查（网站突然不可访问）](#故障排查网站突然不可访问)
- [卸载](#卸载)

---

## 前置要求

| 项 | 要求 |
|----|------|
| 操作系统 | Linux（Ubuntu 20.04+ / Debian 11+ / CentOS 8+）。Windows 仅推荐本地开发，不推荐用作长期服务。 |
| CPU / 内存 | 至少 2 核 4GB；启用嵌入模式建议 8GB 以上。 |
| 磁盘 | 系统盘 5GB；嵌入模型缓存额外 1–2GB；运行历史按使用量增长。 |
| Python（裸机方式） | 3.11+ |
| Docker（容器方式） | Docker 24.0+ + Docker Compose v2 |
| 外网（可选） | 仅在 LLM 模式 / 嵌入模型首次下载时需要；可用国内镜像替代。 |

---

## 快速启动（Docker）

```bash
# 1. 克隆代码
git clone https://github.com/JonasLUfr/RAG_Eval.git rag-eval && cd rag-eval

# 2. 创建数据目录（首次启动会写入示例项目）
mkdir -p app/data app/exports

# 3. 配置环境变量（可选；未配置时仅规则评分可用，其他模式可在 UI 中临时填写）
cat > .env <<'EOF'
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
RAG_EVAL_MAX_LLM_CALLS_PER_RUN=5000
RAG_EVAL_MAX_EVAL=300
EOF

# 4. 构建并启动
docker compose up -d --build

# 5. 健康检查
curl -fsS http://localhost:8501/_stcore/health
# 浏览器访问 http://<server-ip>:8501
```

可选：**挂载 HuggingFace 缓存**避免容器重建后重新下载嵌入模型。编辑 `docker-compose.yml`：

```yaml
services:
  rag-eval:
    # ...
    volumes:
      - ./app/data:/app/app/data
      - ./app/exports:/app/app/exports
      - ./hf-cache:/root/.cache/huggingface     # ← 新增
    environment:
      # ...
      HF_HOME: /root/.cache/huggingface         # 显式指向挂载点
```

---

## 裸机部署（systemd）

适合不愿引入 Docker 或需要更直接日志观测的场景。

```bash
# 1. 准备用户和目录
sudo useradd -m -s /bin/bash rageval
sudo -u rageval -H bash <<'EOF'
cd ~
git clone https://github.com/JonasLUfr/RAG_Eval.git rag-eval
cd rag-eval
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
mkdir -p app/data app/exports
EOF
```

创建 `/etc/systemd/system/rag-eval.service`：

```ini
[Unit]
Description=RAG Eval Workbench (Streamlit)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rageval
Group=rageval
WorkingDirectory=/home/rageval/rag-eval
EnvironmentFile=/home/rageval/rag-eval/.env
ExecStart=/home/rageval/rag-eval/.venv/bin/streamlit run app/main.py \
          --server.address=127.0.0.1 \
          --server.port=8501 \
          --server.headless=true \
          --browser.gatherUsageStats=false
Restart=on-failure
RestartSec=5
# 安全加固
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=/home/rageval/rag-eval/app/data /home/rageval/rag-eval/app/exports

[Install]
WantedBy=multi-user.target
```

启动并设为开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rag-eval
sudo systemctl status rag-eval
```

> 注意：上面 `--server.address=127.0.0.1` 仅监听本地。**公网访问请通过 nginx 反代**，详见下一节。

---

## 反向代理与 HTTPS

直接暴露 8501 端口不安全（无鉴权 + 明文）。生产环境前置 nginx + basic auth + HTTPS 即可。最小示例：

```nginx
server {
    listen 443 ssl;
    server_name rag-eval.example.com;

    ssl_certificate     /etc/letsencrypt/live/rag-eval.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag-eval.example.com/privkey.pem;

    auth_basic           "RAG Eval";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;     # Streamlit websocket 必备
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;                    # 评估可能跑很久
    }
}
```

证书可用 `certbot --nginx -d rag-eval.example.com` 申请并自动续期；basic auth 用户用 `htpasswd -c /etc/nginx/.htpasswd <user>` 创建。

---

## 离线/内网环境：嵌入模型预置

嵌入模式（推荐用于免费 / 不烧 token / 内网评估）首次使用需要下载 sentence-transformers 模型（400 MB – 1 GB）。**应用 UI 会先检测本机缓存，未缓存时要求用户显式勾选「允许联网下载」**，不会偷偷转圈。

如果服务器无外网，按以下任一方式预置模型：

### 方式 A：有外网的机器上下载后整目录拷贝

```bash
# 在有外网的机器
pip install huggingface_hub
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 默认落地：~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
# 整目录 rsync / scp 到目标机的同位置即可
rsync -av ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 \
      rageval@<server>:/home/rageval/.cache/huggingface/hub/
```

容器部署时拷贝到挂载目录：

```bash
rsync -av <model-dir> <project-root>/hf-cache/hub/
```

### 方式 B：浏览器下载

打开 `https://huggingface.co/<repo>/tree/main`，下载全部文件并放到：

```
<HF cache>/hub/models--<org>--<name>/snapshots/<任意目录名>/
```

### 方式 C：使用国内镜像

设置环境变量后重启应用（systemd 写到 `.env`；Docker 写到 `docker-compose.yml`）：

```
HF_ENDPOINT=https://hf-mirror.com
```

### 内置可选模型

| 模型 | 大小 | 适用场景 |
|------|------|----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | ~420 MB | 多语言轻量，推荐首选 |
| `paraphrase-multilingual-mpnet-base-v2` | ~1 GB | 多语言高质量 |
| `shibing624/text2vec-base-chinese` | ~400 MB | 纯中文场景最佳 |

---

## 环境变量

容器启动时从 `.env` 或 `docker-compose.yml` 的 `environment` 读取；裸机部署从 systemd 的 `EnvironmentFile` 读取。所有变量都有默认值。

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_API_BASE` | （空） | LLM 服务地址，如 `https://api.openai.com/v1` |
| `LLM_API_KEY` | （空） | LLM API Key |
| `LLM_MODEL` | `gpt-4o-mini` | 测试集生成 / RAGAS 默认模型 |
| `JUDGE_MODEL` | 同 `LLM_MODEL` | LLM 裁判默认模型 |
| `RAG_EVAL_MAX_LLM_CALLS_PER_RUN` | `5000` | 单次评估硬上限（防烧钱） |
| `RAG_EVAL_MAX_EVAL` | `300` | 单次评估最多样本数 |
| `RAG_EVAL_MAX_GENERATED` | `200` | 单次生成最多问题数 |
| `RAG_EVAL_MAX_WORKERS` | `5` | LLM 并发数；嵌入模式强制 1 |
| `RAG_EVAL_TIMEOUT` | `90` | 单次 LLM 请求超时（秒） |
| `RAG_EVAL_RETRY` | `2` | 请求失败重试次数 |
| `RAG_EVAL_DB_PATH` | `app/data/rag_eval.sqlite3` | SQLite 路径 |
| `RAG_EVAL_EXPORT_DIR` | `app/exports` | 报告导出目录 |
| `RAG_EVAL_LOG_DIR` | `app/data/logs` | 日志目录 |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace 镜像 |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `~/.cache/huggingface` | 嵌入模型缓存位置 |

---

## 数据持久化与备份

### 持久化路径

```
宿主机                          容器内
./app/data/         <-->     /app/app/data/        SQLite + logs
./app/exports/      <-->     /app/app/exports/     导出的报告
./hf-cache/         <-->     /root/.cache/huggingface  （可选）嵌入模型缓存
```

### 备份

最小备份：每天打包一次 `app/data/`。

```bash
# 单次手动
tar -czf /backup/rag-eval-$(date +%F).tar.gz -C /path/to/rag-eval app/data

# 每日 cron
echo '15 3 * * * root tar -czf /backup/rag-eval-$(date +\%F).tar.gz -C /home/rageval/rag-eval app/data && find /backup -name "rag-eval-*.tar.gz" -mtime +30 -delete' | sudo tee /etc/cron.d/rag-eval-backup
```

### 恢复

```bash
sudo systemctl stop rag-eval        # 或 docker compose down
tar -xzf /backup/rag-eval-2025-12-01.tar.gz -C /home/rageval/rag-eval
sudo chown -R rageval:rageval /home/rageval/rag-eval/app/data
sudo systemctl start rag-eval       # 或 docker compose up -d
```

---

## 日志

```bash
# 应用结构化日志（含启动 / 异常 / 关闭标记 —— 看末尾几行即可判定死因）
tail -f app/data/logs/app.log

# Docker stdout
docker compose logs -f rag-eval

# systemd 视角（进程退出码、被哪个信号杀的、重启次数）
systemctl status rag-eval
journalctl -u rag-eval --since "10 minutes ago" --no-pager
```

### 死因判定速查表

看 `app.log` **最后几行**：

| 末尾标记 | 推断 | 处理 |
|----------|------|------|
| `=== STARTUP pid=X py=... ===` | 进程刚启动 | 正常 |
| `=== shutdown signal=SIGTERM/SIGINT ===` | 正常退出（systemd / 人为重启） | 看是否已重启回来 |
| `=== UNCAUGHT EXCEPTION ===` + traceback | 代码未捕获异常 | 按 traceback 修 |
| 无任何 shutdown / exception 标记 | 大概率被 SIGKILL（OOM / kernel） | 查 `dmesg`；加内存或调小 `RAG_EVAL_MAX_WORKERS` |
| 完全为空 / 极旧 | 进程根本没起来 | `systemctl status` 看启动失败原因 |

---

## 升级与回滚

### 升级

```bash
# 1. 备份
tar -czf /backup/rag-eval-pre-upgrade-$(date +%F).tar.gz -C /home/rageval/rag-eval app/data

# 2. 拉新代码
cd /home/rageval/rag-eval
git fetch && git log HEAD..origin/main --oneline   # 看新提交
git pull

# 3a. Docker 方式
docker compose up -d --build

# 3b. 裸机方式
.venv/bin/pip install -r requirements.lock.txt
sudo systemctl restart rag-eval

# 4. 健康检查
curl -fsS http://localhost:8501/_stcore/health
tail -20 app/data/logs/app.log     # 应该看到新的 STARTUP 行
```

### 回滚

```bash
git log --oneline -10                # 找回滚目标 commit
git checkout <commit-sha>
# 重复升级步骤 3-4
# 数据库回滚：替换 app/data/rag_eval.sqlite3 为对应日期的备份
```

> 数据库 schema 变更通常**向前兼容**（启动时自动迁移），但跨大版本回滚前请务必确认 commit 描述无 schema breaking change。

---

## 安全建议

| 项 | 建议 |
|----|------|
| 暴露面 | 不要直接暴露 8501，前置 nginx + auth |
| 鉴权 | 至少 basic auth；多人使用建议接 OAuth proxy（oauth2-proxy / authelia） |
| 网络 | 仅监听 127.0.0.1，由反代转发；如需 LAN 访问限制源 IP |
| 密钥 | LLM_API_KEY 写 `.env`（chmod 600），不要硬编码进镜像或 commit |
| 数据 | `app/data/` 含历史 LLM 输出和上传材料，按敏感级别确定备份加密策略 |
| 更新 | 订阅本仓库 release，及时升级；依赖通过 `requirements.lock.txt` 锁定 |
| 监控 | 配置 systemd `OnFailure=` 或 Prometheus `node_exporter` 监控进程存活 |

---

## 故障排查（网站突然不可访问）

应用在每次启动 / 退出 / 抛异常时都会在 `app/data/logs/app.log` 写显眼标记。**90% 的"突然挂掉"问题，看末尾几行即可定位**。

### 1. 翻日志

```bash
# 应用日志末尾（最先看这个）
tail -200 /home/rageval/rag-eval/app/data/logs/app.log

# systemd 视角
systemctl status rag-eval
journalctl -u rag-eval --since "10 minutes ago" --no-pager

# 内存压力（OOM 嫌疑时）
dmesg | grep -i "killed process\|out of memory" | tail -20

# 端口是否真的在监听
ss -tlnp | grep 8501
```

### 2. 按现象排查

| 现象 | 排查方向 |
|------|----------|
| 之前能用，突然不能用 | 先 `tail app.log` 看死因；再 `journalctl` 看是否在重启循环 |
| 重启后能用一阵又挂 | 多半 OOM 或 fd 泄漏；`dmesg` + `ls /proc/<pid>/fd \| wc -l` |
| 页面能进但操作报错 | 不是宕机，是单次脚本异常，浏览器红框 + `app.log` 对照 |
| 嵌入模式一直转圈 | HuggingFace 下载卡死；按 [离线嵌入模型预置](#离线内网环境嵌入模型预置) 处理 |
| LLM 全部超时 | 检查 `LLM_API_BASE` 可达性、`RAG_EVAL_TIMEOUT`、模型配额 |
| Excel/PDF 导出失败 | 看 `app.log`；PDF 中文乱码改 `app/templates/report.html` 字体 |
| 数据库 locked | SQLite 单进程写，确认没有别的进程同时打开；必要时重启 |

### 3. 紧急恢复

```bash
# 临时回滚到上一个 working commit
cd /home/rageval/rag-eval
git log --oneline -10
git checkout <last-good-commit>
sudo systemctl restart rag-eval

# 数据库损坏（极少见）
sqlite3 app/data/rag_eval.sqlite3 "PRAGMA integrity_check;"
# 损坏时用最近备份替换
```

---

## 卸载

```bash
# Docker
docker compose down
docker rmi rag-eval:latest

# systemd
sudo systemctl disable --now rag-eval
sudo rm /etc/systemd/system/rag-eval.service
sudo systemctl daemon-reload

# 彻底清理数据（不可恢复，请先备份）
sudo rm -rf /home/rageval/rag-eval
sudo userdel -r rageval
```
