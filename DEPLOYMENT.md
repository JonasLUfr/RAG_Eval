# 本RAG评测系统部署指南（DEMO 版）

> 本文档面向**内部 DEMO 部署**，暂无鉴权、无多租户隔离，任何能访问 URL 的人都能使用。
> 本版本内，暂时不要在公网开放，不要用于真实业务/敏感数据评测。(未经压力测试)

---

## 前置要求

- Linux 服务器（Ubuntu 20.04+ / Debian 11+ 推荐），至少 2 核 4GB 内存
- Docker 24.0+ 与 Docker Compose v2
- 如需嵌入模式：额外 2GB 磁盘用于多个模型缓存（首次自动下载约 470MB）

---

## 快速启动（5 步）

```bash
# 1. 克隆代码
git clone <仓库地址> rag-eval && cd rag-eval

# 2. 创建数据目录（用于持久化）
mkdir -p app/data app/exports

# 3. 配置环境变量（可选，也可通过 UI 输入）
cp .env.example .env  # 若提供了示例文件，否则手写
# 编辑 .env，填入 LLM_API_BASE / LLM_API_KEY / LLM_MODEL

# 4. 构建并启动
docker compose up -d --build

# 5. 访问 http://<服务器 IP>:8501
```

首次启动会自动写入示例数据（电商订单问答评测样本）。

---

## 环境变量

容器启动时从 `.env` 或 docker-compose 环境读取。所有变量都有默认值，未配置也能跑（仅规则模式可用）。

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_API_BASE` | （空） | LLM 服务地址，如 `https://api.openai.com/v1` |
| `LLM_API_KEY` | （空） | LLM API Key |
| `LLM_MODEL` | `gpt-4o-mini` | 测试集生成 / RAGAS 默认模型 |
| `JUDGE_MODEL` | 同 `LLM_MODEL` | LLM 裁判模式默认模型 |
| `RAG_EVAL_MAX_LLM_CALLS_PER_RUN` | `5000` | 单次评估硬上限（防止烧钱） |
| `RAG_EVAL_MAX_EVAL` | `300` | 单次评估最多样本数 |
| `RAG_EVAL_MAX_GENERATED` | `200` | 单次生成最多问题数 |
| `RAG_EVAL_MAX_WORKERS` | `5` | 并发线程数 |
| `RAG_EVAL_TIMEOUT` | `30` | LLM 请求超时（秒） |
| `RAG_EVAL_RETRY` | `2` | LLM 请求重试次数 |

---

## 反向代理（可选，但推荐）

直接暴露 8501 端口不安全。建议前置 nginx + HTTPS。

最小 nginx 配置示例：

```nginx
server {
    listen 443 ssl;
    server_name rag-eval.your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/.../fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/.../privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;  # 评估可能跑很久
    }
}
```

如需简单的 HTTP basic auth：在 location 块加 `auth_basic "Restricted"; auth_basic_user_file /etc/nginx/.htpasswd;`

---

## 数据持久化

通过 docker-compose 挂载到宿主机：

```
宿主机                      容器内
./app/data/      <-->     /app/app/data/      (SQLite 数据库 + 日志)
./app/exports/   <-->     /app/app/exports/   (导出的报告)
```

**备份**：定期备份 `./app/data/rag_eval.sqlite3` 即可。

```bash
# 简单备份示例
tar -czf backup-$(date +%F).tar.gz app/data/rag_eval.sqlite3
```

---

## 日志查看

```bash
# 容器内 streamlit 输出
docker compose logs -f rag-eval

# 应用结构化日志（评估 / LLM 调用 / 错误）
tail -f app/data/logs/app.log
```

---

## 故障排查（网站突然不可访问）

应用在每次启动 / 退出 / 抛异常时都会在 `app/data/logs/app.log` 写明显标记，看末尾几行即可判定死因。

### 1. 翻日志

```bash
# 应用日志末尾（最先看这个）
tail -200 /workspace/web/rag-eval/app/data/logs/app.log

# systemd 视角：进程退出码、被哪个信号杀的、重启次数
systemctl status <service-name>
journalctl -u <service-name> --since "10 minutes ago" --no-pager

# 内存压力（OOM 嫌疑时）
dmesg | grep -i "killed process\|out of memory" | tail -20

# 端口是否真的还在监听
ss -tlnp | grep 8501
```

### 2. 死因判定速查表

看 `app.log` **最后几行**：

| 末尾标记 | 推断 | 处理 |
| --- | --- | --- |
| `=== shutdown signal=SIGTERM ===` 或 `SIGINT` | systemd / 人为重启，正常退出 | 看启动 banner 确认是否已重启回来 |
| `=== UNCAUGHT EXCEPTION ===` + traceback | 代码抛了未捕获异常导致进程死 | 按 traceback 修 bug |
| 普通业务日志，**没有任何 shutdown / exception 标记** | 大概率被 SIGKILL（OOM 杀手 / kernel） | 查 `dmesg` 找内存证据；若确认 OOM，加内存或减小 `RAG_EVAL_MAX_WORKERS` |
| 完全为空 / 极旧 | 进程根本没起来 | `systemctl status` 看启动失败原因 |

每次进程启动会写一行 `=== STARTUP pid=X py=... ===`，多次重启之间的边界一目了然。

### 3. 还原现场速查

- 「之前能用，突然不能用」→ 先 `tail app.log` 看死因；再 `journalctl ... --since` 看 systemd 是否在重启循环
- 「重启后能用一阵又挂」→ 多半 OOM 或 文件句柄泄漏，`dmesg` + `ls /proc/<pid>/fd | wc -l` 验证
- 「页面能进但具体操作报错」→ 不是宕机，是单次脚本异常，红框 traceback 就在浏览器里，对照 `app.log` 即可

---

## 升级

```bash
git pull
docker compose up -d --build
```

数据库自动迁移，但**升级前请备份** `app/data/`。

---

## 常见问题

**Q: 嵌入模式首次启动很慢 / 卡住？**
A: 在下载 sentence-transformers 模型（约 470MB），等几分钟。下载后缓存在容器内，重启会丢——可以挂载 `~/.cache/huggingface` 到容器避免重复下载。

**Q: 评估跑到一半中断？**
A: 看 `app/data/logs/app.log`。常见原因：LLM 限流、API Key 失效、超时。

**Q: 想清空所有数据？**
A: `docker compose down && rm -rf app/data/* app/exports/* && docker compose up -d`

**Q: 端口被占用？**
A: 改 `docker-compose.yml` 里 `ports` 映射，比如 `"18501:8501"`。

---

## 卸载

```bash
docker compose down
docker rmi rag-eval:latest
# 如要彻底清理数据：
rm -rf app/data app/exports
```
