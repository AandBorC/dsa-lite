# 部署指南

三种跑法，按「省事程度」从高到低。

| 方式 | 适合 | 成本 | 需要一台常开的机器吗 |
|------|------|------|---------------------|
| GitHub Actions | 只想每天收到看板 | 公开库免费 | 不需要 |
| 本机定时任务 | 数据不出本机 | 免费 | 需要 |
| 手动跑 | 研究/调参 | 免费 | 不需要 |

---

## 零、先体检

不管哪种方式，先确认这台机器跑得动：

```bash
python main.py doctor
```

它会逐项戳数据源和 LLM，把「跑不起来」定位到具体是哪一环。
**这一步别跳过** —— 后面所有问题都会以更难看的形式重现。

---

## 一、GitHub Actions（推荐，零运维）

公开仓库的 Actions 额度不限量，这是最省事的方案。

### 1. Fork 或克隆到自己的仓库

```bash
gh repo fork AandBorC/dsa-lite --clone
cd dsa-lite
```

### 2. 配 Secrets

`Settings → Secrets and variables → Actions → New repository secret`

| Secret | 必需 | 说明 |
|--------|------|------|
| `LLM_BASE_URL` | 否 | 不配就走离线规则模拟器 |
| `LLM_API_KEY` | 否 | 云端 API 的 key |
| `LLM_MODEL` | 否 | 如 `deepseek-chat` |
| `TUSHARE_TOKEN` | 否 | 配了就启用 tushare 主源 |
| `WECOM_WEBHOOK` | 否 | 企业微信机器人 |
| `FEISHU_WEBHOOK` | 否 | 飞书机器人 |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | 否 | Telegram |

> ⚠️ **别把自选股写进 workflow 文件。** Secrets 不公开，但 `.yml` 是公开的。
> 自选股改 `config.yaml`，或者用 `workflow_dispatch` 的 `symbols` 输入临时覆盖。

### 3. 打开 Actions

默认已就绪，`.github/workflows/daily.yml` 会在**每个交易日 18:00（北京时间）**
自动跑 `analyze`，并把当日看板和信号台账提交回 `data/`。

台账（`data/ledger/signals.csv`）是回测的样本来源，**必须让 bot 提交回去**，
否则每天攒的信号就丢了，回测永远没有样本。

### 4. 手动触发一次验证

Actions 页面 → 选 `每日分析并推送决策看板` → `Run workflow`
→ 勾上 `mock`（离线跑，不花钱）先确认链路通。

---

## 二、本机定时任务

数据不出本机，适合不想把自选股放到云上的情况。

### Windows（任务计划程序）

```powershell
$action = New-ScheduledTaskAction -Execute "python" `
  -Argument "main.py analyze --out data\latest_dashboard.md" `
  -WorkingDirectory "C:\path\to\dsa-lite"
$trigger = New-ScheduledTaskTrigger -Daily -At 18:00
Register-ScheduledTask -TaskName "dsa-lite-analyze" `
  -Action $action -Trigger $trigger -Description "A股每日决策看板"
```

### Linux / macOS（cron）

```bash
# crontab -e  → 每个工作日 18:00
0 18 * * 1-5 cd /path/to/dsa-lite && /usr/bin/python3 main.py analyze --out data/latest_dashboard.md >> data/cron.log 2>&1
```

### systemd（更规范，带日志）

```ini
# /etc/systemd/system/dsa-lite.service
[Unit]
Description=dsa-lite 每日分析
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/dsa-lite
EnvironmentFile=/opt/dsa-lite/.env
ExecStart=/usr/bin/python3 main.py analyze --out data/latest_dashboard.md
```

```ini
# /etc/systemd/system/dsa-lite.timer
[Unit]
Description=每个交易日 18:00 触发

[Timer]
OnCalendar=Mon..Fri 18:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now dsa-lite.timer
systemctl list-timers | grep dsa
```

---

## 三、LLM 怎么配

三条路线，按成本排序：

### 路线 A：本地 Ollama（零成本，推荐）

```bash
# 1. 装 Ollama：https://ollama.com/download
# 2. 拉一个模型（实测 qwen2.5:7b 最适合这个任务）
ollama pull qwen2.5:7b

# 3. 配 .env
cat >> .env <<'EOF'
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
LLM_MAX_TOKENS=200
EOF

python main.py doctor --skip-limited    # 验证真的连通了
```

**`LLM_MAX_TOKENS` 一定要设。** 不限输出的话，小模型会被自己的啰嗦拖死
（实测 qwen3:8b 因为思考模式从 16s 拖到 92s）。

### 路线 B：云端 API（便宜、快）

```bash
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=deepseek-chat
```

> **HTTP 401 = Key 无效；HTTP 402 = Key 有效但余额为 0。**
> 这两个完全不是一回事，看到 402 别去重新申请 Key，去充值。

### 路线 C：不配

自动退回离线规则模拟器。所有链路都能跑通，只是判断来自规则而不是模型。
用来验证工程本身是否正常足够了。

---

## 四、数据源怎么配

降级链：`eastmoney → tencent → akshare → tushare → baostock → 本地缓存`

前两个**只用标准库**，开箱即用。所以**不配任何东西也能跑**。

### 启用 tushare

```bash
pip install tushare
echo "TUSHARE_TOKEN=你的token" >> .env
python main.py doctor     # 确认接口权限
```

token 获取：<https://tushare.pro/user/token>

**注意积分档位差异极大**，别以为有 token 就万事大吉：

| 接口 | 门槛 | 低积分账号实测 |
|------|------|----------------|
| `daily` | ≥120 | ✅ 自由调用 |
| `adj_factor` | ≥2000 | ⛔ 1 次/小时 |
| `daily_basic` | ≥2000 | ⛔ 1 次/小时 |
| `index_daily` | ≥2000 | ⛔ 1 次/小时 |

tushare 把「积分不够」和「频率超限」报成同一个错误码 `40203`，
所以单看报错很容易误判成自己代码写错了。

**化解办法不是硬扛限流，而是问「这数据多久变一次」：**
复权因子只在分红除权那天变（一年一两次），指数/换手率只追加不回改。
所以整段历史一次拉下来落盘缓存、按天/周刷新，日常跑日报根本不会再碰这些接口。

---

## 五、常见问题

**Q：`analyze` 报「数据源全部失败」**
A：跑 `doctor`。先看降级链里还有没有 ✅ 的源。东财有 IP 级限流，
连续请求会被 `RemoteDisconnected` 打回，腾讯通常能顶上。

**Q：回测结果和券商软件对不上**
A：检查有没有用复权价。`pro.daily()` 给的是**不复权**原始价，
直接回测会在除权日产生假跳空，且**方向可能相反**（实测茅台 2025-06-26：
不复权 -1.10%，真实 +0.83%）。本项目自实现前复权，`config.yaml` 里保持 `adjust: qfq`。

**Q：为什么回测数字每次跑都变**
A：前复权价格在每次分红除权后会整体重算，所以缓存必须带 TTL
（默认 1 天）。这不是省流量，是正确性问题。

**Q：LLM 回测太慢**
A：别用 LLM 做回测。回测用规则策略（毫秒级，随便调参），
实盘信号才用 LLM（每天几次调用，可以接受）。
prompt 哈希缓存能让重跑零成本，但首跑的时间省不掉。

**Q：`validate` 说「样本不足」**
A：交易笔数少于 20 笔时，任何统计结论都不成立。
扩展标的数量或时间区间，
`python main.py validate --symbols ... --days-back 1500`。
