#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dsa-lite —— 带策略验证的 LLM 股票分析系统
============================================================
工程结构参考 daily_stock_analysis（多源降级 + 定时任务 + 多通道推送），
核心增量是它没有的东西：**可证伪的回测与策略验证层**。

六个命令说白了就是一条完整闭环：

    doctor    先体检：环境、数据源、LLM 到底通不通
    analyze   每天跑一次，产出决策看板并推送（同时把信号写进台账）
    backtest  用历史数据验证策略，回答"这套逻辑到底赚不赚钱"
    validate  进阶验证：五大指标打分 + 样本内外一致性 + 显著性检验
    compare   多策略横向对比，看 LLM 相对朴素规则到底有没有增量
    ledger    查看信号台账

用法：
    python main.py doctor                       # 先跑这个，30 秒定位问题
    python main.py analyze                      # 每日分析（有 LLM_API_KEY 就用真模型）
    python main.py analyze --mock               # 离线模式，纯规则模拟
    python main.py backtest --strategy five_dim --symbols sh600519,sz000858
    python main.py validate --strategy five_dim --split 0.7
    python main.py compare --symbols sh600519,sz000858,sh601318
    python main.py ledger                       # 查看信号台账统计

依赖：零强制依赖（纯标准库）。可选装 akshare/tushare 增加数据源。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import asof                                              # noqa: E402
from core.backtest import BacktestConfig, BacktestEngine          # noqa: E402
from core.fetchers import (CACHE_DIR, Bar, CsvCache, FetcherChain,  # noqa: E402
                           NoDataSourceError, _apply_adjust, _importable,
                           get_benchmark, to_ts_code)
from core.llm import (MockAnalyzer, PROMPT_VERSION, _is_local_url,  # noqa: E402
                      build_analyzer)
from core.notify import Notifier                                   # noqa: E402
from core.report import (render_backtest_summary, render_dashboard,  # noqa: E402
                         render_trades)
from core.signals import SignalCache, SignalLedger                 # noqa: E402
from core.strategies import (LedgerStrategy, LLMStrategy, build_strategy,  # noqa: E402
                             FiveDimStrategy, MACrossStrategy, RSIReversionStrategy)
from core.validator import StrategyValidator, render_validation     # noqa: E402

log = logging.getLogger("main")


def _rel(p) -> str:
    """
    把路径转成相对项目根目录的写法。

    报告是要给别人看的（也会提交进仓库），打印绝对路径等于把
    用户名和目录结构一起交出去。项目内的路径一律相对化，
    项目外的退回原样（那种情况本来就说明配置有问题）。
    """
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return str(p)


DEFAULT_CONFIG = {
    "watchlist": ["sh600519", "sz000858", "sh601318", "sz300750", "sh600036"],
    "benchmark": "sh000300",
    "start_date": "",              # 空 = 自动取一年前
    "adjust": "qfq",
    "fetcher_priority": ["eastmoney", "tencent", "akshare", "tushare", "baostock"],
    "strategy": "five_dim",        # 回测默认策略
    "analyzer": "auto",            # auto | llm | mock
    "backtest": {
        "initial_cash": 100000,
        "per_trade_pct": 0.30,
        "max_positions": 5,
        "commission_rate": 0.00025,
        "stamp_tax_rate": 0.0005,
        "slippage_pct": 0.001,
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.25,
        "trailing_stop_pct": 0.10,
        "max_holding_days": 30,
        "cooldown_after_losses": 3,
        "cooldown_days": 5,
        "risk_free_rate": 0.02,
    },
    "strategy_params": {
        "ma_cross": {"short": 5, "long": 20, "stop_atr": 2.0},
        "five_dim": {"buy_threshold": 66, "sell_threshold": 48, "stop_atr": 2.5},
        "rsi_reversion": {"oversold": 28, "overbought": 72, "trend_filter": True},
    },
    "notify": {
        "console": True,
        "log_file": "data/reports.md",
        "wecom_webhook": "", "feishu_webhook": "",
        "telegram_bot_token": "", "telegram_chat_id": "",
        "discord_webhook": "", "slack_webhook": "",
        "email": {},
    },
    "validate": {"split": 0.7, "min_trades": 20},
}


# ============================================================
# 配置
# ============================================================

def _load_dotenv(path: Path | None = None) -> None:
    """
    极简 .env 加载器 —— 刻意不引入 python-dotenv，保持"零强制依赖"这条线。

    规则：**已存在的环境变量优先**。
    这样临时覆盖（`LLM_MODEL=x python main.py ...`）永远赢过 .env 里的默认值，
    符合一般人对命令行的直觉。
    """
    p = path or (ROOT / ".env")
    if not p.exists():
        return
    loaded = 0
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                loaded += 1
    except OSError as exc:  # noqa: BLE001
        log.warning(".env 读取失败: %s", exc)
        return
    if loaded:
        log.info("已从 %s 载入 %d 项环境变量", p.name, loaded)


def load_config(path: str = "config.yaml", overlay: dict | None = None) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else
               (list(v) if isinstance(v, list) else v)) for k, v in DEFAULT_CONFIG.items()}
    p = ROOT / path
    if p.exists():
        try:
            import yaml
            user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            _deep_merge(cfg, user)
            log.info("已加载配置 %s", _rel(p))
        except ImportError:
            log.warning("未安装 pyyaml，忽略 config.yaml（pip install pyyaml 可启用）")
        except Exception as exc:  # noqa: BLE001
            log.warning("配置文件解析失败，使用默认值: %s", exc)
    if overlay:
        _deep_merge(cfg, {k: v for k, v in overlay.items() if v is not None})
    return cfg


def _deep_merge(base: dict, over: dict) -> None:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def make_chain(cfg: dict) -> FetcherChain:
    return FetcherChain(priority=cfg["fetcher_priority"])


def date_range(cfg: dict, days_back: int = 400) -> tuple[str, str]:
    start = cfg.get("start_date") or (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    return start, end


def make_backtest_cfg(cfg: dict) -> BacktestConfig:
    bt = cfg.get("backtest", {})
    allowed = {f for f in BacktestConfig.__dataclass_fields__}
    return BacktestConfig(**{k: v for k, v in bt.items() if k in allowed})


def make_strategy(kind: str, cfg: dict, analyzer=None, ledger_signals=None):
    """构造策略。ledger 类型需要外部提供信号列表。"""
    if kind == "llm":
        if analyzer is None:
            raise RuntimeError("llm 策略需要 analyzer")
        return LLMStrategy(analyzer, cache=SignalCache(), prompt_version=PROMPT_VERSION)
    if kind == "ledger":
        if not ledger_signals:
            raise RuntimeError("台账为空，先跑 analyze 积累信号")
        return LedgerStrategy(ledger_signals)
    return build_strategy(kind, **cfg.get("strategy_params", {}).get(kind, {}))


# ============================================================
# analyze —— 每日分析
# ============================================================

def cmd_analyze(args, cfg: dict) -> int:
    symbols = args.symbols.split(",") if args.symbols else cfg["watchlist"]
    symbols = [s.strip() for s in symbols if s.strip()]
    start, end = date_range(cfg, days_back=args.days_back)

    chain = make_chain(cfg)
    log.info("抓取 %d 只标的的行情（%s ~ %s）", len(symbols), start, end)
    bars_by_symbol, sources = chain.fetch_many(
        symbols, start, end, adjust=cfg["adjust"],
        force_refresh=args.refresh)

    errors = [f"{s}" for s in symbols if s not in bars_by_symbol]
    if not bars_by_symbol:
        log.error("没有任何标的数据可用，退出")
        return 2

    analyzer = build_analyzer("mock" if args.mock else cfg["analyzer"])
    ledger = SignalLedger()
    signals = []

    # 持仓上下文：从台账里最近一次 BUY 推断（简单版）
    for sym, bars in bars_by_symbol.items():
        if len(bars) < 65:
            log.warning("[%s] 仅 %d 根K线，不足 65 根，跳过", sym, len(bars))
            continue
        i = len(bars) - 1
        try:
            sig = analyzer.analyze(_snap(sym, bars, i), holding=False)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] 分析失败: %s", sym, exc)
            errors.append(f"{sym}(分析异常)")
            continue
        sig.meta.setdefault("snapshot", _snap(sym, bars, i).to_prompt_dict())
        signals.append(sig)

    if not signals:
        log.error("未产出任何信号")
        return 2

    added = ledger.append(signals)

    body = render_dashboard(
        signals, data_sources=sources, errors=errors,
        title_date=bars_by_symbol[list(bars_by_symbol)[0]][-1].date)

    # 控制台既可能由 print 输出，也可能由 Notifier 的 console 通道输出 ——
    # 两者同时开会把看板打两遍，所以这里二选一。
    if args.no_notify:
        print(body)
    else:
        title = f"决策看板 {datetime.now().strftime('%Y-%m-%d')}"
        notifier = Notifier(cfg.get("notify", {}))
        notifier.send(title, body)
        log.info(notifier.summary())

    log.info("台账新增 %d 条信号（共 %d 条）", added, ledger.stats().get("total", 0))
    if isinstance(analyzer, object) and hasattr(analyzer, "cost_report"):
        log.info("LLM 用量: %s", analyzer.cost_report())

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        log.info("看板已写出 → %s", out)
    return 0


def _snap(sym, bars, i):
    from core.indicators import build_snapshot
    return build_snapshot(sym, bars, i)


# ============================================================
# backtest —— 基础回测
# ============================================================

def _load_bars(cfg: dict, symbols: list[str], args) -> tuple[dict, list]:
    chain = make_chain(cfg)
    start, end = date_range(cfg, days_back=getattr(args, "days_back", 700))
    as_of = _as_of(args)
    bars, sources = chain.fetch_many(symbols, start, end, adjust=cfg["adjust"],
                                     force_refresh=getattr(args, "refresh", False),
                                     as_of=as_of)
    bench = get_benchmark(start, end, cfg["benchmark"], chain, as_of=as_of)
    log.info("拿到 %d 只标的（来源 %s），基准 %d 根%s",
             len(bars), set(sources.values()), len(bench),
             f"，时点门控 as_of={as_of}" if as_of else "")
    if chain.audit is not None and chain.audit.dirty:
        for line in chain.audit.summary():
            log.warning("时点审计 | %s", line)
    return bars, bench


def _as_of(args) -> str | None:
    """从命令行取 as_of，顺手校验格式（写错了要立刻报，不能当成 None 静默放行）。"""
    raw = getattr(args, "as_of", None)
    if not raw:
        return None
    try:
        return asof.norm(raw)
    except ValueError as exc:
        raise SystemExit(f"--as-of 格式无法识别：{raw}（应形如 2025-06-20）") from exc


def cmd_backtest(args, cfg: dict) -> int:
    symbols = [s.strip() for s in (args.symbols or ",".join(cfg["watchlist"])).split(",") if s.strip()]
    bars, bench = _load_bars(cfg, symbols, args)
    if not bars:
        log.error("无数据，退出")
        return 2

    st = make_strategy(args.strategy, cfg)
    engine = BacktestEngine(st, make_backtest_cfg(cfg))
    result = engine.run(bars, bench, as_of=_as_of(args))

    print(render_backtest_summary(result))
    if args.show_trades:
        print()
        print(render_trades(result, limit=args.trades_limit))

    if args.out:
        md = [f"# 回测报告 · {st.name}", "", "```", render_backtest_summary(result), "```", ""]
        if result.trades:
            md += ["## 交易明细", "", render_trades(result, limit=200), ""]
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md), encoding="utf-8")
        log.info("报告已写出 → %s", _rel(out))
    return 0


# ============================================================
# validate —— 策略体检（本项目的核心）
# ============================================================

def cmd_validate(args, cfg: dict) -> int:
    symbols = [s.strip() for s in (args.symbols or ",".join(cfg["watchlist"])).split(",") if s.strip()]
    bars, bench = _load_bars(cfg, symbols, args)
    if not bars:
        log.error("无数据，退出")
        return 2

    if args.strategy == "llm":
        analyzer = build_analyzer("mock" if args.mock else cfg["analyzer"])
        st = make_strategy("llm", cfg, analyzer=analyzer)
    elif args.strategy == "ledger":
        st = make_strategy("ledger", cfg, ledger_signals=SignalLedger().read())
    else:
        st = make_strategy(args.strategy, cfg)

    vcfg = cfg.get("validate", {})
    validator = StrategyValidator(
        config=make_backtest_cfg(cfg),
        is_ratio=args.split or vcfg.get("split", 0.7),
        min_trades_for_verdict=args.min_trades or vcfg.get("min_trades", 20))

    log.info("开始体检：策略=%s 标的=%d 只 切分=%.0f%%", st.name, len(bars), validator.is_ratio * 100)
    result, report = validator.validate(st, bars, bench, as_of=_as_of(args))

    md = render_validation(report, result)
    print(render_backtest_summary(result))
    print()
    print(md)

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        log.info("体检报告已写出 → %s", _rel(out))
    return 0 if report.grade != "🔴 需检修" else 1


# ============================================================
# compare —— 横向对比
# ============================================================

def cmd_compare(args, cfg: dict) -> int:
    symbols = [s.strip() for s in (args.symbols or ",".join(cfg["watchlist"])).split(",") if s.strip()]
    bars, bench = _load_bars(cfg, symbols, args)
    if not bars:
        log.error("无数据，退出")
        return 2

    strat_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = []
    for n in strat_names:
        if n == "llm":
            strategies.append(make_strategy("llm", cfg,
                                            analyzer=build_analyzer("mock" if args.mock else
                                                                    cfg["analyzer"])))
        else:
            try:
                strategies.append(make_strategy(n, cfg))
            except Exception as exc:  # noqa: BLE001
                log.warning("跳过策略 %s: %s", n, exc)

    validator = StrategyValidator(make_backtest_cfg(cfg), is_ratio=args.split or 0.7)
    rows = validator.compare(strategies, bars, bench, as_of=_as_of(args))

    header = ("| 排名 | 策略 | 累计% | 年化% | 超额% | 笔数 | 胜率% | 盈亏比 | 回撤% | "
              "夏普 | 期望% | 显著 | 评级 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for i, (name, r, rep) in enumerate(rows, 1):
        m = r.metrics
        print(f"| {i} | {name} | {m.total_return_pct:+.2f} | {m.annual_return_pct:+.2f} | "
              f"{m.excess_return_pct:+.2f} | {m.trades} | {m.win_rate_pct:.1f} | "
              f"{m.profit_loss_ratio:.2f} | {m.max_drawdown_pct:.2f} | {m.sharpe:.2f} | "
              f"{m.expectancy_pct:+.3f} | {'✅' if m.is_significant else '—'} | {rep.grade} |")

    print()
    print("## 结论要点")
    for name, r, rep in rows:
        print(f"\n**{name}**　{rep.grade}")
        print(f"- {rep.overfit_verdict}")
        print(f"- {rep.significance_note}")
        if rep.benchmark_note:
            print(f"- 基准：{rep.benchmark_note}")
    return 0


# ============================================================
# doctor —— 先确认「跑得起来」，再谈「准不准」
# ============================================================

_PROBE_SYMBOL = "sh600519"      # 体检标的：流动性好、除权历史清楚
_PROBE_DAYS = 45                # 数据源探测窗口：短窗口就够证明"通不通"，还快
_LLM_PROBE_DAYS = 200           # LLM 探测窗口：指标要 60 日均线，短窗口凑不出快照

# tushare 各接口的官方积分门槛。用途不是炫知识，而是把「报什么错」
# 翻译成「为什么」——低积分账号最容易把「限流/积分不够」
# 误判成「我代码写错了」，然后去改一堆没问题的代码。
_TS_REQUIREMENTS = {
    "daily":       ("120", "历史日线（主源，行情全靠它）"),
    "adj_factor":  ("2000", "复权因子（前复权必需）"),
    "daily_basic": ("2000", "每日指标（换手率/估值）"),
    "index_daily": ("2000", "指数日线（基准对比）"),
}


def _probe(fn, *a, **kw) -> tuple[bool, object, float, str]:
    """跑一次探测，返回 (成功?, 结果, 耗时秒, 错误文本)。"""
    t0 = time.perf_counter()
    try:
        return True, fn(*a, **kw), time.perf_counter() - t0, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def _tushare_pro():
    """构造 tushare pro 客户端。token 缺失或缺包时抛错，由调用方接住。"""
    token = (os.environ.get("TUSHARE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("未配置 TUSHARE_TOKEN")
    import tushare as ts
    ts.set_token(token)
    return ts.pro_api()


def _skip_reason(name: str) -> str:
    """「跳过」必须说清楚是没装还是没配 —— 两者的修法完全不同。"""
    if name == "tushare":
        if not (os.environ.get("TUSHARE_TOKEN") or "").strip():
            return "未配置 TUSHARE_TOKEN（见 .env.example）"
        return "未安装 tushare（pip install tushare）"
    return f"未安装 {name}（pip install {name}）"


def _classify_ts_error(msg: str) -> str:
    """
    把 tushare 的原始报错翻译成人话。

    tushare 把「积分不够」也报成「频率超限」，报错原文长得一样，
    所以这里顺带把限流频次抠出来 —— 频次是一小时一次还是每分钟几百次，
    决定了「要不要上缓存」还是「纯属配置问题」。
    """
    m = str(msg)
    if "频率超限" in m or "40203" in m:
        mt = re.search(r"\((\d+)次/([^\)]+)\)", m)
        rate = f"（{mt.group(1)} 次/{mt.group(2)}）" if mt else ""
        return f"限流{rate}"
    if "权限" in m or "积分" in m:
        return "积分不足"
    if "token" in m.lower():
        return "token 无效"
    return "失败"


def cmd_doctor(args, cfg: dict) -> int:
    """
    体检：把「能不能跑」和「跑得准不准」分开确认。

    存在的理由很实在 —— 数据源报错时，人最容易做错的两件事
    是「反复重试」和「凭印象猜」。
    doctor 把每个源、每个接口单独戳一下，直接区分：
    没装 / 没配 token / 积分不够 / 被限流 / 网络不通。
    """
    t_start = time.perf_counter()
    sym = args.symbol or _PROBE_SYMBOL
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=_PROBE_DAYS)).strftime("%Y%m%d")
    order = list(cfg["fetcher_priority"])
    warns: list[str] = []
    lines: list[str] = []

    lines += ["## dsa-lite 体检报告", "",
              f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
              f"- 探测标的：`{sym}`（{start} ~ {end}）",
              f"- 降级链：{' → '.join(order)}", ""]

    # ---------- 1) 环境 ----------
    lines += ["### 1. 环境", "", "| 项目 | 状态 |", "|------|------|",
              f"| Python | {sys.version.split()[0]} |"]
    for mod, why in (("yaml", "读取 config.yaml"),
                     ("tushare", "tushare 数据源"),
                     ("akshare", "akshare 数据源"),
                     ("baostock", "baostock 数据源")):
        lines.append(f"| `{mod}` | {'✅ 已安装' if _importable(mod) else '— 未安装'}"
                     f"（{why}） |")
    tok = (os.environ.get("TUSHARE_TOKEN") or "").strip()
    lines.append(f"| `TUSHARE_TOKEN` | "
                 f"{f'✅ 已配置（{len(tok)} 位）' if tok else '— 未配置'} |")
    cfgp = ROOT / args.config
    lines.append(f"| 配置 | `{args.config}`"
                 f"{'' if cfgp.exists() else '（不存在，用内置默认值）'} |")
    lines.append("")

    # ---------- 2) 数据源逐个实测 ----------
    lines += ["### 2. 数据源实测", "",
              "| 源 | 优先级 | 状态 | 耗时 | 说明 |",
              "|----|--------|------|------|------|"]
    chain = FetcherChain(priority=order, use_cache=False)
    for name in order:
        inst = chain._get(name)                                  # noqa: SLF001
        if inst is None:
            continue
        rk = order.index(name) + 1
        if hasattr(inst, "available") and not inst.available():
            lines.append(f"| `{name}` | {rk} | ⏭ 跳过 | — | {_skip_reason(name)} |")
            continue
        ok, bars, sec, err = _probe(inst.fetch, sym, start, end,
                                    adjust=cfg["adjust"], is_index=False)
        if ok and bars:
            lines.append(f"| `{name}` | {rk} | ✅ 可用 | {sec:.1f}s | "
                         f"{len(bars)} 根K线，最新 {bars[-1].date} 收 {bars[-1].close:.2f} |")
        else:
            lines.append(f"| `{name}` | {rk} | ❌ 失败 | {sec:.1f}s | {err[:70]} |")
            warns.append(name)
    lines.append("")

    # ---------- 3) tushare 接口分级 ----------
    lines += ["### 3. tushare 接口分级体检", ""]
    if not tok or not _importable("tushare"):
        lines += ["跳过：需要 `TUSHARE_TOKEN` **且** 安装 `tushare`。",
                  "token 获取：https://tushare.pro/user/token", ""]
    else:
        lines += ["同一个 token 上，不同接口的门槛差得很远。逐个戳一遍，"
                  "免得把「积分不够」误判成「代码写错了」。", "",
                  "| 接口 | 积分门槛 | 状态 | 说明 |",
                  "|------|----------|------|------|"]
        try:
            pro = _tushare_pro()
        except Exception as exc:  # noqa: BLE001
            lines.append(f"| — | — | ❌ | 客户端构造失败：{exc} |")
            pro = None
        if pro is not None:
            code = to_ts_code(sym)
            bench = to_ts_code(cfg["benchmark"])
            probes = [
                ("daily", lambda: pro.daily(ts_code=code,
                                            start_date=start, end_date=end)),
                ("adj_factor", lambda: pro.adj_factor(ts_code=code,
                                                      start_date="20150101")),
                ("daily_basic", lambda: pro.daily_basic(
                    ts_code=code, start_date=start, end_date=end,
                    fields="trade_date,turnover_rate")),
                ("index_daily", lambda: pro.index_daily(ts_code=bench,
                                                        start_date=start, end_date=end)),
            ]
            for api, fn in probes:
                need, cn = _TS_REQUIREMENTS.get(api, ("?", api))
                if args.skip_limited and api != "daily":
                    lines.append(f"| `{api}` | ≥{need} | ⏭ 跳过 | {cn}（--skip-limited） |")
                    continue
                ok, df, sec, err = _probe(fn)
                if ok and df is not None and not getattr(df, "empty", True):
                    lines.append(f"| `{api}` | ≥{need} | ✅ 可用 | {cn}，{len(df)} 行，{sec:.1f}s |")
                elif ok:
                    lines.append(f"| `{api}` | ≥{need} | ⚠️ 空结果 | "
                                 f"{cn}（非交易日或区间无数据） |")
                else:
                    kind = _classify_ts_error(err)
                    lines.append(f"| `{api}` | ≥{need} | ❌ {kind} | {err[:70]} |")
                    if kind.startswith("限流"):
                        warns.append(api)
        lines += ["",
                  "> 限流不一定拦得住你 —— **关键看这个数据多久变一次**：",
                  "> `adj_factor` 只在分红除权那天变（一年一两次），",
                  "> `index_daily` / `daily_basic` 只追加不回改。",
                  "> 所以整段一次拉下来落盘、按天/周刷新即可，"
                  "日常跑日报根本不会再碰这些接口。",
                  f"> 复权因子缓存：`{_rel(CACHE_DIR / 'adj_factor')}`（TTL 7 天）", ""]

    # ---------- 4) 本地缓存 ----------
    cache = CsvCache()
    files = sorted(cache.dir.glob("*.csv"))
    lines += ["### 4. 本地缓存", "",
              f"- 目录：`{_rel(cache.dir)}`", f"- 行情文件：{len(files)} 个"]
    if files:
        for f in files[:8]:
            pr = f.stem.split("_", 1)
            fresh = cache.is_fresh(pr[0], pr[1] if len(pr) > 1 else "qfq")
            age = (time.time() - f.stat().st_mtime) / 86400.0
            lines.append(f"  - `{f.name}`　{age:.1f} 天前　"
                         f"{'🟢 未过期' if fresh else '🟡 已过期（下次会自动重抓）'}")
        lines.append(f"- TTL：{cache.ttl_days:g} 天"
                     "（不是为省流量，是为正确性：前复权价每次分红后整体重算）")
    lines.append("")

    # ---------- 5) LLM ----------
    lines += ["### 5. LLM 分析器", ""]
    if args.skip_llm:
        lines += ["跳过（--skip-llm）。", ""]
    else:
        analyzer = build_analyzer("mock" if args.mock else cfg["analyzer"])
        if isinstance(analyzer, MockAnalyzer):
            lines += ["⚠️ 未检测到可用 LLM，当前会走**离线规则模拟器**（MockAnalyzer）。",
                      "配置真模型见 `.env.example`（本地 Ollama 零成本）。", ""]
            warns.append("LLM(未配置)")
        else:
            local = _is_local_url(analyzer.base_url)
            lines += [f"- 端点：`{analyzer.base_url}`"
                      f"（{'本地推理，零成本' if local else '云端 API'}）",
                      f"- 模型：`{analyzer.model}`　超时 {analyzer.timeout}s"
                      f"{f'　max_tokens={analyzer.max_tokens}' if analyzer.max_tokens else ''}",
                      ""]
            # LLM 探测要 65 根以上K线（60 日均线 + 缓冲才能拼出快照），
            # 所以单独用更长的窗口取数 —— 别被上面 45 天的探测窗口卡住，
            # 那样会得到"配置看起来对、但其实一次都没真调过"的假绿灯。
            ls, le = date_range(cfg, days_back=_LLM_PROBE_DAYS)
            bars = None
            try:
                bars, _ = FetcherChain(priority=order).fetch(sym, ls, le,
                                                             adjust=cfg["adjust"])
            except Exception as exc:  # noqa: BLE001
                log.debug("LLM 探测取数失败: %s", exc)
            if bars and len(bars) >= 65:
                snap = _snap(sym, bars, len(bars) - 1)
                # 必须绕过 prompt 哈希缓存真打一次 —— 体检命中缓存等于没验证，
                # 那是"假绿灯"，比红灯更危险。
                prev_cache = getattr(analyzer, "use_cache", True)
                analyzer.use_cache = False
                try:
                    ok, sig, sec, err = _probe(analyzer.analyze, snap)
                finally:
                    analyzer.use_cache = prev_cache
                if ok:
                    cost = ("本地推理，无 API 费用" if local
                            else f"本次约 ¥{analyzer.cost_report().get('est_cost_cny', 0):.4f}")
                    lines += [f"✅ 连通成功，单次分析 {sec:.1f}s（未走缓存，快照 {snap.date}）",
                              f"- 输出：`{sig.action.value}`　score={sig.score:.0f}　"
                              f"confidence={sig.confidence:.2f}",
                              f"- 理由：{sig.reason}",
                              f"- 成本：{cost}", ""]
                else:
                    lines += [f"❌ 调用失败（{sec:.1f}s）：{err[:200]}", ""]
                    warns.append("LLM")
            else:
                lines += [f"⚠️ 取到 {len(bars) if bars else 0} 根K线，不足 65 根，"
                          "无法构造快照 —— 只验证了配置，没有真调过模型。", ""]

    # ---------- 6) 结论 ----------
    lines += ["### 6. 结论", ""]
    if warns:
        lines += [f"⚠️ {len(warns)} 项需要注意：" + "、".join(f"`{w}`" for w in warns),
                  "",
                  "处理顺序：**先看降级链兜不兜得住，再决定要不要花钱。**"
                  "只要链上还有 ✅ 的源，日常 `analyze` 就跑得动；"
                  "`validate` 也只需要主源 + 基准能取到。", ""]
    else:
        lines += ["✅ 全部通过，可以跑 `analyze` / `validate` 了。", ""]
    lines += [f"_体检耗时 {time.perf_counter() - t_start:.1f}s_", ""]

    body = "\n".join(lines)
    print(body)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        log.info("体检报告已写出 → %s", _rel(out))
    return 1 if warns else 0


# ============================================================
# ledger —— 台账
# ============================================================

def cmd_ledger(args, cfg: dict) -> int:
    led = SignalLedger()
    st = led.stats()
    print("## 信号台账")
    print()
    if not st.get("total"):
        print(f"台账为空（{led.path}）")
        print("先跑 `python main.py analyze` 积累信号；攒够 30 条以上再跑 validate --strategy ledger")
        return 0
    print(f"- 文件：{led.path}")
    print(f"- 总信号：{st['total']} 条（可执行 {st['actionable']} 条）")
    print(f"- 区间：{st['date_range'][0]} ~ {st['date_range'][1]}")
    print(f"- 覆盖标的：{st['symbols']} 只")
    print(f"- 按来源：{st['by_source']}")
    print(f"- 按方向：{st['by_action']}")
    print()
    if args.tail:
        rows = led.read()[-args.tail:]
        print(f"最近 {len(rows)} 条：")
        print()
        print("| 日期 | 标的 | 信号 | 评分 | 来源 | 理由 |")
        print("|------|------|------|------|------|------|")
        for s in rows:
            print(f"| {s.date} | {s.symbol} | {s.action.value} | {s.score:.0f} | "
                  f"{s.source} | {s.reason[:28]} |")
    cache = SignalCache()
    print(f"\nLLM 缓存条目：{len(cache)}（缓存让回测重跑免费）")
    return 0


# ============================================================
# asof —— 时点审计：这次运行「本不该看到」哪些东西
# ============================================================

# 真实的茅台除权样本，用来演示「前复权基准日」的时点问题。
# 因子不靠外部数据源给，而是从除权事件本身反解出来（推导见 tests/test_adjust.py），
# 保证这段演示在任何机器上都能跑，且数字是真实的。
_ADJ_DEMO = {
    "event_date": "2025-06-26",       # 除权日
    "probe_date": "2025-06-25",       # 探针日：除权的前一天
    "raw_close": 1435.86,             # tushare 不复权收盘
    "ex_close_ref": 1408.26,          # tushare 给的除权后参考价
    "tencent_qfq": 1356.28,           # 腾讯前复权（独立源，用来验证算法而非论证观点）
    "latest_date": "2026-09-15",
    "latest_factor": 8.6463,
    "latest_raw": 1272.75,
}


def _try_fetch(chain, sym, start, end, cfg, as_of):
    """抓一次行情，失败不抛错（审计命令要能容错，不能因为没网就什么都不给）。"""
    try:
        bars, src = chain.fetch(sym, start, end, adjust=cfg["adjust"], as_of=as_of)
        return bars, src, ""
    except Exception as exc:  # noqa: BLE001
        return [], "", f"{type(exc).__name__}: {exc}"


def cmd_asof(args, cfg: dict) -> int:
    """
    把「时点门控」从一句承诺变成一次可复跑的演示。

    两段：
      [1] 数据层截断   真实数据。请求区间被钉在 as_of 内之后，实际裁掉几根 K 线
      [2] 复权基准日   A 股特有的一维：qfq 的分母该取 as_of 还是取「今天」
    """
    sym = args.symbol or _PROBE_SYMBOL
    as_of = asof.norm(args.as_of) if args.as_of else asof.today()
    start, end = date_range(cfg, days_back=args.days_back)
    chain = make_chain(cfg)
    kind = "历史运行" if asof.is_historical(as_of) else "实时运行"

    print(f"# 时点审计 · {sym}")
    print()
    print(f"- as_of：`{as_of}`（{kind}）")
    print(f"- 请求区间：`{asof.norm(start)}` ~ `{asof.norm(end)}`")
    print()

    # ---------- 1) 数据层截断 ----------
    print("## 1. 数据层截断 —— as_of 之后才存在的 K 线")
    print()
    raw, src_raw, err_raw = _try_fetch(chain, sym, start, end, cfg, None)
    gated, src_g, err_g = _try_fetch(chain, sym, start, end, cfg, as_of)

    if err_g or not gated:
        print(f"⚠️ 无法取数（{err_g or '返回空'}），跳过本段 —— 这一段需要行情源可用")
    else:
        tail_raw = raw[-1].date if raw else "—"
        tail_gated = gated[-1].date if gated else "—"
        cut = len(raw) - len(gated)
        pct_cut = (cut / len(raw) * 100) if raw else 0.0
        print("| 口径 | 实际请求区间 | K线根数 | 末根日期 | 数据源 |")
        print("|------|------------|--------|---------|--------|")
        print(f"| 未门控 | {asof.norm(start)} ~ {asof.norm(end)} | {len(raw)} | {tail_raw} | {src_raw or '—'} |")
        print(f"| 门控后 | {asof.norm(start)} ~ {as_of} | {len(gated)} | {tail_gated} | {src_g or '—'} |")
        print()
        if cut > 0:
            print(f"**{cut} 根 K 线（{pct_cut:.1f}%）被挡在门外** —— "
                  f"它们在 {as_of} 当天还不存在，却足以污染均线、量比和区间位置。")
        else:
            print("本次 as_of 与数据末尾重合，没有可裁的 K 线 —— 门控是空转的。")
        print()
        print("注意：请求区间在数据层就被压缩了，**未来数据连请求都没发出去**。")
        print("对带限流的数据源（如 tushare adj_factor 1 次/小时），这不是小事。")

    # ---------- 2) 复权基准日 ----------
    print()
    print("## 2. 复权基准日 —— A 股特有的那一维")
    print()
    d = _ADJ_DEMO
    ratio = d["raw_close"] / d["ex_close_ref"]              # adj_new / adj_old
    adj_old = d["latest_factor"] / (d["raw_close"] / d["tencent_qfq"])
    adj_new = adj_old * ratio
    factors = {
        d["probe_date"]: adj_old,
        d["event_date"]: adj_new,
        d["latest_date"]: d["latest_factor"],
    }
    bars = [Bar(date=d["probe_date"], open=d["raw_close"], close=d["raw_close"],
                high=d["raw_close"], low=d["raw_close"], volume=0.0, amount=0.0,
                pct_chg=0.0, turnover=0.0, amplitude=0.0),
            Bar(date=d["latest_date"], open=d["latest_raw"], close=d["latest_raw"],
                high=d["latest_raw"], low=d["latest_raw"], volume=0.0, amount=0.0,
                pct_chg=0.0, turnover=0.0, amplitude=0.0)]

    p_asof = _apply_adjust(bars, factors, "qfq", as_of=d["probe_date"])[0].close
    p_today = _apply_adjust(bars, factors, "qfq", as_of=None)[0].close
    drift = (p_today / p_asof - 1) * 100

    print(f"样本：{d['event_date']} 除权（因子 {adj_old:.4f} → {adj_new:.4f}，"
          f"跳变 {(ratio - 1) * 100:.3f}%）")
    print(f"探针日：{d['probe_date']}（除权**前**一天）")
    print()
    print("| 基准日取法 | 探针日收盘价 | 说明 |")
    print("|-----------|-------------|------|")
    print(f"| `as_of={d['probe_date']}` | **{p_asof:.2f}** | 当天交易所显示的价格 |")
    print(f"| 「今天」最新因子 {d['latest_factor']:.4f} | {p_today:.2f} | 含 {d['event_date']} 那次除权 |")
    print()
    print(f"**偏差 {drift:+.2f}%** —— 这 {abs(drift):.2f}% 全部来自 {d['event_date']} 的除权，"
          f"而在 {d['probe_date']} 那天它还**没有发生**。")
    print(f"（顺带验证算法没错：今天口径 {p_today:.2f} 与腾讯前复权 "
          f"{d['tencent_qfq']:.2f} 一致 —— 这正是「用最新因子」的标准结果。）")
    print()
    print("### 这个偏差影响什么、不影响什么")
    print()
    print("| | 影响 |")
    print("|---|---|")
    print("| 日收益率序列 | 不受影响（复权是等比缩放，缩放系数约掉） |")
    print("| 绝对价位 | **受影响**：止损价、目标价、「突破某价位」类规则都按绝对价算 |")
    print("| 跨次运行一致性 | **受影响**：每次分红后基准日漂移，同一段历史算出不同曲线 |")
    print("| 将来接入的新闻/基本面 | **受影响**：那是另一维，见 `core/asof.py` 的 filter_dated |")
    print()
    print("所以正确做法不是「复权了就行」，而是**把基准日钉在 as_of** —— "
          "或者干脆回测统一用后复权（基准固定在最早一日，天然不随时间漂移）。")
    return 0


# ============================================================
# 入口
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dsa-lite",
        description="带策略验证的 LLM 股票分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python main.py analyze --mock\n"
               "  python main.py validate --strategy five_dim --split 0.7 --out reports/validate.md\n"
               "  python main.py compare --strategies ma_cross,five_dim,rsi_reversion\n"
               "  python main.py asof --as-of 2025-06-20        # 看这次运行本不该看到哪些数据\n")
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    p.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze", help="每日分析并推送")
    a.add_argument("--symbols", help="逗号分隔，覆盖配置")
    a.add_argument("--mock", action="store_true", help="离线模式（不调 LLM）")
    a.add_argument("--no-notify", action="store_true", help="不推送")
    a.add_argument("--out", default="", help="看板输出路径")
    a.add_argument("--days-back", type=int, default=400)
    a.add_argument("--refresh", action="store_true", help="忽略缓存重新抓取")

    b = sub.add_parser("backtest", help="基础回测")
    _add_common(b)
    b.add_argument("--strategy", default="five_dim")
    b.add_argument("--show-trades", action="store_true")
    b.add_argument("--trades-limit", type=int, default=30)

    v = sub.add_parser("validate", help="策略体检（五大指标+过拟合+显著性）")
    _add_common(v)
    v.add_argument("--strategy", default="five_dim",
                   help="five_dim / ma_cross / rsi_reversion / llm / ledger")
    v.add_argument("--split", type=float, default=0.0, help="样本内占比，默认 0.7")
    v.add_argument("--min-trades", type=int, default=0, help="判定所需最少交易笔数")
    v.add_argument("--mock", action="store_true")

    c = sub.add_parser("compare", help="多策略横向对比")
    _add_common(c)
    c.add_argument("--strategies", default="ma_cross,five_dim,rsi_reversion")
    c.add_argument("--split", type=float, default=0.7, help="样本内占比")
    c.add_argument("--mock", action="store_true")

    l = sub.add_parser("ledger", help="查看信号台账")
    l.add_argument("--tail", type=int, default=10)

    d = sub.add_parser("doctor", help="体检：环境 / 数据源 / LLM 通不通")
    d.add_argument("--symbol", default="", help="探测标的，默认 sh600519")
    d.add_argument("--skip-llm", action="store_true", help="跳过 LLM 连通性测试")
    d.add_argument("--skip-limited", action="store_true",
                   help="跳过 tushare 限流接口（省下每小时一次的配额）")
    d.add_argument("--mock", action="store_true", help="只验证离线模拟器")
    d.add_argument("--out", default="", help="报告输出路径")

    s = sub.add_parser("asof", help="时点审计：这次运行本不该看到哪些数据")
    s.add_argument("--symbol", default="", help="探测标的，默认 sh600519")
    s.add_argument("--as-of", default="", metavar="YYYY-MM-DD",
                   help="审计时点，默认今天")
    s.add_argument("--days-back", type=int, default=400)
    s.add_argument("--refresh", action="store_true")
    return p


def _add_common(sp) -> None:
    sp.add_argument("--symbols", help="逗号分隔标的，默认取配置 watchlist")
    sp.add_argument("--days-back", type=int, default=700)
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--as-of", default="", metavar="YYYY-MM-DD",
                    help="时点门控：只使用该日期及之前可见的数据")
    sp.add_argument("--out", default="")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    if not args.cmd:
        build_parser().print_help()
        return 0

    _load_dotenv()
    cfg = load_config(args.config)
    handler = {
        "doctor": cmd_doctor,
        "analyze": cmd_analyze, "backtest": cmd_backtest,
        "validate": cmd_validate, "compare": cmd_compare, "ledger": cmd_ledger,
        "asof": cmd_asof,
    }.get(args.cmd)
    if handler is None:
        print(f"未知命令: {args.cmd}")
        return 2
    try:
        return handler(args, cfg)
    except NoDataSourceError as exc:
        log.error("数据源全部失败：%s", exc)
        log.error("检查网络，或 pip install akshare 增加备用源")
        return 3
    except KeyboardInterrupt:
        log.info("已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
