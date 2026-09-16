#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告渲染层 —— 决策看板
============================================================
把 Signal 列表渲染成一份人能直接看的 Markdown。

关键设计：报告里强制带上「这条信号的可信度来源」——
数据源是谁、模型是什么、提示词版本号。没有这三样，
三个月后你看到一条"买入"信号，根本不知道该怎么复盘。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .backtest import BacktestResult
from .signals import Action, Signal

# A股习惯：涨红跌绿
ICON = {Action.BUY: "🔴", Action.SELL: "🟢", Action.HOLD: "⚪", Action.AVOID: "⛔"}
LABEL = {Action.BUY: "买入", Action.SELL: "卖出", Action.HOLD: "观望", Action.AVOID: "回避"}


def render_dashboard(signals: list[Signal], positions: Optional[list[dict]] = None,
                     data_sources: Optional[dict[str, str]] = None,
                     errors: Optional[list[str]] = None,
                     title_date: Optional[str] = None) -> str:
    """生成每日决策仪表盘。"""
    d = title_date or (signals[0].date if signals else datetime.now().strftime("%Y-%m-%d"))
    L = []
    A = L.append
    A(f"🎯 {d} 决策看板")
    A("")

    if not signals:
        A("本次无可用信号。")
        if errors:
            A("")
            A("**失败原因：**")
            for e in errors[:10]:
                A(f"- {e}")
        return "\n".join(L)

    counts = {a: 0 for a in Action}
    for s in signals:
        counts[s.action] = counts.get(s.action, 0) + 1
    A(f"共分析 {len(signals)} 只 | "
      f"{ICON[Action.BUY]}买入 {counts[Action.BUY]} · "
      f"{ICON[Action.SELL]}卖出 {counts[Action.SELL]} · "
      f"{ICON[Action.HOLD]}观望 {counts[Action.HOLD]} · "
      f"{ICON[Action.AVOID]}回避 {counts[Action.AVOID]}")
    A("")

    # ---- 汇总表：按分数降序，可执行信号排前面 ----
    A("## 信号汇总")
    A("")
    A("| 标的 | 信号 | 评分 | 置信 | 止损 | 目标 | 盈亏比 | 理由 |")
    A("|------|------|------|------|------|------|--------|------|")
    order = {Action.BUY: 0, Action.SELL: 1, Action.AVOID: 2, Action.HOLD: 3}
    for s in sorted(signals, key=lambda x: (order[x.action], -x.score)):
        rr = f"{s.risk_reward:.2f}" if s.risk_reward else "—"
        A(f"| {s.symbol} | {ICON[s.action]}{LABEL[s.action]} | {s.score:.0f} | "
          f"{s.confidence:.0%} | {_p(s.stop)} | {_p(s.target)} | {rr} | {s.reason[:34]} |")
    A("")

    # ---- 持仓状态 ----
    if positions:
        A("## 当前持仓")
        A("")
        A("| 标的 | 成本 | 现价 | 浮盈 | 持仓天数 | 状态 |")
        A("|------|------|------|------|----------|------|")
        for p in positions:
            pnl = (p.get("price", 0) / p["cost"] - 1) * 100 if p.get("cost") else 0
            A(f"| {p['symbol']} | {p['cost']:.2f} | {p.get('price', 0):.2f} | "
              f"{pnl:+.2f}% | {p.get('hold_days', '—')} | {p.get('state', '持有')} |")
        A("")

    # ---- 值得关注的个股详情：只展开可执行信号 ----
    actionable = [s for s in signals if s.action in (Action.BUY, Action.SELL)]
    if actionable:
        A("## 可执行信号详情")
        A("")
        for s in actionable:
            A(f"### {ICON[s.action]} {s.symbol} — {LABEL[s.action]}（评分 {s.score:.0f}）")
            A("")
            A(f"- 理由：{s.reason}")
            ref = s.meta.get("snapshot", {}).get("收盘价")
            if s.action == Action.BUY:
                if s.stop:
                    A(f"- 止损：{s.stop:.2f}"
                      + (f"（距参考价 {_dist(ref, s.stop):+.2f}%）" if ref else ""))
                if s.target:
                    A(f"- 目标：{s.target:.2f}"
                      + (f"（距参考价 {_dist(ref, s.target):+.2f}%）" if ref else ""))
                if s.risk_reward:
                    A(f"- 盈亏比：{s.risk_reward:.2f}"
                      + (" ⚠️ 低于 1.5，不建议参与" if s.risk_reward < 1.5 else " ✅"))
                A(f"- 预期周期：{s.horizon_days} 个交易日")
                if ref:
                    A(f"- ⚠️ 参考价 {ref} 仅为信号生成日收盘价，"
                      f"实际入场以次日开盘为准（回测同样如此）")
            else:
                A("- 性质：**离场信号** —— A股不能做空，卖出后转为空仓，"
                  "等待下一个入场机会，不给止损/目标价")
            snap = s.meta.get("snapshot")
            if snap:
                A("")
                A("| 指标 | 值 | 指标 | 值 |")
                A("|------|-----|------|-----|")
                keys = list(snap.items())
                for i in range(0, len(keys) - 1, 2):
                    k1, v1 = keys[i]
                    k2, v2 = keys[i + 1]
                    A(f"| {k1} | {v1} | {k2} | {v2} |")
                if len(keys) % 2:
                    k1, v1 = keys[-1]
                    A(f"| {k1} | {v1} | | |")
            A("")

    # ---- 数据溯源（强制打印，复盘的生命线）----
    A("## 数据溯源")
    A("")
    src = {}
    for s in signals:
        src[s.source] = src.get(s.source, 0) + 1
    A(f"- 信号来源分布：{src}")
    if signals[0].model:
        A(f"- 模型：`{signals[0].model}`")
    A(f"- 提示词版本：`{signals[0].prompt_version}`"
      + ("（换版本必须重跑回测）" if signals[0].prompt_version else ""))
    if data_sources:
        A("- 行情数据源：" + "，".join(f"{k}→{v}" for k, v in list(data_sources.items())[:12])
          + ("…" if len(data_sources) > 12 else ""))
    A(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A("")

    # ---- 免责声明（照抄 DSA 的克制，但把话说透）----
    A("---")
    A("")
    A("> ⚠️ 本看板由量化规则与语言模型生成，**未经实盘验证**。"
      "模型给出的评分与价位不代表投资建议。")
    A("> 用之前先跑 `python main.py validate`，看这套逻辑在历史数据上是否有正期望。")
    if errors:
        A("")
        A(f"> 本次有 {len(errors)} 只标的获取数据失败：" + "；".join(errors[:5]))
    return "\n".join(L)


def render_backtest_summary(result: BacktestResult) -> str:
    """回测结果的一页纸摘要（终端用）。"""
    m = result.metrics
    L = [
        f"策略：{result.strategy_name}    区间：{result.period[0]} ~ {result.period[1]}",
        "-" * 58,
        f"累计收益 {m.total_return_pct:+.2f}%   年化 {m.annual_return_pct:+.2f}%   "
        f"基准 {m.benchmark_return_pct:+.2f}%   超额 {m.excess_return_pct:+.2f}%",
        f"交易 {m.trades} 笔（胜 {m.wins}/负 {m.losses}）  胜率 {m.win_rate_pct:.1f}%   "
        f"盈亏比 {m.profit_loss_ratio:.2f}  期望 {m.expectancy_pct:+.3f}%",
        f"最大回撤 {m.max_drawdown_pct:.2f}%   夏普 {m.sharpe:.2f}   "
        f"索提诺 {m.sortino:.2f}   卡玛 {m.calmar:.2f}",
        f"频率 {m.trades_per_month:.2f} 笔/月   平均持仓 {m.avg_hold_days:.1f} 日   "
        f"持仓占比 {m.exposure_pct:.1f}%",
        f"显著性 t={m.alpha_t_stat:.2f}  p={m.alpha_p_value:.4f}  "
        f"{'✅显著' if m.is_significant else '⚠️不显著'}",
    ]
    if result.warnings:
        L.append("-" * 58)
        for w in result.warnings[:5]:
            L.append(f"⚠️ {w}")
    return "\n".join(L)


def render_trades(result: BacktestResult, limit: int = 30) -> str:
    L = ["| # | 标的 | 买入日 | 买价 | 卖出日 | 卖价 | 收益% | 持有 | 出场原因 |",
         "|---|------|--------|------|--------|------|-------|------|----------|"]
    for i, t in enumerate(result.trades[:limit], 1):
        L.append(f"| {i} | {t.symbol} | {t.entry_date} | {t.entry_price:.2f} | "
                 f"{t.exit_date} | {t.exit_price:.2f} | {t.pnl_pct:+.2f} | "
                 f"{t.hold_days} | {t.exit_reason} |")
    if len(result.trades) > limit:
        L.append(f"| … | 共 {len(result.trades)} 笔，此处仅显示前 {limit} 笔 | | | | | | | |")
    return "\n".join(L)


def _p(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.2f}"


def _dist(cur, target) -> float:
    if not cur or not target:
        return 0.0
    return (target / cur - 1) * 100
