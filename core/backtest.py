#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎 —— 策略验证的执行内核
============================================================
这是 daily_stock_analysis 没有的部分，也是「LLM 炒股到底行不行」
唯一能给出答案的地方。

设计上把三件事做死：

一、杜绝未来函数
    T 日收盘后生成信号 → T+1 日开盘成交。
    止损/止盈允许在日内触发（用当日 high/low 判定），
    但入场永远在下一根K线的开盘价，且成交价必须叠加上不利滑点。
    引擎自带 verify_no_lookahead()，用「扰动未来K线」的方式
    反向验证策略没偷看未来 —— 这是回测可信度的底线检查。

二、成本必须真实
    佣金（万2.5，最低5元）+ 印花税（卖出千1）+ 过户费 + 滑点，
    再加 A 股两个硬规则：T+1 不可当日卖出、涨停买不进/跌停卖不出。
    很多"年化 80%"的回测，把成本补上就变成 8%。

三、指标对齐 a-share-master 验证层
    胜率 / 盈亏比 / 最大回撤 / 夏普 / 交易频率，五项与
    《验证层-回测框架》的及格线、优秀线一一对应，
    回测结果可以直接填进季度体检表。
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, asdict, field
from typing import Optional

from . import asof
from .indicators import drawdown_series
from .signals import Action, Signal

log = logging.getLogger("backtest")


# ============================================================
# 配置
# ============================================================

@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    per_trade_pct: float = 0.30        # 单笔最大占总权益比例
    max_positions: int = 5             # 最大同时持仓数

    # 成本
    commission_rate: float = 0.00025   # 万 2.5
    commission_min: float = 5.0
    stamp_tax_rate: float = 0.0005     # 卖出千 0.5（2023-08 起降至 0.05%）
    transfer_rate: float = 0.00001     # 过户费
    slippage_pct: float = 0.001        # 单边滑点 0.1%

    # 风控
    stop_loss_pct: float = 0.08        # 硬止损 -8%
    take_profit_pct: float = 0.25      # 止盈 +25%
    trailing_stop_pct: float = 0.10    # 移动止损：从最高点回撤 10%
    max_holding_days: int = 30         # 超期强制离场
    cooldown_after_losses: int = 3     # 连亏 N 笔
    cooldown_days: int = 5             # 熔断 M 个交易日

    # 规则
    enforce_t_plus_1: bool = True
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    min_hold_to_use_target: int = 1     # 至少持有几天才允许止盈

    # 其他
    risk_free_rate: float = 0.02
    use_stop_from_signal: bool = True   # 优先用信号自带止损价，其次用百分比
    lookahead_check: bool = True


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: str
    entry_cost: float          # 含费用的总成本
    stop: Optional[float] = None
    target: Optional[float] = None
    high_water: float = 0.0    # 持仓期最高价（移动止损用）
    hold_days: int = 0
    signal_score: float = 50.0
    entry_reason: str = ""
    mae: float = 0.0           # 最大不利偏移 %
    mfe: float = 0.0           # 最大有利偏移 %

    def __post_init__(self):
        self.high_water = max(self.high_water, self.entry_price)


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str
    entry_reason: str = ""
    signal_score: float = 50.0
    mae: float = 0.0
    mfe: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class Metrics:
    # 收益
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    excess_return_pct: float = 0.0
    final_equity: float = 0.0

    # 交易质量
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_loss_ratio: float = 0.0     # 盈亏比
    profit_factor: float = 0.0         # 总盈利 / 总亏损
    expectancy_pct: float = 0.0        # 单笔期望收益

    # 风险
    max_drawdown_pct: float = 0.0
    max_drawdown_days: int = 0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    volatility_pct: float = 0.0

    # 频率
    trades_per_month: float = 0.0
    avg_hold_days: float = 0.0
    exposure_pct: float = 0.0          # 有持仓的交易日占比

    # 显著性
    alpha_t_stat: float = 0.0
    alpha_p_value: float = 1.0
    is_significant: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    metrics: Metrics
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    benchmark_curve: list[tuple[str, float]] = field(default_factory=list)
    equity_dates: list[str] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    benchmark_values: list[float] = field(default_factory=list)
    signal_counts: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    period: tuple[str, str] = ("", "")
    strategy_name: str = ""
    as_of: str = ""            # 本次运行的时点（空 = 未指定门控）

    def in_sample_split(self, ratio: float = 0.7) -> tuple["BacktestResult", "BacktestResult"]:
        """按时间切分为样本内 / 样本外两份结果，用于过拟合检测。"""
        if not self.trades:
            return self, self
        cut = sorted({t.entry_date for t in self.trades})[
            int(len({t.entry_date for t in self.trades}) * ratio)] if self.trades else None
        if cut is None:
            return self, self
        ins = [t for t in self.trades if t.entry_date < cut]
        outs = [t for t in self.trades if t.entry_date >= cut]
        return (
            _rebuild(self, ins, f"{self.strategy_name}_IS"),
            _rebuild(self, outs, f"{self.strategy_name}_OOS"),
        )


# ============================================================
# 引擎
# ============================================================

class BacktestEngine:
    """多标的、单账户、事件驱动的日线回测引擎。"""

    def __init__(self, strategy, config: Optional[BacktestConfig] = None):
        self.strategy = strategy
        self.cfg = config or BacktestConfig()

    # ---------- 入口 ----------

    def run(self, bars_by_symbol: dict[str, list],
            benchmark_bars: Optional[list] = None,
            as_of: Optional[str] = None) -> BacktestResult:
        cfg = self.cfg
        result = BacktestResult(metrics=Metrics(), strategy_name=self.strategy.name)
        result.as_of = asof.norm(as_of) if as_of is not None else ""

        symbols = [s for s, b in bars_by_symbol.items() if b]
        if not symbols:
            result.warnings.append("无可用K线数据")
            return result

        # ---------------- 时点门控：数据层最后一道关 ----------------
        # 策略只能读 bars[:i+1] 是「索引约束」，它挡不住进来的数据本身就带
        # as_of 之后的 K 线（缓存回退、多源合并、将来接入的新闻都可能带）。
        # 这里做的是「数据约束」：越界的 bar 根本不存在于本次运行里。
        if as_of is not None:
            audit = asof.Audit(as_of=as_of)
            bars_by_symbol = {
                s: asof.clip_bars(b, as_of, label=f"{s}", audit=audit)
                for s, b in bars_by_symbol.items()}
            symbols = [s for s in symbols if bars_by_symbol.get(s)]
            if benchmark_bars:
                benchmark_bars = asof.clip_bars(benchmark_bars, as_of,
                                                label="benchmark", audit=audit)
            if not symbols:
                result.warnings.append(f"时点门控后无可用K线（as_of={audit.as_of}）")
                return result
            if audit.dirty:
                result.warnings.append("时点门控生效：" + "；".join(audit.summary()[1:]))
        else:
            # 门控没开也必须说一声 —— 静默地"没有门控"比门控裁掉数据更危险
            result.warnings.append(
                "未指定 as_of：本次回测不额外截断数据，时点正确性依赖调用方的数据区间")

        calendar = sorted({b.date for s in symbols for b in bars_by_symbol[s]})
        if len(calendar) < 30:
            result.warnings.append(f"交易日仅 {len(calendar)} 天，回测意义有限")

        # 建立 (symbol, date) → index 映射，避免 O(n) 线性查找
        idx = {s: {b.date: k for k, b in enumerate(bars_by_symbol[s])} for s in symbols}

        warmup = max(self.strategy.warmup(), 20)
        cash = cfg.initial_cash
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        equity_curve: list[tuple[str, float]] = []
        equity_vals: list[float] = []
        daily_rets: list[float] = []
        pending: list[Signal] = []          # 待 T+1 开盘执行的信号
        loss_streak = 0
        cooldown_until: Optional[str] = None
        exposure_days = 0
        prev_equity = cfg.initial_cash
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "AVOID": 0}

        # 基准序列
        bench_map = {b.date: b.close for b in (benchmark_bars or [])} if benchmark_bars else {}

        for di, date in enumerate(calendar):
            # ---------------- 1) 先处理待执行订单（T+1 开盘成交） ----------------
            if pending:
                still: list[Signal] = []
                for sig in pending:
                    k = idx.get(sig.symbol, {}).get(date)
                    if k is None:                       # 当日停牌，顺延
                        still.append(sig)
                        continue
                    bar = bars_by_symbol[sig.symbol][k]
                    prev_close = (bars_by_symbol[sig.symbol][k - 1].close
                                  if k > 0 else bar.open)

                    # -------- 卖出：信号次日开盘离场 --------
                    if sig.action == Action.SELL:
                        pos = positions.get(sig.symbol)
                        if pos is None:
                            continue
                        if cfg.enforce_t_plus_1 and date == pos.entry_date:
                            still.append(sig)           # T+1，顺延到明日
                            continue
                        if cfg.block_limit_down_sell and _is_limit_down(bar.open, prev_close, sig.symbol):
                            still.append(sig)           # 跌停卖不出，顺延
                            continue
                        fill = bar.open * (1 - cfg.slippage_pct)
                        cash, trade = self._close(cash, pos, date, fill, "策略卖出信号")
                        trades.append(trade)
                        loss_streak = 0 if trade.pnl > 0 else loss_streak + 1
                        if loss_streak >= cfg.cooldown_after_losses:
                            cooldown_until = calendar[min(len(calendar) - 1, di + cfg.cooldown_days)]
                        del positions[sig.symbol]
                        continue

                    # -------- 买入 --------
                    if sig.action != Action.BUY or sig.symbol in positions:
                        continue
                    if len(positions) >= cfg.max_positions:
                        continue
                    if cooldown_until and date <= cooldown_until:
                        continue
                    # A股规则：涨停买不进
                    if cfg.block_limit_up_buy and _is_limit_up(bar.open, prev_close, sig.symbol):
                        continue

                    alloc = min(cash, (equity_vals[-1] if equity_vals else cfg.initial_cash)
                                * cfg.per_trade_pct)
                    fill = bar.open * (1 + cfg.slippage_pct)
                    shares = int(alloc / fill / 100) * 100
                    if shares < 100:
                        continue
                    cost = fill * shares
                    fee = self._buy_fee(cost)
                    if cost + fee > cash:
                        shares -= 100
                        if shares < 100:
                            continue
                        cost = fill * shares
                        fee = self._buy_fee(cost)

                    cash -= cost + fee
                    stop = sig.stop if (cfg.use_stop_from_signal and sig.stop) else fill * (1 - cfg.stop_loss_pct)
                    target = sig.target if (cfg.use_stop_from_signal and sig.target) else fill * (1 + cfg.take_profit_pct)
                    positions[sig.symbol] = Position(
                        symbol=sig.symbol, shares=shares, entry_price=fill,
                        entry_date=date, entry_cost=cost + fee,
                        stop=stop, target=target, signal_score=sig.score,
                        entry_reason=sig.reason)
                pending = still

            # ---------------- 2) 持仓管理：日内触发止损/止盈 ----------------
            for sym in list(positions.keys()):
                pos = positions[sym]
                k = idx[sym].get(date)
                if k is None:
                    continue
                bar = bars_by_symbol[sym][k]
                prev_bar = bars_by_symbol[sym][k - 1] if k > 0 else bar
                pos.hold_days += 1
                pos.high_water = max(pos.high_water, bar.high)
                pos.mae = min(pos.mae, (bar.low / pos.entry_price - 1) * 100)
                pos.mfe = max(pos.mfe, (bar.high / pos.entry_price - 1) * 100)

                exit_price, reason = self._exit_decision(pos, bar, prev_bar, date, sym)
                if exit_price is None:
                    continue
                # 跌停卖不出：顺延到下一日
                if cfg.block_limit_down_sell and _is_limit_down(bar.open, prev_bar.close, sym):
                    continue
                cash, trade = self._close(cash, pos, date, exit_price, reason)
                trades.append(trade)
                loss_streak = 0 if trade.pnl > 0 else loss_streak + 1
                if loss_streak >= cfg.cooldown_after_losses:
                    cooldown_until = calendar[min(len(calendar) - 1,
                                                  di + cfg.cooldown_days)]
                del positions[sym]

            # ---------------- 3) 生成信号（只用到 date 及之前的数据） ----------------
            if di + 1 < len(calendar):
                for sym in symbols:
                    k = idx[sym].get(date)
                    if k is None or k + 1 < warmup:
                        continue
                    held = positions.get(sym)
                    ctx = None
                    if held:
                        ctx = {"entry": held.entry_price, "shares": held.shares,
                               "entry_date": held.entry_date, "high_water": held.high_water,
                               "hold_days": held.hold_days,
                               "pnl_pct": (bars_by_symbol[sym][k].close / held.entry_price - 1) * 100}
                    try:
                        sig = self.strategy.decide(sym, bars_by_symbol[sym], k, ctx)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("[%s] %s 策略异常: %s", sym, date, exc)
                        continue
                    signal_counts[sig.action.value] = signal_counts.get(sig.action.value, 0) + 1

                    if sig.action == Action.SELL and held:
                        # 卖出信号：次日开盘离场（交给 pending 机制）
                        pending.append(sig)
                    elif sig.action == Action.BUY and not held:
                        if len(positions) < cfg.max_positions and not (
                                cooldown_until and date <= cooldown_until):
                            pending.append(sig)

            # ---------------- 4) 收盘估值 ----------------
            mv = 0.0
            for sym, pos in positions.items():
                k = idx[sym].get(date)
                if k is not None:
                    mv += bars_by_symbol[sym][k].close * pos.shares
                else:
                    mv += pos.entry_price * pos.shares
            equity = cash + mv
            if positions:
                exposure_days += 1
            equity_curve.append((date, round(equity, 2)))
            equity_vals.append(equity)
            if prev_equity > 0:
                daily_rets.append(equity / prev_equity - 1)
            prev_equity = equity

        # ---------------- 收尾：强制平仓 ----------------
        if positions:
            last_date = calendar[-1]
            for sym, pos in list(positions.items()):
                k = idx[sym].get(last_date, len(bars_by_symbol[sym]) - 1)
                bar = bars_by_symbol[sym][k]
                cash, trade = self._close(cash, pos, last_date, bar.close, "回测结束强制平仓")
                trades.append(trade)
                del positions[sym]
            equity_vals[-1] = cash
            equity_curve[-1] = (last_date, round(cash, 2))

        # ---------------- 基准曲线对齐 ----------------
        bench_vals, bench_curve = [], []
        if bench_map:
            base = None
            for d, _ in equity_curve:
                if d in bench_map:
                    if base is None:
                        base = bench_map[d]
                    v = cfg.initial_cash * bench_map[d] / base
                else:
                    v = bench_vals[-1] if bench_vals else cfg.initial_cash
                bench_vals.append(v)
                bench_curve.append((d, round(v, 2)))

        result.trades = trades
        result.equity_curve = equity_curve
        result.equity_dates = [d for d, _ in equity_curve]
        result.equity_values = equity_vals
        result.daily_returns = daily_rets
        result.benchmark_curve = bench_curve
        result.benchmark_values = bench_vals
        result.signal_counts = signal_counts
        result.period = (calendar[0], calendar[-1])
        result.metrics = self._metrics(
            equity_vals, daily_rets, trades, bench_vals,
            exposure_days / max(1, len(calendar)), calendar)

        if cfg.lookahead_check:
            result.warnings.extend(self._lookahead_scan(bars_by_symbol, symbols, idx, warmup))
        return result

    # ---------- 平仓（统一出口，避免费用/滑点算法分叉） ----------

    def _close(self, cash: float, pos: Position, date: str,
               exit_price: float, reason: str) -> tuple[float, Trade]:
        fill = exit_price * (1 - self.cfg.slippage_pct)
        proceeds = fill * pos.shares
        fee = self._sell_fee(proceeds)
        cash += proceeds - fee
        pnl = proceeds - fee - pos.entry_cost
        trade = Trade(
            symbol=pos.symbol, entry_date=pos.entry_date, entry_price=pos.entry_price,
            exit_date=date, exit_price=round(fill, 3), shares=pos.shares,
            pnl=round(pnl, 2), pnl_pct=round(pnl / pos.entry_cost * 100, 3),
            hold_days=pos.hold_days, exit_reason=reason,
            entry_reason=pos.entry_reason, signal_score=pos.signal_score,
            mae=round(pos.mae, 2), mfe=round(pos.mfe, 2))
        return cash, trade

    # ---------- 出场逻辑 ----------

    def _exit_decision(self, pos: Position, bar, prev_bar, date: str,
                       sym: str) -> tuple[Optional[float], str]:
        cfg = self.cfg
        # T+1：当日买入不得当日卖出
        if cfg.enforce_t_plus_1 and date == pos.entry_date:
            return None, ""
        # 开盘已跌破止损 → 以开盘价成交（跳空缺口）
        if pos.stop and bar.open <= pos.stop:
            return bar.open, "跳空跌破止损"
        if pos.stop and bar.low <= pos.stop:
            return pos.stop, "触发止损"
        # 移动止损
        if pos.hold_days > 1 and pos.high_water > pos.entry_price * 1.05:
            trail = pos.high_water * (1 - cfg.trailing_stop_pct)
            if bar.open <= trail:
                return bar.open, "跳空跌破移动止损"
            if bar.low <= trail:
                return trail, "触发移动止损"
        # 止盈
        if pos.hold_days >= cfg.min_hold_to_use_target and pos.target:
            if bar.open >= pos.target:
                return bar.open, "跳空高开达标止盈"
            if bar.high >= pos.target:
                return pos.target, "达标止盈"
        # 超期离场
        if pos.hold_days >= cfg.max_holding_days:
            return bar.close, f"持有超{cfg.max_holding_days}日强制离场"
        return None, ""

    # ---------- 费用 ----------

    def _buy_fee(self, amount: float) -> float:
        c = self.cfg
        return max(amount * c.commission_rate, c.commission_min) + amount * c.transfer_rate

    def _sell_fee(self, amount: float) -> float:
        c = self.cfg
        return (max(amount * c.commission_rate, c.commission_min)
                + amount * c.stamp_tax_rate + amount * c.transfer_rate)

    # ---------- 指标计算 ----------

    def _metrics(self, equity: list[float], rets: list[float], trades: list[Trade],
                 bench: list[float], exposure: float, calendar: list[str]) -> Metrics:
        m = Metrics()
        cfg = self.cfg
        if not equity:
            return m

        m.final_equity = round(equity[-1], 2)
        m.total_return_pct = round((equity[-1] / cfg.initial_cash - 1) * 100, 2)

        n_days = len(equity)
        years = max(n_days / 252.0, 1e-6)
        if equity[-1] > 0 and cfg.initial_cash > 0:
            m.annual_return_pct = round(((equity[-1] / cfg.initial_cash) ** (1 / years) - 1) * 100, 2)

        if bench:
            m.benchmark_return_pct = round((bench[-1] / bench[0] - 1) * 100, 2)
        m.excess_return_pct = round(m.total_return_pct - m.benchmark_return_pct, 2)

        # 交易质量
        m.trades = len(trades)
        if trades:
            wins = [t for t in trades if t.is_win]
            losses = [t for t in trades if not t.is_win]
            m.wins, m.losses = len(wins), len(losses)
            m.win_rate_pct = round(len(wins) / len(trades) * 100, 2)
            m.avg_win_pct = round(statistics.mean([t.pnl_pct for t in wins]), 2) if wins else 0.0
            m.avg_loss_pct = round(statistics.mean([t.pnl_pct for t in losses]), 2) if losses else 0.0
            if wins and losses and m.avg_loss_pct != 0:
                m.profit_loss_ratio = round(abs(m.avg_win_pct / m.avg_loss_pct), 2)
            gross_win = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            m.profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
            m.expectancy_pct = round(statistics.mean([t.pnl_pct for t in trades]), 3)
            m.avg_hold_days = round(statistics.mean([t.hold_days for t in trades]), 1)
            months = max(len(calendar) / 21.0, 1e-6)
            m.trades_per_month = round(len(trades) / months, 2)

        # 风险
        dd = drawdown_series(equity)
        if dd:
            m.max_drawdown_pct = round(abs(min(dd)), 2)
            i_worst = dd.index(min(dd))
            peak_i = max(range(i_worst + 1), key=lambda j: equity[j]) if i_worst > 0 else 0
            m.max_drawdown_days = i_worst - peak_i
        if len(rets) > 1:
            sd = statistics.pstdev(rets)
            m.volatility_pct = round(sd * math.sqrt(252) * 100, 2)
            rf_daily = cfg.risk_free_rate / 252
            excess = [r - rf_daily for r in rets]
            if sd > 0:
                m.sharpe = round(statistics.mean(excess) / sd * math.sqrt(252), 2)
            downside = [r for r in excess if r < 0]
            if downside:
                dsd = math.sqrt(sum(r ** 2 for r in downside) / len(downside))
                if dsd > 0:
                    m.sortino = round(statistics.mean(excess) / dsd * math.sqrt(252), 2)
        if m.max_drawdown_pct > 0:
            m.calmar = round(m.annual_return_pct / m.max_drawdown_pct, 2)
        m.exposure_pct = round(exposure * 100, 1)

        # 显著性：日收益均值是否显著大于 0（单样本 t 检验，纯标准库实现）
        if len(rets) > 5:
            sd = statistics.pstdev(rets)
            if sd > 0:
                t = statistics.mean(rets) / (sd / math.sqrt(len(rets)))
                m.alpha_t_stat = round(t, 2)
                m.alpha_p_value = round(_t_p_value(t, len(rets) - 1), 4)
                m.is_significant = m.alpha_p_value < 0.05 and t > 0
        return m

    # ---------- 未来函数扫描 ----------

    def _lookahead_scan(self, bars_by_symbol, symbols, idx, warmup) -> list[str]:
        """
        反向验证：把 i 之后的K线全部替换成极端值，
        如果第 i 日的决策发生变化，说明策略偷看了未来。
        """
        warns = []
        for sym in symbols:
            bars = bars_by_symbol[sym]
            for probe in range(warmup, len(bars) - 1, max(1, len(bars) // 40)):
                try:
                    base = self.strategy.decide(sym, bars, probe, None)
                except Exception:  # noqa: BLE001
                    continue
                mutated = list(bars)
                for j in range(probe + 1, len(mutated)):
                    b = mutated[j]
                    mutated[j] = type(b)(
                        date=b.date, open=b.open * 3, close=b.close * 3, high=b.high * 3,
                        low=b.low * 3, volume=b.volume, amount=b.amount,
                        pct_chg=b.pct_chg, turnover=b.turnover, amplitude=b.amplitude)
                try:
                    alt = self.strategy.decide(sym, mutated, probe, None)
                except Exception:  # noqa: BLE001
                    continue
                if alt.action != base.action or abs((alt.score or 0) - (base.score or 0)) > 1e-6:
                    warns.append(
                        f"[未来函数] {sym} 在第 {probe} 根({bars[probe].date})："
                        f"篡改未来K线后信号由 {base.action.value} 变为 {alt.action.value}")
                    break
        return warns


# ============================================================
# 统计工具（纯标准库）
# ============================================================

def _t_p_value(t: float, df: int) -> float:
    """
    t 分布双尾 p 值的数值近似（Abramowitz & Stegun 26.7.8 连分式）。
    够用于判断显著性，不引入 scipy。
    """
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    ib = _betainc(a, b, x)
    return ib if t > 0 else ib


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b)，连分式实现。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    front = math.exp(lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x)
    return 1.0 - _betainc(b, a, 1 - x)


def _cf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-9) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _is_limit_up(price: float, prev_close: float, symbol: str) -> bool:
    if prev_close <= 0:
        return False
    pct = price / prev_close - 1
    limit = _price_limit(symbol)
    return pct >= limit - 0.002


def _is_limit_down(price: float, prev_close: float, symbol: str) -> bool:
    if prev_close <= 0:
        return False
    pct = price / prev_close - 1
    limit = _price_limit(symbol)
    return pct <= -limit + 0.002


def _price_limit(symbol: str) -> float:
    """涨跌停幅度：创业板/科创板 20%，主板 10%，北交所 30%。"""
    s = str(symbol)
    if s.startswith(("sz300", "sh688", "sh689")):
        return 0.20
    if s.startswith("bj"):
        return 0.30
    return 0.10


def _rebuild(src: BacktestResult, trades: list[Trade], name: str) -> BacktestResult:
    """
    用交易子集重建一份结果，用于样本内/外分段比较。

    净值按「单笔收益顺序复利」近似 —— 分段验证只关心两组交易
    的质量差异（胜率、盈亏比、期望），不需要精确还原仓位曲线。
    """
    eq = 1.0
    curve: list[tuple[str, float]] = []
    for t in trades:
        eq *= (1 + t.pnl_pct / 100)
        curve.append((t.exit_date, round(eq, 6)))
    vals = [v for _, v in curve]

    m = Metrics()
    m.trades = len(trades)
    if trades:
        wins = [t for t in trades if t.is_win]
        losers = [t for t in trades if not t.is_win]
        m.wins, m.losses = len(wins), len(losers)
        m.win_rate_pct = round(len(wins) / len(trades) * 100, 2)
        m.avg_win_pct = round(statistics.mean([t.pnl_pct for t in wins]), 2) if wins else 0.0
        m.avg_loss_pct = round(statistics.mean([t.pnl_pct for t in losers]), 2) if losers else 0.0
        if wins and losers and m.avg_loss_pct:
            m.profit_loss_ratio = round(abs(m.avg_win_pct / m.avg_loss_pct), 2)
        gw = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losers))
        m.profit_factor = round(gw / gl, 2) if gl > 0 else float("inf")
        m.expectancy_pct = round(statistics.mean([t.pnl_pct for t in trades]), 3)
        m.avg_hold_days = round(statistics.mean([t.hold_days for t in trades]), 1)
    m.total_return_pct = round((eq - 1) * 100, 2)
    m.final_equity = round(eq * src.metrics.final_equity, 2)
    if vals:
        m.max_drawdown_pct = round(abs(min(drawdown_series(vals))), 2)
    return BacktestResult(
        metrics=m, trades=trades, equity_curve=curve, equity_values=vals,
        equity_dates=[d for d, _ in curve], strategy_name=name,
        as_of=src.as_of,
        period=(trades[0].entry_date, trades[-1].exit_date) if trades else ("", ""))
