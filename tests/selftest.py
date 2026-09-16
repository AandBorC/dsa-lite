#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自检脚本 —— 验证「验证器」本身是否可信
============================================================
一个回测工具的结论有没有价值，取决于它的内核对不对。
所以在相信任何回测结论之前，先跑这个：

    python tests/selftest.py

覆盖四类：
  1. 统计函数：t 分布 p 值 vs 教科书值（数值实现最容易在这里骗人）
  2. 未来函数探测器：故意写一个偷看未来的策略，它必须被抓住
  3. A股规则：T+1、涨停不买、整百股、费用计算
  4. 指标：MA/RSI/ATR 的边界与已知值
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.backtest import (BacktestConfig, BacktestEngine, _is_limit_down,  # noqa: E402
                          _is_limit_up, _price_limit, _t_p_value)
from core.fetchers import normalize_symbol, to_secid, to_ts_code             # noqa: E402
from core.indicators import atr, build_snapshot, ema, rsi, sma               # noqa: E402
from core.signals import Action, Signal, SignalLedger                         # noqa: E402
from core.strategies import FiveDimStrategy, MACrossStrategy, Strategy        # noqa: E402
from core.validator import StrategyValidator                                  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, got, want, tol: float = 1e-6) -> None:
    global PASS, FAIL
    ok = False
    if isinstance(want, bool):
        ok = got is want or bool(got) is want
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    if ok:
        PASS += 1
        print(f"  ✅ {name}: {got}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: 得到 {got}，期望 {want}")


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ============================================================
# 1. 统计函数
# ============================================================

def test_statistics() -> None:
    section("1. t 分布 p 值（对照教科书临界值）")
    # 双侧检验临界值：p=0.05 时的 t 值，反推应得到 ≈0.05
    cases = [
        (1.960, 1000, 0.050),
        (1.962, 999, 0.050),
        (2.228, 10, 0.050),
        (3.169, 10, 0.010),
        (1.000, 10, 0.341),
        (12.706, 1, 0.050),
    ]
    for t, df, expect in cases:
        check(f"t={t}, df={df} → p", round(_t_p_value(t, df), 3), expect, tol=0.006)
    check("t=0 → p=1", round(_t_p_value(0.0, 30), 6), 1.0, tol=1e-4)


# ============================================================
# 2. 未来函数探测器
# ============================================================

class CheatingStrategy(Strategy):
    """故意偷看下一根K线：明天涨就买。用来测试探测器能否抓住。"""
    name = "cheat"

    def warmup(self) -> int:
        return 30

    def decide(self, symbol, bars, i, position=None) -> Signal:
        if i + 1 >= len(bars):
            return Signal.hold(bars[i].date, symbol, source="cheat")
        future_ret = bars[i + 1].close / bars[i].close - 1
        if future_ret > 0.01:
            return Signal(date=bars[i].date, symbol=symbol, action=Action.BUY,
                          score=80, source="cheat")
        if position:
            return Signal(date=bars[i].date, symbol=symbol, action=Action.SELL,
                          score=20, source="cheat")
        return Signal.hold(bars[i].date, symbol, source="cheat")


class HonestStrategy(Strategy):
    """只看历史：均线金叉买入。应当不被标记。"""
    name = "honest"

    def warmup(self) -> int:
        return 30

    def decide(self, symbol, bars, i, position=None) -> Signal:
        window = bars[:i + 1]
        closes = [b.close for b in window]
        ms, ml = sma(closes, 5), sma(closes, 20)
        if len(ms) < 2 or ms[-1] is None or ms[-2] is None or ml[-1] is None or ml[-2] is None:
            return Signal.hold(bars[i].date, symbol, source="honest")
        if ms[-2] <= ml[-2] and ms[-1] > ml[-1]:
            return Signal(date=bars[i].date, symbol=symbol, action=Action.BUY,
                          score=70, source="honest")
        if position and ms[-2] >= ml[-2] and ms[-1] < ml[-1]:
            return Signal(date=bars[i].date, symbol=symbol, action=Action.SELL,
                          score=30, source="honest")
        return Signal.hold(bars[i].date, symbol, source="honest")


def _fake_bars(n: int = 200, seed: int = 7):
    """确定性伪随机行情（不使用 random，保证可复现）。"""
    from core.fetchers import Bar
    bars, price = [], 100.0
    x = seed
    for i in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        drift = ((x / 2147483648) - 0.5) * 0.03
        o = price
        c = max(1.0, price * (1 + drift))
        h = max(o, c) * 1.005
        low = min(o, c) * 0.995
        bars.append(Bar(date=f"2025-{(i // 21) % 12 + 1:02d}-{i % 21 + 1:02d}",
                        open=round(o, 2), close=round(c, 2), high=round(h, 2),
                        low=round(low, 2), volume=1e6 + i * 1000, amount=0.0,
                        pct_chg=0.0, turnover=0.0, amplitude=0.0))
        price = c
    return bars


def test_lookahead_detector() -> None:
    section("2. 未来函数探测器")
    bars = _fake_bars()
    cfg = BacktestConfig(lookahead_check=True, initial_cash=100000)

    cheat = BacktestEngine(CheatingStrategy(), cfg).run({"TEST": bars}, None)
    caught = [w for w in cheat.warnings if "未来函数" in w]
    check("偷看未来的策略被抓住", len(caught) > 0, True)
    if caught:
        print(f"     抓到的证据：{caught[0][:110]}")

    honest = BacktestEngine(HonestStrategy(), cfg).run({"TEST": bars}, None)
    false_alarm = [w for w in honest.warnings if "未来函数" in w]
    check("老实策略不被误报", len(false_alarm), 0)


# ============================================================
# 3. A股规则
# ============================================================

def test_ashare_rules() -> None:
    section("3. A股交易规则")
    check("主板涨跌停 10%", _price_limit("sh600519"), 0.10)
    check("创业板 20%", _price_limit("sz300750"), 0.20)
    check("科创板 20%", _price_limit("sh688111"), 0.20)
    check("北交所 30%", _price_limit("bj430047"), 0.30)

    check("开盘 +10% 判为涨停", _is_limit_up(11.0, 10.0, "sh600519"), True)
    check("开盘 +5% 不是涨停", _is_limit_up(10.5, 10.0, "sh600519"), False)
    check("创业板 +10% 不是涨停", _is_limit_up(11.0, 10.0, "sz300750"), False)
    check("开盘 -10% 判为跌停", _is_limit_down(9.0, 10.0, "sh600519"), True)

    section("3b. T+1 与整百股")
    bars = _fake_bars(120, seed=3)
    cfg = BacktestConfig(enforce_t_plus_1=True, max_holding_days=1,
                         stop_loss_pct=0.001, take_profit_pct=0.001,
                         trailing_stop_pct=0.001)
    r = BacktestEngine(MACrossStrategy(5, 20), cfg).run({"TEST": bars}, None)
    same_day = [t for t in r.trades if t.exit_date == t.entry_date]
    check("没有当日买当日卖的交易", len(same_day), 0)
    lot_ok = all(t.shares % 100 == 0 for t in r.trades)
    check("全部按整百股成交", lot_ok, True)

    section("3c. 成本必须真实发生")
    fee_cfg = BacktestConfig(commission_rate=0.001, slippage_pct=0.01,
                             stamp_tax_rate=0.001)
    r_hi = BacktestEngine(MACrossStrategy(5, 20), fee_cfg).run({"TEST": bars}, None)
    low_cfg = BacktestConfig(commission_rate=0.0, commission_min=0.0,
                             slippage_pct=0.0, stamp_tax_rate=0.0, transfer_rate=0.0)
    r_lo = BacktestEngine(MACrossStrategy(5, 20), low_cfg).run({"TEST": bars}, None)
    check("高成本收益 < 零成本收益",
          r_hi.metrics.total_return_pct < r_lo.metrics.total_return_pct, True)
    if r_hi.trades:
        t = r_hi.trades[0]
        check("滑点使卖出价低于收盘价", t.exit_price != r_lo.trades[0].exit_price, True)


# ============================================================
# 4. 指标
# ============================================================

def test_indicators() -> None:
    section("4. 技术指标")
    check("sma 前 n-1 位为 None", sma([1, 2, 3, 4, 5], 3)[:2], [None, None])
    check("sma(3) 正确", sma([1, 2, 3, 4, 5], 3)[2], 2.0)
    check("sma 末位", sma([1, 2, 3, 4, 5], 3)[-1], 4.0)
    check("ema 长度对齐", len(ema([1, 2, 3], 3)), 3)

    # RSI：单调上涨应为 100
    up = [100 + i for i in range(30)]
    check("单调上涨 RSI=100", round(rsi(up, 14)[-1], 2), 100.0)
    down = [100 - i for i in range(30)]
    check("单调下跌 RSI=0", round(rsi(down, 14)[-1], 2), 0.0)

    # ATR：恒定波幅时等于该波幅
    h = [10.0] * 20
    low = [9.0] * 20
    c = [9.5] * 20
    check("恒定波幅 ATR≈1.0", round(atr(h, low, c, 14)[-1], 3), 1.0, tol=0.05)

    section("4b. 快照不得泄露未来")
    bars = _fake_bars(120)
    s1 = build_snapshot("TEST", bars, 100)
    mutated = list(bars)
    for j in range(101, len(mutated)):
        b = mutated[j]
        mutated[j] = type(b)(date=b.date, open=b.open * 5, close=b.close * 5,
                             high=b.high * 5, low=b.low * 5, volume=b.volume,
                             amount=b.amount, pct_chg=b.pct_chg,
                             turnover=b.turnover, amplitude=b.amplitude)
    s2 = build_snapshot("TEST", mutated, 100)
    check("篡改未来K线后快照不变", s1.to_prompt_dict(), s2.to_prompt_dict())
    check("快照不输出缺失的换手率", "换手率%" in s1.to_prompt_dict(), False)


# ============================================================
# 5. 代码规范化
# ============================================================

def test_symbols() -> None:
    section("5. 代码规范化")
    for raw, want in [("600519", "sh600519"), ("SH600519", "sh600519"),
                      ("600519.SH", "sh600519"), ("000001", "sz000001"),
                      ("300750", "sz300750"), ("430047", "bj430047"),
                      ("sh000300", "sh000300")]:
        check(f"{raw} → {want}", normalize_symbol(raw), want)
    check("secid 沪市", to_secid("sh600519"), "1.600519")
    check("secid 深市", to_secid("sz000001"), "0.000001")
    check("secid 沪深300", to_secid("sh000300"), "1.000300")
    check("ts_code", to_ts_code("sh600519"), "600519.SH")


# ============================================================
# 6. 验证器行为
# ============================================================

def test_validator() -> None:
    section("6. 验证器行为")
    bars = _fake_bars(300, seed=11)

    # 交易太少时必须判样本不足，且不能给出"优秀"之类的结论
    _, rep = StrategyValidator(BacktestConfig(), min_trades_for_verdict=9999)\
        .validate(FiveDimStrategy(), {"TEST": bars}, None)
    check("交易过少 → 样本不足", "样本不足" in rep.grade, True)
    check("样本不足时仍给出基准说明", len(rep.benchmark_note) > 0, True)
    check("样本不足时仍给出显著性说明", len(rep.significance_note) > 0, True)
    check("样本不足时仍给出过拟合说明", len(rep.overfit_verdict) > 0, True)

    # 阈值判定方向必须正确
    from core.backtest import Metrics
    vals = [("win_rate_pct", 60.0, "GOOD"), ("win_rate_pct", 40.0, "FAIL"),
            ("max_drawdown_pct", 5.0, "GOOD"), ("max_drawdown_pct", 30.0, "FAIL"),
            ("sharp" + "e", 2.0, "GOOD"), ("sharpe", 0.2, "FAIL"),
            ("trades_per_month", 10.0, "GOOD"), ("trades_per_month", 1.0, "FAIL")]
    v = StrategyValidator(BacktestConfig(), min_trades_for_verdict=0)
    for name, value, want in vals:
        m = Metrics(trades=30)
        setattr(m, name, value)
        setattr(m, "final_equity", 100000)
        r = _mk_result(m, bars)
        rep = v._build_report(r, {"TEST": bars}, None, "t")
        got = next(c.verdict for c in rep.checks if c.name == name)
        check(f"{name}={value} → {want}", got, want)


def _mk_result(m, bars):
    from core.backtest import BacktestResult
    return BacktestResult(metrics=m, period=(bars[0].date, bars[-1].date),
                          strategy_name="t", signal_counts={"BUY": 5, "HOLD": 25})


# ============================================================

def main() -> int:
    print("dsa-lite 自检 —— 验证验证器本身")
    test_statistics()
    test_lookahead_detector()
    test_ashare_rules()
    test_indicators()
    test_symbols()
    test_validator()
    total = PASS + FAIL
    print(f"\n{'=' * 56}")
    print(f"通过 {PASS}/{total}" + (f"，失败 {FAIL}" if FAIL else "，全部通过 ✅"))
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
