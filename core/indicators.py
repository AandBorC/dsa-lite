#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标与市场环境计算 —— 纯标准库实现
============================================================
不依赖 pandas/numpy，避免在 GitHub Actions 里为了几个均线
装一整套科学计算栈。所有函数输入 list[float]，输出同长度 list[float]，
前置不足的位置用 None 占位，方便判断「数据够不够」。

约定：所有指标计算只允许使用「截至当日」的数据。
回测里任何一次调用都必须传入 bars[:i+1] 切片，绝不传全量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 基础序列运算
# ============================================================

def sma(values: list[float], n: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    if n <= 0:
        return out
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n:
            s -= values[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(values: list[float], n: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    if not values or n <= 0:
        return out
    k = 2.0 / (n + 1)
    prev = None
    for i, v in enumerate(values):
        prev = v if prev is None else v * k + prev * (1 - k)
        if i >= n - 1:
            out[i] = prev
    return out


def rolling_std(values: list[float], n: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for i in range(n - 1, len(values)):
        w = values[i - n + 1:i + 1]
        m = sum(w) / n
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / n)
    return out


def rsi(closes: list[float], n: int = 14) -> list[Optional[float]]:
    """Wilder 平滑的 RSI，与通达信/同花顺口径接近。"""
    out: list[Optional[float]] = [None] * len(closes)
    if len(closes) < n + 1:
        return out
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def atr(highs: list[float], lows: list[float], closes: list[float],
        n: int = 14) -> list[Optional[float]]:
    trs = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[0] - lows[0])
        else:
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
    return ema(trs, n)


def pct_change(values: list[float], n: int = 1) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for i in range(n, len(values)):
        base = values[i - n]
        out[i] = None if base == 0 else (values[i] / base - 1) * 100
    return out


def drawdown_series(equity: list[float]) -> list[float]:
    """净值曲线 → 回撤序列（负数，%）"""
    out, peak = [], (equity[0] if equity else 0.0)
    for v in equity:
        peak = max(peak, v)
        out.append(0.0 if peak == 0 else (v / peak - 1) * 100)
    return out


# ============================================================
# 市场环境快照：喂给 LLM 的「事实层」
# ============================================================

@dataclass
class Snapshot:
    """
    某一交易日的市场环境快照。

    设计意图：LLM 的幻觉大多来自「没数据只好编」。
    解决办法不是加更多提示词，而是把可验证的事实算好、塞给它，
    并明确告诉它「只能基于以下字段判断」。
    """
    symbol: str
    date: str
    close: float
    pct_chg: float

    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    ma20_slope: Optional[float] = None      # MA20 近 5 日斜率 %
    rsi14: Optional[float] = None
    atr14: Optional[float] = None
    atr_pct: Optional[float] = None         # ATR / 收盘价 %
    vol_ratio: Optional[float] = None       # 量比：当日量 / 5日均量
    momentum_20d: Optional[float] = None    # 20 日涨幅 %
    momentum_60d: Optional[float] = None
    pos_in_60d: Optional[float] = None      # 收盘价在 60 日区间的位置 0-100
    volatility_20d: Optional[float] = None  # 20 日年化波动率 %
    turnover: Optional[float] = None
    ext: dict = field(default_factory=dict)

    def to_prompt_dict(self) -> dict:
        """
        裁剪成给 LLM 的紧凑字典，不丢精度也不塞垃圾。

        数据缺失的字段一律置 None 而不是填 0 —— 填 0 会骗模型：
        换手率 0% 对活跃个股是不可能的，"0" 实际含义是"这个源不提供"。
        把缺失伪装成数值，是回测里最隐蔽的一类错误信息。
        """
        out = {
            "代码": self.symbol,
            "日期": self.date,
            "收盘价": round(self.close, 2),
            "当日涨跌幅%": round(self.pct_chg, 2),
            "MA5": _r(self.ma5), "MA10": _r(self.ma10),
            "MA20": _r(self.ma20), "MA60": _r(self.ma60),
            "MA20近5日斜率%": _r(self.ma20_slope),
            "RSI14": _r(self.rsi14, 1),
            "ATR14": _r(self.atr14, 3),
            "ATR占价格比%": _r(self.atr_pct, 2),
            "量比": _r(self.vol_ratio, 2),
            "20日涨幅%": _r(self.momentum_20d, 1),
            "60日涨幅%": _r(self.momentum_60d, 1),
            "60日区间位置": _r(self.pos_in_60d, 1),
            "20日年化波动率%": _r(self.volatility_20d, 1),
            "换手率%": _r(self.turnover, 2),
        }
        if not out["换手率%"]:
            out.pop("换手率%")
        return out

    def render_text(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.to_prompt_dict().items()]
        return "\n".join(lines)


def _r(v, nd: int = 2):
    return None if v is None else round(float(v), nd)


def build_snapshot(symbol: str, bars: list, i: int) -> Snapshot:
    """
    用 bars[:i+1] 构造第 i 日的快照。

    注意：函数内部强制切片，调用方即使传了全量 bars 也不会泄露未来信息。
    """
    window = bars[:i + 1]
    closes = [b.close for b in window]
    highs = [b.high for b in window]
    lows = [b.low for b in window]
    vols = [b.volume for b in window]
    cur = window[-1]

    ma5, ma10, ma20, ma60 = sma(closes, 5), sma(closes, 10), sma(closes, 20), sma(closes, 60)
    ma20_s = ma20_slope = None
    if len(ma20) >= 6 and ma20[-1] is not None and ma20[-6] not in (None, 0):
        ma20_slope = (ma20[-1] / ma20[-6] - 1) * 100
        ma20_s = ma20[-1]

    a = atr(highs, lows, closes, 14)
    atr14 = a[-1] if a else None
    atr_pct = (atr14 / cur.close * 100) if (atr14 and cur.close) else None

    a5 = sma(vols, 5)
    prev5 = vols[-6:-1] if len(vols) >= 6 else []
    vol_ratio = (cur.volume / (sum(prev5) / len(prev5))) if prev5 and sum(prev5) > 0 else None

    mom20 = _pct_ago(closes, 20)
    mom60 = _pct_ago(closes, 60)

    pos60 = None
    if len(closes) >= 60:
        w = closes[-60:]
        lo, hi = min(w), max(w)
        pos60 = 0.0 if hi == lo else (cur.close - lo) / (hi - lo) * 100

    vol20 = None
    if len(closes) >= 21:
        rets = [(closes[k] / closes[k - 1] - 1) for k in range(len(closes) - 20, len(closes))]
        m = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))
        vol20 = sd * math.sqrt(252) * 100

    return Snapshot(
        symbol=symbol, date=cur.date, close=cur.close, pct_chg=cur.pct_chg,
        ma5=ma5[-1], ma10=ma10[-1], ma20=ma20[-1], ma60=ma60[-1],
        ma20_slope=ma20_slope, rsi14=(rsi(closes)[-1] if len(closes) > 15 else None),
        atr14=atr14, atr_pct=atr_pct, vol_ratio=vol_ratio,
        momentum_20d=mom20, momentum_60d=mom60, pos_in_60d=pos60,
        volatility_20d=vol20, turnover=cur.turnover,
    )


def _pct_ago(closes: list[float], n: int) -> Optional[float]:
    if len(closes) <= n or closes[-n - 1] == 0:
        return None
    return (closes[-1] / closes[-n - 1] - 1) * 100
