#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略层 —— 统一的 decide() 接口
============================================================
回归测试的核心矛盾：要验证 LLM 到底有没有 alpha，必须拿它跟基准比。
所以这里提供三类可互换的策略，接口完全一致：

    rule     规则策略（MA金叉、五维评分）—— 基准线，说明"不用LLM能拿到什么"
    llm      LLM策略（真调模型）—— 待验证对象
    ledger   台账回放（读历史信号）—— 零成本重跑，用于调参

三者共用 decide(symbol, bars, i, position) 签名，
回测引擎不需要知道背后是人是模型 —— 这是能对比的前提。

硬性契约：decide() 只能读取 bars[:i+1]。
引擎会在调试模式下用「扰动未来K线」的方式检测违规（见 backtest.py）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from .indicators import build_snapshot, sma, rsi
from .signals import Action, Signal

log = logging.getLogger("strategies")


class Strategy(ABC):
    """所有策略的基类。"""

    name: str = "base"
    source: str = "rule"

    @abstractmethod
    def decide(self, symbol: str, bars: list, i: int,
               position: Optional[dict] = None) -> Signal:
        """
        在第 i 根K线收盘后给出信号。

        bars  : 完整K线序列（实现方必须只使用 bars[:i+1]）
        i     : 当前索引
        position: 当前持仓 {'entry':float,'shares':int,'entry_date':str,
                            'high_water':float} 或 None
        """

    def warmup(self) -> int:
        """需要多少根历史K线才开始出信号。"""
        return 60


# ============================================================
# 规则策略 1：均线金叉/死叉
# ============================================================

class MACrossStrategy(Strategy):
    name = "ma_cross"

    def __init__(self, short: int = 5, long: int = 20, stop_atr: float = 2.0):
        self.short, self.long, self.stop_atr = short, long, stop_atr

    def warmup(self) -> int:
        return self.long + 10

    def decide(self, symbol: str, bars: list, i: int, position=None) -> Signal:
        snap = build_snapshot(symbol, bars, i)
        window = bars[:i + 1]
        closes = [b.close for b in window]
        ms, ml = sma(closes, self.short), sma(closes, self.long)

        if len(ms) < 2 or ms[-1] is None or ms[-2] is None or ml[-1] is None or ml[-2] is None:
            return Signal.hold(snap.date, symbol, source=self.source)

        golden = ms[-2] <= ml[-2] and ms[-1] > ml[-1]
        death = ms[-2] >= ml[-2] and ms[-1] < ml[-1]
        atr = snap.atr14 or snap.close * 0.02

        if position:  # 持仓中只考虑卖
            if death:
                return Signal(date=snap.date, symbol=symbol, action=Action.SELL,
                              score=30, confidence=0.6, entry=snap.close,
                              source=self.source, reason=f"MA{self.short}下穿MA{self.long}，死叉离场")
            return Signal.hold(snap.date, symbol, source=self.source)

        if golden:
            stop = snap.close - self.stop_atr * atr
            return Signal(
                date=snap.date, symbol=symbol, action=Action.BUY, score=70, confidence=0.6,
                entry=snap.close, stop=round(stop, 2),
                target=round(snap.close + 3 * self.stop_atr * atr, 2),
                horizon_days=20, source=self.source,
                reason=f"MA{self.short}上穿MA{self.long}，金叉入场")
        return Signal.hold(snap.date, symbol, source=self.source)


# ============================================================
# 规则策略 2：五维评分（趋势/动量/量能/位置/波动）
# ============================================================

class FiveDimStrategy(Strategy):
    """
    对应 a-share-master《选股层-五维评分》的可计算子集。

    只保留能用日线算出来的维度，基本面/资金面维度故意不做 ——
    宁可少做，也不要在回测里引入拿不到的历史数据（那是未来函数温床）。
    """

    name = "five_dim"

    def __init__(self, buy_threshold: float = 66.0, sell_threshold: float = 48.0,
                 stop_atr: float = 2.5, weights: Optional[dict] = None):
        # 阈值取 66/48 而非 70/35：实测 70/35 会让持仓只能靠止损止盈离场，
        # 20 个月只出 3 笔交易 —— 样本太少，回测结论不可用。
        # 卖出阈值贴近中性（48）才能让"评分转弱"真正触发换手。
        self.buy_th, self.sell_th = buy_threshold, sell_threshold
        self.stop_atr = stop_atr
        self.w = weights or {"trend": 0.30, "momentum": 0.25, "volume": 0.15,
                             "position": 0.15, "volatility": 0.15}

    def warmup(self) -> int:
        return 70

    # ---- 各维度打分（0-100） ----

    @staticmethod
    def _trend(s: dict) -> float:
        c, ma20, ma60 = s["close"], s["ma20"], s["ma60"]
        score = 50.0
        if ma20 and c > ma20:
            score += 15
        if ma60 and c > ma60:
            score += 15
        if ma20 and ma60 and ma20 > ma60:
            score += 10
        slope = s["ma20_slope"]
        if slope is not None:
            score += max(-15, min(15, slope * 3))
        return _clamp(score)

    @staticmethod
    def _momentum(s: dict) -> float:
        m20 = s["momentum_20d"]
        if m20 is None:
            return 50.0
        # 温和上涨最好，暴涨反而扣分（追高风险）
        if m20 <= -20:
            return 20.0
        if m20 <= 0:
            return 50 + m20 * 1.5
        if m20 <= 20:
            return 50 + m20 * 2.0
        return max(40.0, 90 - (m20 - 20) * 1.5)

    @staticmethod
    def _volume(s: dict) -> float:
        vr = s["vol_ratio"]
        if vr is None:
            return 50.0
        if vr >= 3:
            return 85.0
        if vr >= 1.5:
            return 70.0
        if vr >= 0.8:
            return 50.0
        return 35.0

    @staticmethod
    def _position(s: dict) -> float:
        p = s["pos_in_60d"]
        if p is None:
            return 50.0
        if p >= 95:
            return 40.0        # 贴近60日高点，追高扣分
        if p >= 60:
            return 80.0
        if p >= 30:
            return 60.0
        return 45.0

    @staticmethod
    def _volatility(s: dict) -> float:
        v = s["volatility_20d"]
        if v is None:
            return 50.0
        if v <= 20:
            return 75.0
        if v <= 35:
            return 60.0
        if v <= 50:
            return 45.0
        return 25.0

    def score(self, snap) -> tuple[float, dict]:
        s = {
            "close": snap.close, "ma20": snap.ma20, "ma60": snap.ma60,
            "ma20_slope": snap.ma20_slope, "momentum_20d": snap.momentum_20d,
            "vol_ratio": snap.vol_ratio, "pos_in_60d": snap.pos_in_60d,
            "volatility_20d": snap.volatility_20d,
        }
        parts = {k: fn(s) for k, fn in (
            ("trend", self._trend), ("momentum", self._momentum),
            ("volume", self._volume), ("position", self._position),
            ("volatility", self._volatility))}
        total = sum(parts[k] * self.w[k] for k in parts)
        return round(total, 1), {k: round(v, 1) for k, v in parts.items()}

    def decide(self, symbol: str, bars: list, i: int, position=None) -> Signal:
        snap = build_snapshot(symbol, bars, i)
        total, parts = self.score(snap)
        atr = snap.atr14 or snap.close * 0.02

        if position:
            if total <= self.sell_th:
                return Signal(date=snap.date, symbol=symbol, action=Action.SELL,
                              score=total, confidence=0.6, entry=snap.close,
                              source=self.source, reason=f"五维评分转弱({total})，触发退出",
                              meta={"dims": parts})
            return Signal(date=snap.date, symbol=symbol, action=Action.HOLD,
                          score=total, source=self.source, meta={"dims": parts})

        if total >= self.buy_th:
            return Signal(
                date=snap.date, symbol=symbol, action=Action.BUY, score=total,
                confidence=min(0.9, (total - self.buy_th) / 30 + 0.5),
                entry=snap.close, stop=round(snap.close - self.stop_atr * atr, 2),
                target=round(snap.close + 3 * self.stop_atr * atr, 2),
                horizon_days=20, source=self.source,
                reason=f"五维评分达标({total})", meta={"dims": parts})
        return Signal(date=snap.date, symbol=symbol, action=Action.HOLD,
                      score=total, source=self.source, meta={"dims": parts})


# ============================================================
# 规则策略 3：RSI 均值回归（与趋势策略负相关，用于组合对照）
# ============================================================

class RSIReversionStrategy(Strategy):
    name = "rsi_reversion"

    def __init__(self, oversold: float = 28.0, overbought: float = 72.0,
                 trend_filter: bool = True):
        self.oversold, self.overbought = oversold, overbought
        self.trend_filter = trend_filter

    def warmup(self) -> int:
        return 60

    def decide(self, symbol: str, bars: list, i: int, position=None) -> Signal:
        snap = build_snapshot(symbol, bars, i)
        r = snap.rsi14
        if r is None:
            return Signal.hold(snap.date, symbol, source=self.source)
        atr = snap.atr14 or snap.close * 0.02

        if position:
            if r >= self.overbought:
                return Signal(date=snap.date, symbol=symbol, action=Action.SELL,
                              score=100 - r, entry=snap.close, source=self.source,
                              reason=f"RSI={r:.1f} 超买，均值回归离场")
            return Signal.hold(snap.date, symbol, source=self.source)

        if r <= self.oversold:
            # 可选趋势过滤：只在上行趋势里抄底，避免下跌中继
            if self.trend_filter and snap.ma60 and snap.close < snap.ma60 * 0.9:
                return Signal(date=snap.date, symbol=symbol, action=Action.AVOID,
                              score=r, source=self.source,
                              reason=f"RSI={r:.1f} 超卖但深度跌破MA60，回避")
            return Signal(date=snap.date, symbol=symbol, action=Action.BUY,
                          score=max(0, 100 - r), confidence=0.55, entry=snap.close,
                          stop=round(snap.close - 2 * atr, 2),
                          target=round(snap.close + 3 * atr, 2), horizon_days=10,
                          source=self.source, reason=f"RSI={r:.1f} 超卖，均值回归买入")
        return Signal.hold(snap.date, symbol, source=self.source)


# ============================================================
# LLM 策略：把模型包成策略
# ============================================================

class LLMStrategy(Strategy):
    """
    用 LLM 做决策的策略适配器。

    四个工程要点（也是 LLM 回测最容易翻车的地方）：
      1. 只把 Snapshot 交给模型，不喂原始K线 —— 控制 token，也逼模型看指标
      2. 输出强约束为 JSON —— 解析失败即降级为 HOLD，绝不猜
      3. 按 prompt 哈希缓存 —— 否则每次回测都在烧钱，也永远无法复现
      4. 记忆按当根K线的日期门控 —— 回测到 2025-06-20 那一根时，
         只看得见 2025-06-20 时点已知的教训

    第 4 点是这里最容易做错的地方：记忆是跨时间的，如果按"今天"取，
    整段回测就会拿到全部 hindsight，等于开卷考试，而且**表面上完全看不出来**。

    回测**只读**记忆，绝不写。写了就会污染记忆库，之后再拿这个库做回测
    就变成自我循环 —— 模型的判断被自己的判断强化，看起来越来越准。
    """

    name = "llm"
    source = "llm"

    def __init__(self, analyzer, cache=None, prompt_version: str = "v1",
                 min_score_to_buy: float = 65.0, max_score_to_sell: float = 35.0,
                 memory=None):
        self.analyzer = analyzer
        self.cache = cache
        self.prompt_version = prompt_version
        self.buy_th, self.sell_th = min_score_to_buy, max_score_to_sell
        self.memory = memory
        # 记忆使用台账：没有它就无法回答"这次回测到底有没有真的注入记忆"
        # —— 一个静默失效的记忆注入，和没注入的区别，只有在对比里才看得出来。
        self.stats = {"memory_injected": 0, "lessons_shown": 0,
                      "blocked": 0, "pending": 0, "empty": 0}

    def warmup(self) -> int:
        return 70

    def decide(self, symbol: str, bars: list, i: int, position=None) -> Signal:
        snap = build_snapshot(symbol, bars, i)

        # ---- 记忆按本根K线日期门控 ----
        block, lesson_ids = "", []
        if self.memory is not None:
            ctx = self.memory.get_past_context(symbol, snap.date)
            block = ctx.render()
            lesson_ids = ctx.visible_lesson_ids()
            self.stats["memory_injected"] += 1
            self.stats["lessons_shown"] += len(ctx.lessons)
            self.stats["blocked"] += ctx.blocked
            self.stats["pending"] += ctx.pending
            self.stats["empty"] += 1 if not ctx.lessons else 0

        actx = {"position": position, "holding": bool(position),
                "memory_block": block}
        try:
            sig = self.analyzer.analyze(snap, **actx)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] %s LLM 分析失败，降级为 HOLD: %s", symbol, snap.date, exc)
            return Signal.hold(snap.date, symbol, source=self.source,
                               model=getattr(self.analyzer, "model", ""),
                               prompt_version=self.prompt_version,
                               reason=f"LLM调用失败: {type(exc).__name__}")
        sig.prompt_version = self.prompt_version
        sig.meta.setdefault("snapshot", snap.to_prompt_dict())
        if self.memory is not None:
            sig.meta["memory_lessons"] = lesson_ids

        # 分数闸门：模型说买但分数不够 → 降为观望。防止模型"嘴上一套分数一套"。
        if sig.action == Action.BUY and sig.score < self.buy_th:
            sig.action = Action.HOLD
            sig.reason += f"（分数{sig.score}低于买入阈值{self.buy_th}，降级观望）"
        if sig.action == Action.SELL and not position:
            sig.action = Action.AVOID
        return sig


# ============================================================
# 台账回放策略：零成本重跑历史 LLM 信号
# ============================================================

class LedgerStrategy(Strategy):
    """
    直接回放已落盘的信号，不重新调模型。

    用途：调回测参数（止损倍数、仓位、持有上限）时反复重跑，
    因为真正贵的 LLM 调用已经固化在台账里了。
    """

    name = "ledger"
    source = "ledger"

    def __init__(self, signals: list[Signal]):
        self.by_key: dict[tuple[str, str], Signal] = {}
        for s in signals:
            self.by_key[(s.date, s.symbol)] = s

    def warmup(self) -> int:
        return 0

    def decide(self, symbol: str, bars: list, i: int, position=None) -> Signal:
        date = bars[i].date
        s = self.by_key.get((date, symbol))
        if s is None:
            return Signal.hold(date, symbol, source=self.source)
        if s.action == Action.SELL and not position:
            return Signal.hold(date, symbol, source=self.source, score=s.score)
        return s


# ============================================================
# 工具
# ============================================================

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def build_strategy(kind: str, **kw) -> Strategy:
    """工厂：从字符串构造策略，供 config.yaml 与 CLI 使用。"""
    k = (kind or "").lower()
    if k in ("ma_cross", "ma", "金叉"):
        return MACrossStrategy(**{a: kw[a] for a in ("short", "long", "stop_atr") if a in kw})
    if k in ("five_dim", "five", "五维"):
        return FiveDimStrategy(**{a: kw[a] for a in
                                  ("buy_threshold", "sell_threshold", "stop_atr", "weights")
                                  if a in kw})
    if k in ("rsi", "rsi_reversion", "均值回归"):
        return RSIReversionStrategy(**{a: kw[a] for a in
                                       ("oversold", "overbought", "trend_filter") if a in kw})
    raise ValueError(f"未知策略: {kind}（可选 ma_cross / five_dim / rsi_reversion）")
