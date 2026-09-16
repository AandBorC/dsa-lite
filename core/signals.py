#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号层 —— LLM 结论与回测引擎之间的唯一接口
============================================================
这是「让 LLM 可回测」的关键设计：

LLM 说的是人话（"短期偏强但量能不足，建议观望"），
回测引擎只认机器话（BUY/SELL/HOLD + 价格 + 仓位）。

所以中间必须有这一层做翻译，而且翻译结果必须可复现、可落盘、可追责。
三个硬约束：

1. 信号只能由 Snapshot（截至当日的数据）生成 —— 从源头掐死未来函数
2. 信号必须落盘成台账 —— 否则无法事后复盘「模型当时到底说了什么」
3. 信号必须带 prompt_version —— 换提示词等于换策略，回测要能区分

台账同时解决了 LLM 回测的成本问题：
每天调一次 API 很贵，但结果缓存后，重跑回测是免费的。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("signals")

LEDGER_DIR = Path(__file__).resolve().parent.parent / "data" / "ledger"


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    AVOID = "AVOID"

    @classmethod
    def parse(cls, raw) -> "Action":
        if isinstance(raw, Action):
            return raw
        s = str(raw or "").strip().upper()
        alias = {
            "买入": "BUY", "买": "BUY", "加仓": "BUY", "看多": "BUY",
            "卖出": "SELL", "卖": "SELL", "减仓": "SELL", "清仓": "SELL", "看空": "SELL",
            "观望": "HOLD", "持有": "HOLD", "中性": "HOLD", "震荡": "HOLD",
            "回避": "AVOID", "不参与": "AVOID",
        }
        s = alias.get(s.strip(), s)
        try:
            return cls(s)
        except ValueError:
            return cls.HOLD


@dataclass
class Signal:
    """一条可执行、可回测的交易信号。"""
    date: str                      # 信号生成日（= 决策所依据的最后一根K线日期）
    symbol: str
    action: Action
    score: float = 50.0            # 0-100 多空强度
    confidence: float = 0.5        # 0-1 模型自评置信度

    entry: Optional[float] = None  # 建议入场价（None 表示市价）
    stop: Optional[float] = None   # 止损价
    target: Optional[float] = None # 目标价
    horizon_days: int = 10         # 预期持有周期（自然交易日）

    source: str = "rule"           # llm | rule | manual | ledger
    model: str = ""
    prompt_version: str = "v1"
    reason: str = ""
    meta: dict = field(default_factory=dict)

    # ---------- 派生属性，供风控与回测使用 ----------

    @property
    def risk_reward(self) -> Optional[float]:
        """盈亏比 = (目标-入场) / (入场-止损)。"""
        if None in (self.entry, self.stop, self.target):
            return None
        risk = self.entry - self.stop
        if risk <= 0:
            return None
        return (self.target - self.entry) / risk

    @property
    def is_actionable(self) -> bool:
        return self.action in (Action.BUY, Action.SELL)

    def uid(self) -> str:
        raw = f"{self.date}|{self.symbol}|{self.source}|{self.prompt_version}|{self.action.value}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def as_row(self) -> dict:
        row = asdict(self)
        row["action"] = self.action.value
        row["risk_reward"] = self.risk_reward
        row["meta"] = json.dumps(self.meta, ensure_ascii=False) if self.meta else ""
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Signal":
        meta = {}
        if row.get("meta"):
            try:
                meta = json.loads(row["meta"])
            except Exception:  # noqa: BLE001
                meta = {}
        return cls(
            date=row["date"], symbol=row["symbol"],
            action=Action.parse(row.get("action")),
            score=_f(row.get("score"), 50.0),
            confidence=_f(row.get("confidence"), 0.5),
            entry=_f(row.get("entry")), stop=_f(row.get("stop")),
            target=_f(row.get("target")),
            horizon_days=int(_f(row.get("horizon_days"), 10) or 10),
            source=row.get("source", "rule"), model=row.get("model", ""),
            prompt_version=row.get("prompt_version", "v1"),
            reason=row.get("reason", ""), meta=meta,
        )

    @classmethod
    def hold(cls, date: str, symbol: str, source: str = "rule", **kw) -> "Signal":
        return cls(date=date, symbol=symbol, action=Action.HOLD, source=source, **kw)


LEDGER_COLUMNS = [
    "date", "symbol", "action", "score", "confidence", "entry", "stop", "target",
    "horizon_days", "source", "model", "prompt_version", "reason", "risk_reward", "meta",
]


class SignalLedger:
    """
    信号台账（append-only CSV）。

    两个用途：
      1. 实盘留痕 —— 事后能精确复盘「那天模型说了什么、后来走势如何」
      2. 回测样本 —— 积累足够多条真实 LLM 信号后，直接回放而不必重调 API
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else LEDGER_DIR / "signals.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 写 ----------

    def append(self, signals: list[Signal]) -> int:
        """按 uid 去重后追加。返回实际新增条数。"""
        existing = self.known_uids()
        fresh = [s for s in signals if s.uid() not in existing]
        if not fresh:
            return 0
        new_file = not self.path.exists()
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            for s in fresh:
                w.writerow(s.as_row())
        log.info("台账新增 %d 条信号 → %s", len(fresh), self.path)
        return len(fresh)

    # ---------- 读 ----------

    def read(self, symbols: Optional[list[str]] = None,
             source: Optional[str] = None,
             prompt_version: Optional[str] = None) -> list[Signal]:
        if not self.path.exists():
            return []
        out: list[Signal] = []
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s = Signal.from_row(row)
                except Exception:  # noqa: BLE001
                    continue
                if symbols and s.symbol not in symbols:
                    continue
                if source and s.source != source:
                    continue
                if prompt_version and s.prompt_version != prompt_version:
                    continue
                out.append(s)
        out.sort(key=lambda x: (x.date, x.symbol))
        return out

    def known_uids(self) -> set[str]:
        return {s.uid() for s in self.read()}

    def stats(self) -> dict:
        rows = self.read()
        if not rows:
            return {"total": 0}
        by_action, by_source, by_symbol = {}, {}, {}
        for s in rows:
            by_action[s.action.value] = by_action.get(s.action.value, 0) + 1
            by_source[s.source] = by_source.get(s.source, 0) + 1
            by_symbol[s.symbol] = by_symbol.get(s.symbol, 0) + 1
        return {
            "total": len(rows),
            "date_range": [rows[0].date, rows[-1].date],
            "by_action": by_action, "by_source": by_source,
            "symbols": len(by_symbol),
            "actionable": sum(1 for s in rows if s.is_actionable),
        }


# ============================================================
# 信号缓存（LLM 输出按内容哈希缓存，让回测重跑免费）
# ============================================================

class SignalCache:
    """
    键 = prompt 内容的 md5。

    为什么不是 (symbol, date)？因为提示词模板一改，同一个日期也该重新问一遍。
    哈希里包含完整 prompt 文本，天然覆盖了模板版本、上下文数据、
    模型名这些变量 —— 改了任何一个，缓存自动失效。
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else LEDGER_DIR / "llm_cache.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._data = {}

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.md5(f"{model}||{prompt}".encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[dict]:
        return self._data.get(self.key(model, prompt))

    def put(self, model: str, prompt: str, value: dict) -> None:
        self._data[self.key(model, prompt)] = value
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
            self._dirty = False

    def __len__(self) -> int:
        return len(self._data)


def _f(v, default=None) -> Optional[float]:
    if v in (None, "", "None", "nan"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
