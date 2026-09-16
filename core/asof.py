#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时点门控（point-in-time guard）—— 让每次运行只看得见「那个时点能看见的东西」
============================================================

回测里的未来函数有两条入口，之前只堵了一条：

    策略层：decide() 偷看 bars[i+1]
        → 引擎用「扰动未来K线」反向验证（backtest.py::_lookahead_scan）

    数据层：进来的数据本身就带着 as_of 之后的信息
        → 本模块负责。策略代码看着完全正确，但结论已经被污染了。

数据层泄漏的四个真实来源（本项目里都存在，前三个是「已经有代码可以触发」的）：

1. **数据源按「今天」返回全量**。回测某一历史区间时，尾巴上多出 as_of
   之后的 K 线。索引约束（bars[:i+1]）能挡住策略，但日历、估值、
   强制平仓、缓存回退这些路径不经过索引约束。

2. **缓存回退路径不切片**。FetcherChain 在全部数据源失败时会回退到过期
   缓存 —— 那条路径原本把整段缓存原样返回，越界部分就是未来。

3. **前复权基准日取了「最新」因子**。qfq(t) = raw(t) × adj(t) / adj(最新)，
   分母里的「最新」是**今天**。如果 as_of 落在某个分红除权日之前，那个
   除权事件在 as_of 当天还没发生，却被用来缩放历史价格 —— 这是把未来的
   除权信息注入了历史。正确做法是分母取 adj(<=as_of 的最后一个)。

4. **将来接入新闻/公告/财报时**，一条发布时间晚于 as_of 的记录。
   本项目现在没有接这类数据源，但门控先立好 —— 等接的时候再用，
   代价是零；等出了 bug 再补，代价是一次白跑的回测和一堆错误结论。

设计参考 TradingAgents 的 dataflows/date_window.py，并补上 A 股特有的
第 3 条（它默认走 yfinance，auto_adjust 已经处理；tushare 不处理）。

三条原则
--------

    出口过滤    不信任上游。所有数据在离开数据层时统一裁一次，
                无论它是从网络来的、从缓存来的、还是从过期缓存来的。
    请求不发    请求区间本身钉在 as_of 内。未来数据连请求都不发出去 ——
                既从源头断了泄漏，也顺手省下数据源配额。
    丢弃有声    裁掉的每一根 K 线都记进 Audit，绝不静默。
                静默的截断比不截断更危险：你以为有门控，其实门是虚掩的。

用法：
    from core.asof import Audit, clip_bars, clip_range, withheld_notice
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger("asof")


# ============================================================
# 日期归一化与窗口判定
# ============================================================

def today() -> str:
    """今天的日期（本地时区，YYYY-MM-DD）。"""
    return datetime.now().strftime("%Y-%m-%d")


def norm(value: Any) -> str:
    """
    把各种日期写法归一成 YYYY-MM-DD。

    能吃：datetime/date 对象、'2025-06-20'、'20250620'、'2025/06/20'、Bar。
    吃不下就抛 ValueError —— 宁可报错，也不要把一个无法判定的日期
    当成「今天」，那等于把门控失效伪装成通过。
    """
    if value is None:
        raise ValueError("日期为空")
    if not isinstance(value, str) and hasattr(value, "date"):
        value = getattr(value, "date")
    s = str(value).strip()
    digits = s.replace("-", "").replace("/", "").replace(".", "")
    if len(digits) >= 8 and digits[:8].isdigit():
        d = digits[:8]
        if "1900" <= d[:4] <= "2999" and "01" <= d[4:6] <= "12" and "01" <= d[6:8] <= "31":
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    raise ValueError(f"无法识别日期: {value!r}")


def safe_norm(value: Any) -> Optional[str]:
    """norm 的不抛错版本，失败返回 None（用于「无时间戳」这类判定）。"""
    try:
        return norm(value)
    except (ValueError, TypeError):
        return None


def is_historical(as_of: Any, ref: Any = None) -> bool:
    """
    as_of 是否属于「过去」。

    这个判断决定无时间戳内容的去留：
      历史运行（回测）→ 丢弃，因为你无法证明它当时可见
      实时运行        → 保留，它就是当下的
    """
    a = norm(as_of)
    r = norm(ref) if ref is not None else today()
    return a < r


def in_window(pub: Any, start: Any = None, end: Any = None) -> bool:
    """
    一条带时间戳的记录是否落在 [start, end] 内（含 end 当天）。

    日频数据里「含 end 当天」等价于半开区间 [start, end+1天) ——
    一条恰好打在 end 次日 00:00 的记录不会被算进来。
    这个边界是踩出来的：不加它，收盘后发布的公告会溜进当天的决策上下文。
    """
    p = safe_norm(pub)
    if p is None:
        return False
    if start is not None and p < norm(start):
        return False
    if end is not None and p > norm(end):
        return False
    return True


# ============================================================
# 区间与 K 线裁剪
# ============================================================

def clip_range(start: Any, end: Any, as_of: Any) -> tuple[str, str]:
    """
    把请求区间钉在 as_of 内。用于「未来数据连请求都不发」。

    三种情形：
        end <= as_of          → 原样
        start <= as_of < end  → 终点压到 as_of
        as_of < start         → 整段都在未来，压成 (as_of, as_of)：
                                取不到数据是正确结果，不要悄悄放行
    """
    if as_of is None:
        return norm(start), norm(end)
    a = norm(as_of)
    s, e = norm(start), norm(end)
    if e <= a:
        return s, e
    if s > a:
        log.warning("请求区间 %s~%s 整体落在 as_of=%s 之后，压缩为空区间", s, e, a)
        return a, a
    return s, a


def clip_bars(bars: Sequence, as_of: Any, label: str = "",
              audit: Optional["Audit"] = None) -> list:
    """
    裁掉 as_of 之后的 K 线。数据层的统一出口，所有来源都要过这一关。

    按值过滤而不是按尾部切片 —— 停牌、多源合并、乱序都可能让
    越界的 bar 出现在中间，而不是整齐地待在末尾。
    """
    if as_of is None or not bars:
        return list(bars)
    a = norm(as_of)
    if bars[-1].date <= a and bars[0].date <= a:
        # 常见情形：整段都在窗口内，快速返回（不复制）
        return list(bars)
    kept = [b for b in bars if b.date <= a]
    if audit is not None and len(kept) != len(bars):
        audit.record_clip(label or "bars", len(bars), len(kept))
    if not kept:
        log.warning("%s：全部 %d 根 K 线都晚于 as_of=%s", label or "bars", len(bars), a)
    return kept


def filter_dated(rows: Iterable, as_of: Any, date_key: str = "date",
                 keep_undated: Optional[bool] = None,
                 audit: Optional["Audit"] = None) -> tuple[list, list]:
    """
    按发布时间过滤带时间戳的记录（新闻/公告/财报），返回 (保留, 丢弃)。

    keep_undated 默认自动：历史运行丢弃、实时运行保留。
    这不是洁癖 —— 一条没有时间戳的新闻放进回测，等于开卷考试。
    """
    rows = list(rows)
    if as_of is None:
        return rows, []
    if keep_undated is None:
        keep_undated = not is_historical(as_of)
    a = norm(as_of)
    kept: list = []
    dropped: list = []
    for r in rows:
        raw = r.get(date_key) if isinstance(r, dict) else getattr(r, date_key, None)
        d = safe_norm(raw)
        if d is None:
            if keep_undated:
                kept.append(r)
            else:
                dropped.append(r)
                if audit is not None:
                    audit.dropped_undated += 1
        elif d <= a:
            kept.append(r)
        else:
            dropped.append(r)
            if audit is not None:
                audit.dropped_dated += 1
    return kept, dropped


def withheld_notice(label: str, as_of: Any, reason: str = "") -> str:
    """
    数据被扣留时，别留一片空白 —— 空白会被模型读成「这里没有信号」，
    然后它就开始编。要明确告诉它：缺数据是因为时点，不是因为你没看到。

    这个函数是给「接入新闻/基本面」那天准备的，现在就能用。
    """
    a = norm(as_of)
    why = reason or "该数据源只提供当前值，没有历史版本"
    return (
        f"# {label}（时点：{a}）\n\n"
        f"{a} 时点下该数据不可用，已按「时点纪律」扣留。原因：{why}。\n"
        f"扣留的是 {a} 之后才发布的版本（今天是 {today()}）。\n"
        f"请把这段空白理解为「当时确实不知道」，不要读成看空/看多信号，"
        f"也不要凭训练数据里的记忆补全。"
    )


# ============================================================
# 审计台账
# ============================================================

@dataclass
class Audit:
    """
    记录一次运行里所有被裁掉、被丢弃的东西。

    存在的意义：把「静默截断」变成「有据可查」。
    一次回测如果裁掉了 300 根 K 线而没人知道，那这次回测的结论
    就是在另一个数据区间上得到的 —— 必须让人看见。
    """

    as_of: Optional[str] = None
    clipped: dict[str, int] = field(default_factory=dict)
    dropped_dated: int = 0
    dropped_undated: int = 0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.as_of is not None:
            self.as_of = norm(self.as_of)

    @property
    def dirty(self) -> bool:
        return bool(self.clipped or self.dropped_dated or self.dropped_undated)

    def record_clip(self, label: str, before: int, after: int) -> None:
        n = before - after
        if n <= 0:
            return
        self.clipped[label] = self.clipped.get(label, 0) + n
        self.note(f"{label}: 裁掉 {n} 根，保留 {after}/{before}")

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        log.warning("时点门控 | %s", msg)

    def summary(self) -> list[str]:
        """人类可读的审计结论，供报告与 doctor 使用。"""
        if self.as_of is None:
            return ["门控未启用（as_of 未指定，视为实时运行）"]
        head = f"as_of = {self.as_of}（{'历史运行' if is_historical(self.as_of) else '实时运行'}）"
        if not self.dirty:
            return [head, "无数据越界：所有数据都在该时点可见"]
        out = [head, f"裁掉 {sum(self.clipped.values())} 根 K 线"
                      f"（{len(self.clipped)} 个来源）；"
                      f"丢弃带时间戳记录 {self.dropped_dated} 条、"
                      f"无时间戳 {self.dropped_undated} 条"]
        out.extend(f"  · {n}" for n in self.notes)
        return out
