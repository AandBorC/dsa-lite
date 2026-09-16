#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反思记忆（reflection memory）—— 让每天的分析记得住自己错在哪
============================================================

没有这个模块时，每次 analyze 都是失忆的：它不知道自己上周说过什么、
说对没说过、有没有在同一个坑里连摔三次。这不是"模型不够聪明"，
是**工程上没给它记忆**。

设计参考 TradingAgents 的 reflection 机制，但改了一处最要命的地方 ——
它把 LLM 写的反思直接喂回下一轮，那份反思是"用未来信息生产出来的"，
喂回去就是开卷考试。所以：

    ⚠️ 教训本身就是未来函数

一条"事后复盘写下的教训"，天然带着决策之后的信息。这不是要回避的问题，
是要**给它盖一个诚实的日期戳**：

    learned_date = 决策日后第 horizon 个交易日
                   —— 这是"这个结果最早什么时候能知道"，不是"我什么时候写的"

于是注入时只要一个判据：`learned_date <= as_of` 才可见。
哪怕你今天才写下这条教训，只要它描述的结果窗口整段落在 as_of 之前，
它在那个时点就是合格的已知信息。反之，昨天写的教训若讲的是下个月的事，
它也进不来。

四条纪律
--------

    不进早结算     horizon 没走完就是 open。不许用"已经走了 3 天"的数据
                   去给一个 10 天的判断打分 —— 那是自己骗自己。

    日期戳诚实     lesson 的可追溯日期是 learned_date，不是写入日期。
                   这样"今天写的教训"也能安全地用于三个月前的回测。

    未结算不可用   open 状态的判断**不得**作为"你错了"出现在上下文里。
                   拿未结算的判断当反馈，等于让模型对着噪声自我修正。

    扣留要发声     正在验证期的判断必须明说"结果当时还不知道"，
                   不能留白。留白会被读成"这里没有教训"，然后它就开始编。

两类记录
--------

    decisions.jsonl   每次 analyze 写下一条。append-only，永不回改。
    lessons.jsonl     每条判断"结算"时写下一条。append-only。
                      「resolved 标签」= 该 decision 在 lessons 里有对应记录。

状态是 join 出来的，不是存在字段里的 —— 因为 append-only 的价值就在于
"当时写下什么就是什么"，事后回改状态会让台账失去证据资格。

用法：
    from core.memory import MemoryStore, get_past_context

    store = MemoryStore()
    ctx = get_past_context("sz300750", as_of="2025-06-20")   # 只看得见该时点已知的
    print(ctx.render())                                      # 喂进提示词的那段文本
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from . import asof

log = logging.getLogger("memory")

MEMORY_DIR = Path(__file__).resolve().parent.parent / "data" / "memory"


def default_root() -> Path:
    """
    记忆库位置。`DSA_MEMORY_DIR` 可覆盖 —— CI 与冒烟测试必须能把数据
    导到临时目录，否则测试跑一次就往仓库里塞一堆假判断，
    而假记忆会被真的注入到后续分析里（比假信号危险得多）。
    """
    env = (os.environ.get("DSA_MEMORY_DIR") or "").strip()
    return Path(env) if env else MEMORY_DIR

# 判定「结果成立/不成立」的中性带。±2% 以内视为噪声 ——
# 这个数是刻意的：A股日波动 1-2% 是常态，拿 ±2% 内的结果谈对错，
# 谈的是运气不是判断力。
FLAT_BAND_PCT = 2.0

# HOLD/AVOID 的判定带宽。观望没有成本，所以门槛要比持仓严格得多 ——
# 涨 2% 就说"你错过机会"是事后诸葛，涨 5% 才算真有信息。
AVOID_BAND_PCT = 5.0

# 长期停牌判定：窗口内相邻K线日历间隔超过这个天数，判定为异常，
# 该条判断标记 void（结算无效）而不是硬算 —— 停牌期间的价格不连续，
# 算出来的收益率没有意义，还会污染统计。
MAX_GAP_DAYS = 20

# 要说"这是规律"至少需要几条样本。与 validator.py 的 min_trades 同源逻辑：
# 2 次里错 2 次不叫规律，叫样本太小。
MIN_SAMPLE_FOR_PATTERN = 5


# ============================================================
# 记录
# ============================================================

@dataclass
class DecisionRecord:
    """
    一条判断的下单时刻留痕。

    和 signals.Signal 的区别：Signal 是"要做什么"，DecisionRecord 是
    "当时凭什么这么做、当时看得见什么"。后者才是复盘时真正需要的东西 ——
    只看 Signal，事后永远分不清「模型忽略了教训」和「模型根本没收到教训」。
    """

    decision_id: str
    symbol: str
    trade_date: str            # 决策所依据的最后一根K线日期（= 当时的 as_of）
    action: str
    score: float = 50.0
    confidence: float = 0.5
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    horizon_days: int = 10
    source: str = "llm"
    model: str = ""
    prompt_version: str = ""
    reason: str = ""

    facts_digest: str = ""              # 快照哈希：能识别「同样的依据、不同的结论」
    facts_gist: dict = field(default_factory=dict)
    visible_lessons: list = field(default_factory=list)   # 当时可见的教训 id
    decided_on: str = ""                # 实际写下这条记录的日期（审计用）

    @property
    def context_fingerprint(self) -> str:
        """当时上下文里有哪些教训 —— 用于区分「没学会」和「没看到」。"""
        raw = ",".join(sorted(self.visible_lessons))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


@dataclass
class LessonRecord:
    """
    一条判断的结算结果。**教训的日期戳是 learned_date，不是 created_on。**

    created_on 只有审计意义：它告诉我们"这条教训是今天才写下的"。
    真正决定它能不能被注入的，是 learned_date。
    """

    lesson_id: str
    decision_id: str
    symbol: str
    action: str = ""            # 当时的方向，复盘表格要用，不能再回查（决策可能被归档）
    decision_date: str = ""
    learned_date: str = ""      # ⭐ 结果最早可知的日期 = 决策日后第 horizon 个交易日
    horizon_days: int = 10
    bars_used: int = 0

    ret_pct: float = 0.0        # 决策日收盘 → 第 horizon 日收盘
    agent_ret_pct: float = 0.0  # 按方向调整后：BUY=ret，SELL=-ret，HOLD/AVOID=0
    mfe_pct: float = 0.0        # 期间最大浮盈
    mae_pct: float = 0.0        # 期间最大浮亏
    stop_hit: bool = False
    target_hit: bool = False
    stop_first: Optional[bool] = None

    verdict: str = "flat"       # hit | miss | flat | void
    error_type: str = "flat"
    lesson: str = ""
    prompt_version: str = ""
    created_on: str = ""


@dataclass
class MemoryContext:
    """
    某个 (symbol, as_of) 下**可以看见**的记忆。

    blocked / pending 不是装饰字段 —— 它们让「扣留」这件事有了输出位置。
    没有它们，被门控挡掉的历史就变成一片空白，而空白会被读成"没有教训"。
    """

    as_of: str
    symbol: str
    decisions: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    pending: int = 0            # 尚未到验证期的判断数
    blocked: int = 0            # 已结算、但结果在 as_of 时点还不可知的教训数
    blocked_until: list = field(default_factory=list)
    patterns: list = field(default_factory=list)   # [(error_type, count, hit, avg_agent_ret)]
    has_store: bool = True

    @property
    def empty(self) -> bool:
        return not self.decisions and not self.lessons and not self.blocked and not self.pending

    def visible_lesson_ids(self) -> list:
        return [l.lesson_id for l in self.lessons]

    def render(self) -> str:
        """渲染成喂进提示词的那段 markdown。绝不返回空字符串（空白会撒谎）。"""
        L: list = []
        A = L.append
        A(f"# 历史判断回顾（时点：{self.as_of}）")
        A("")
        if not self.has_store:
            A("（记忆库尚未建立。这是首次运行，你没有任何历史判断可参考。）")
            return "\n".join(L)
        if self.empty:
            A(f"你在 {self.symbol} 上还没有任何已结算的历史判断"
              "（首次运行，或全部判断仍在验证期内）。")
            A("不要假设下面应该有内容 —— 这段空白是真实的空白，不是被裁掉的。")
            return "\n".join(L)

        # 开场白必须与事实相符：只有待验证判断时不能说"及已结算结果" ——
        # 这段文本是模型直接读的，一句不准确的铺垫会误导它对全文的解读。
        if self.lessons:
            A(f"以下是**你自己**过去对 {self.symbol} 的判断及已结算结果。"
              "用途只有一个：避免重复同一个错误。")
        else:
            A(f"以下是**你自己**过去对 {self.symbol} 的判断，目前**尚无已结算结果**。")
        A("")

        if self.lessons:
            A("## 已结算（可以当作教训）")
            A("")
            A("| 决策日 | 方向 | 结果 | 区间收益 | 教训 |")
            A("|--------|------|------|---------|------|")
            flag = {"hit": "✅ 成立", "miss": "❌ 未成立", "flat": "— 无信息",
                    "void": "⚠️ 结算无效"}
            for l in self.lessons:
                A(f"| {l.decision_date} | {_dir_cn(l)} | {flag.get(l.verdict, l.verdict)} | "
                  f"{l.ret_pct:+.1f}% | {l.lesson} |")
            A("")

        if self.patterns:
            A(f"## 反复出现的错误（样本 ≥{MIN_SAMPLE_FOR_PATTERN} 才列出）")
            A("")
            for name, cnt, hit, avg in self.patterns:
                A(f"- `{name}`（{ERROR_LABEL.get(name, name)}）：{cnt} 次中成立 {hit} 次，"
                  f"按方向计平均影响 {avg:+.1f}%")
            A("")

        if self.blocked or self.pending:
            A("## 仍在验证中（结果在该时点还不知道）")
            A("")
            if self.blocked_until:
                A(f"有 {self.blocked} 条已可结算的判断，但其结果窗口延伸到 "
                  f"{max(self.blocked_until)}，在 {self.as_of} 时点尚未走完，已按"
                  f"「时点纪律」扣留。")
            if self.pending:
                A(f"另有 {self.pending} 条判断仍在验证期内。")
            A("请把这段理解为「当时确实还不知道」，**不要当成失误**，"
              "也不要凭训练数据里的记忆补全结果。")
            A("")

        if self.lessons:
            A("## 使用纪律")
            A("")
            A("- 上面的教训是参考，不是命令。如果你认为本次情况确实不同，"
              "可以推翻它，但必须在外层 JSON 的 reason 里写明理由。")
            A("- 不要把历史判断当成对本次走势的预知 —— 它只说明过去的模式，"
              "不说明这一次。")
        else:
            A("以上判断**都还没有结果**。不要假设它们对或错 —— "
              "你只能基于本次指标做判断。")
        return "\n".join(L)


# ============================================================
# 错误归因
# ============================================================

ERROR_LABEL = {
    "chased_high": "高位追进",
    "false_breakout": "入场即套",
    "stop_shaken": "止损被扫后行情回来",
    "wrong_direction": "方向看错",
    "right_call": "判断成立",
    "sold_too_early": "卖飞",
    "missed_opportunity": "过度保守",
    "correct_avoid": "回避正确",
    "flat": "落在噪声区间",
    "void": "结算无效",
}


@dataclass
class Outcome:
    """一条判断的前瞻窗口统计。"""
    bars_used: int = 0
    ret_pct: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    agent_ret_pct: float = 0.0   # 按方向调整后：BUY=ret，SELL/HOLD/AVOID=-ret
    stop_hit: bool = False
    target_hit: bool = False
    stop_first: Optional[bool] = None
    gap_days: int = 0


def _dir_cn(l: LessonRecord) -> str:
    return {"BUY": "买入", "SELL": "卖出", "HOLD": "观望", "AVOID": "回避"}.get(
        getattr(l, "action", "") or "", "—")


def _norm_action(a: Any) -> str:
    s = str(getattr(a, "value", a) or "").strip().upper()
    alias = {"买入": "BUY", "卖出": "SELL", "观望": "HOLD", "回避": "AVOID"}
    return alias.get(s, s)


def _outcome(d: DecisionRecord, bars: list, idx: int) -> Optional[Outcome]:
    """
    计算决策日之后 horizon 个交易日的表现。

    返回 None 表示**现在还不能算** —— 数据不够长，判断继续留在 open。
    这是「不进早结算」的落地点：宁可不下结论，也不用半截数据下结论。
    """
    need = d.horizon_days
    if idx + need >= len(bars):
        return None                     # 前瞻窗口还没走完
    window = bars[idx + 1: idx + need + 1]
    if not window:
        return None

    base = bars[idx].close
    if not base:
        return None

    # 停牌/数据缺口检测
    dates = [bars[idx].date] + [b.date for b in window]
    gap = 0
    for a, b in zip(dates, dates[1:]):
        try:
            gap = max(gap, (datetime.strptime(b, "%Y-%m-%d")
                            - datetime.strptime(a, "%Y-%m-%d")).days)
        except ValueError:
            continue

    o = Outcome(
        bars_used=len(window),
        ret_pct=(window[-1].close / base - 1) * 100,
        mfe_pct=(max(b.high for b in window) / base - 1) * 100,
        mae_pct=(min(b.low for b in window) / base - 1) * 100,
        gap_days=gap,
    )

    # 止损/目标谁先触及 —— 顺序比"是否触及"重要：先扫止损再涨回来的
    # 判断，和先摸目标再跌回去的判断，教训完全相反。
    if d.stop is not None or d.target is not None:
        for b in window:
            s = d.stop is not None and b.low <= d.stop
            t = d.target is not None and b.high >= d.target
            if s or t:
                o.stop_hit = o.stop_hit or s
                o.target_hit = o.target_hit or t
                if o.stop_first is None:
                    o.stop_first = bool(s and not t)
                break
        o.stop_hit = o.stop_hit or any(
            d.stop is not None and b.low <= d.stop for b in window)
        o.target_hit = o.target_hit or any(
            d.target is not None and b.high >= d.target for b in window)

    act = _norm_action(d.action)
    # 统一口径：只有买入记「拿到的涨」，其余三种（卖出/观望/回避）都记
    # 「不做多的价值」= -ret。这样每个错误类型的平均影响都有信息量 ——
    # 早先版本把观望记成 0，于是 correct_avoid 那一行永远显示 +0.00%，
    # 等于把一个有效的统计列变成装饰。
    o.agent_ret_pct = o.ret_pct if act == "BUY" else -o.ret_pct
    return o


def classify(d: DecisionRecord, o: Outcome) -> tuple:
    """
    判定结果 + 归因到错误类型 + 生成一句话教训。

    纯规则、纯确定性 —— 刻意不用 LLM 生成教训。理由不是省成本：
    LLM 写的教训会顺手带进窗口之外的记忆，而它的日期戳是按 learned_date
    盖章的，多出来的信息就悄悄漏进去了。规则文本只由 Outcome 里的数字拼出，
    信息边界天然等于窗口边界。

    要用 LLM 生成教训也可以，但必须用 as_of=learned_date 的提示词去生成
    （即让它也活在那个时点），并把 learn 的日期戳设为 learned_date。
    """
    act = _norm_action(d.action)
    ret, mfe, mae = o.ret_pct, o.mfe_pct, o.mae_pct
    g = d.facts_gist or {}
    rsi, pos = g.get("rsi14"), g.get("pos_in_60d")
    high_zone = (rsi is not None and rsi >= 72) or (pos is not None and pos >= 88)
    h = d.horizon_days
    hz = f"{h} 个交易日"

    if act == "BUY":
        if o.stop_hit and o.stop_first and ret > 0:
            return ("miss", "stop_shaken",
                    f"止损 {d.stop} 被扫（最大浮亏 {mae:.1f}%），但{h}后价格回到 "
                    f"{ret:+.1f}% —— 止损距离过紧，被正常波动打掉了。")
        if ret >= FLAT_BAND_PCT:
            return ("hit", "right_call",
                    f"判断成立：{hz}后 {ret:+.1f}%（最大浮盈 {mfe:+.1f}%）。")
        if ret <= -FLAT_BAND_PCT:
            if high_zone:
                return ("miss", "chased_high",
                        f"在 RSI={_n(rsi)}／60日位置 {_n(pos)} 的高位区买入，{hz}后 "
                        f"{ret:+.1f}%（最大浮亏 {mae:.1f}%）—— 超买区追进的胜率明显偏低。")
            if mae <= -5 and mfe < 2:
                return ("miss", "false_breakout",
                        f"入场即套：最大浮亏 {mae:.1f}%，最大浮盈只有 {mfe:+.1f}%，"
                        f"最终 {ret:+.1f}% —— 形态没等量能确认就进场了。")
            return ("miss", "wrong_direction",
                    f"方向看错：{hz}后 {ret:+.1f}%，期间最大浮亏 {mae:.1f}%。")

    elif act == "SELL":
        if ret <= -FLAT_BAND_PCT:
            return ("hit", "right_call",
                    f"离场正确：卖出后{h}内下跌 {ret:+.1f}%（最低 {mae:.1f}%）。")
        if ret >= FLAT_BAND_PCT:
            return ("miss", "sold_too_early",
                    f"卖出后{h}内继续上涨 {ret:+.1f}%（最高 {mfe:+.1f}%）—— 卖早了。")

    else:   # HOLD / AVOID
        if ret >= AVOID_BAND_PCT:
            return ("miss", "missed_opportunity",
                    f"观望期间上涨 {ret:+.1f}%（最高 {mfe:+.1f}%）—— 过度保守，"
                    f"这类信号出现时本标的确实会走。")
        if ret <= -AVOID_BAND_PCT:
            return ("hit", "correct_avoid",
                    f"回避正确：期间下跌 {ret:+.1f}%（最低 {mae:.1f}%），不参与是对的。")

    return ("flat", "flat",
            f"{hz}后 {ret:+.1f}%，落在 ±{FLAT_BAND_PCT:.0f}% 的噪声区间，"
            f"这次判断没有信息量。")


def _n(v) -> str:
    return "—" if v is None else f"{float(v):.1f}"


# ============================================================
# 存储
# ============================================================

class MemoryStore:
    """
    两份 append-only JSONL。刻意不做「更新状态」这件事 ——
    状态是 join 出来的，台账本体永不回改，这样才能当证据用。
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_root()
        self.decisions_path = self.root / "decisions.jsonl"
        self.lessons_path = self.root / "lessons.jsonl"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # noqa: BLE001
            log.warning("记忆目录创建失败（%s），记忆功能将只读", exc)

    # ---------- 底层 I/O ----------

    @staticmethod
    def _read(path: Path) -> list:
        if not path.exists():
            return []
        out: list = []
        bad = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            # 坏行不炸、不静默 —— 半截写入或手工编辑都会留下坏行，
            # 静默吞掉会让"记忆里到底有多少条"永远对不上账。
            log.warning("%s 有 %d 行无法解析，已跳过（记忆条数可能不完整）",
                        path.name, bad)
        return out

    @staticmethod
    def _append(path: Path, records: Iterable[Any]) -> int:
        rows = list(records)
        if not rows:
            return 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            for r in rows:
                d = asdict(r) if not isinstance(r, dict) else r
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return len(rows)

    # ---------- 读 ----------

    def decisions(self, symbol: Optional[str] = None) -> list:
        rows = [d for d in self._read(self.decisions_path)
                if not symbol or d.get("symbol") == symbol]
        out = []
        for r in rows:
            try:
                out.append(DecisionRecord(**{k: v for k, v in r.items()
                                             if k in DecisionRecord.__dataclass_fields__}))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda x: (x.trade_date, x.symbol))
        return out

    def lessons(self, symbol: Optional[str] = None) -> list:
        rows = [d for d in self._read(self.lessons_path)
                if not symbol or d.get("symbol") == symbol]
        out = []
        for r in rows:
            try:
                out.append(LessonRecord(**{k: v for k, v in r.items()
                                           if k in LessonRecord.__dataclass_fields__}))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda x: (x.learned_date, x.symbol))
        return out

    def lesson_index(self) -> dict:
        """decision_id → lesson。这就是「resolved 标签」的实现。"""
        return {l.decision_id: l for l in self.lessons()}

    def open_decisions(self, symbol: Optional[str] = None) -> list:
        idx = self.lesson_index()
        return [d for d in self.decisions(symbol) if d.decision_id not in idx]

    def known_decision_ids(self) -> set:
        return {d.decision_id for d in self.decisions()}

    # ---------- 写：记录判断 ----------

    def record_decision(self, signal, snap: Any = None,
                        facts: Optional[dict] = None,
                        visible_lessons: Optional[list] = None) -> Optional[DecisionRecord]:
        """
        把一次 analyze 的判断落下。重复（同日期+同标的+同方向+同版本）不重复写。

        trade_date 用 signal.date —— 它是「决策所依据的最后一根K线日期」，
        也就是这次判断的 as_of。用它而不是 datetime.now()：回放/补跑时
        now() 会把门控算错。
        """
        did = _decision_id(signal)
        if did in self.known_decision_ids():
            return None
        facts = facts if facts is not None else _snap_dict(snap)
        rec = DecisionRecord(
            decision_id=did,
            symbol=signal.symbol,
            trade_date=asof.norm(signal.date),
            action=_norm_action(signal.action),
            score=float(getattr(signal, "score", 50.0) or 50.0),
            confidence=float(getattr(signal, "confidence", 0.5) or 0.5),
            entry=_f(getattr(signal, "entry", None)),
            stop=_f(getattr(signal, "stop", None)),
            target=_f(getattr(signal, "target", None)),
            horizon_days=int(getattr(signal, "horizon_days", 10) or 10),
            source=getattr(signal, "source", ""),
            model=getattr(signal, "model", ""),
            prompt_version=getattr(signal, "prompt_version", ""),
            reason=getattr(signal, "reason", "")[:200],
            facts_digest=_digest(facts),
            facts_gist=_gist(snap, facts),
            visible_lessons=list(visible_lessons or []),
            decided_on=asof.today(),
        )
        self._append(self.decisions_path, [rec])
        log.info("记忆 | 记下判断 %s %s %s（horizon %d 日）",
                 rec.trade_date, rec.symbol, rec.action, rec.horizon_days)
        return rec

    # ---------- 写：结算 ----------

    def resolve(self, bars_by_symbol: dict,
                as_of: Optional[str] = None,
                symbol: Optional[str] = None) -> list:
        """
        给所有「前瞻窗口已走完」的 open 判断结算，写出教训。

        参数 as_of 的作用：用它卡住**结算本身**。`reflect --as-of 2025-06-20`
        只允许用到 2025-06-20 的K线来算结果，于是这次结算可以在未来任意时刻
        重跑而结果不变 —— 结算过程也必须是可复现的。
        """
        out: list = []
        idx = self.lesson_index()
        gate = asof.norm(as_of) if as_of else None

        for d in self.decisions(symbol):
            if d.decision_id in idx:
                continue
            bars = bars_by_symbol.get(d.symbol)
            if not bars:
                continue
            pos = _index_of(bars, d.trade_date)
            if pos is None:
                # 决策日不在本次数据窗口内 —— 是"取数窗口不够长"，不是"结算不了"。
                # 不写 void，留给下次更长的窗口。
                continue
            usable = bars if gate is None else asof.clip_bars(bars, gate, f"memory:{d.symbol}")
            if len(usable) != len(bars):
                pos = _index_of(usable, d.trade_date)
                if pos is None:
                    continue

            o = _outcome(d, usable, pos)
            if o is None:
                continue                       # 窗口没走完，继续 open

            learned = usable[pos + d.horizon_days].date
            if gate is not None and learned > gate:
                continue                       # 结果在该时点还不可知，不许结算

            if o.gap_days > MAX_GAP_DAYS:
                verdict, et, text = ("void", "void",
                                     f"窗口内相邻K线间隔达 {o.gap_days} 天（疑似停牌），"
                                     f"价格不连续，结算无效。")
            else:
                verdict, et, text = classify(d, o)

            rec = LessonRecord(
                lesson_id=hashlib.md5(f"lesson::{d.decision_id}".encode()).hexdigest()[:16],
                decision_id=d.decision_id, symbol=d.symbol, action=d.action,
                decision_date=d.trade_date, learned_date=learned,
                horizon_days=d.horizon_days, bars_used=o.bars_used,
                ret_pct=round(o.ret_pct, 3),
                agent_ret_pct=round(getattr(o, "agent_ret_pct", 0.0), 3),
                mfe_pct=round(o.mfe_pct, 3), mae_pct=round(o.mae_pct, 3),
                stop_hit=o.stop_hit, target_hit=o.target_hit, stop_first=o.stop_first,
                verdict=verdict, error_type=et, lesson=text,
                prompt_version=d.prompt_version, created_on=asof.today(),
            )
            self._append(self.lessons_path, [rec])
            out.append(rec)
        return out

    # ---------- 读：给模型看的那一份 ----------

    def get_past_context(self, symbol: str, as_of: Any,
                         limit_decisions: int = 5,
                         limit_lessons: int = 5,
                         min_sample: int = MIN_SAMPLE_FOR_PATTERN) -> MemoryContext:
        """
        取「在 as_of 时点确实已知」的记忆。

        两道门，缺一不可：
          判断本身   trade_date < as_of   严格小于 —— 当天的判断不进当天上下文，
                                          否则它会拿自己当先例，形成自我强化
          结果可知   learned_date <= as_of 教训必须已经"可知"，而不是已经"被写下"
        """
        a = asof.norm(as_of)
        ctx = MemoryContext(as_of=a, symbol=symbol)

        # ---- 判断：只取决策日严格早于 as_of 的 ----
        past = [d for d in self.decisions(symbol) if d.trade_date < a]
        ctx.decisions = sorted(past, key=lambda x: x.trade_date, reverse=True)[:limit_decisions]
        past_ids = {d.decision_id for d in past}

        # ---- 教训：两道门 ----
        idx = self.lesson_index()
        visible_all = []
        for d in past:
            l = idx.get(d.decision_id)
            if l is None:
                continue
            if l.learned_date <= a:
                visible_all.append(l)
            else:
                # 已结算，但结果在 as_of 时点还没发生 —— 扣留，并且要记账。
                ctx.blocked += 1
                ctx.blocked_until.append(l.learned_date)

        # 仍在验证期（没有 lesson 记录）的判断
        ctx.pending = sum(1 for d in past if d.decision_id not in idx)
        ctx.lessons = sorted(visible_all, key=lambda x: x.learned_date,
                             reverse=True)[:limit_lessons]

        # ---- 反复错误：只用可见教训统计 ----
        ctx.patterns = self._patterns(visible_all, min_sample)
        return ctx

    @staticmethod
    def _patterns(lessons: list, min_sample: int) -> list:
        bucket: dict = {}
        for l in lessons:
            if l.verdict == "void":
                continue
            b = bucket.setdefault(l.error_type, {"n": 0, "hit": 0, "sum": 0.0})
            b["n"] += 1
            b["hit"] += 1 if l.verdict == "hit" else 0
            b["sum"] += l.agent_ret_pct
        out = []
        for name, b in bucket.items():
            # 样本不够就不下结论。这一条和 validator 的 min_trades 是同一个道理：
            # 2 次里错 2 次不是规律，是样本小。
            if b["n"] < min_sample:
                continue
            out.append((name, b["n"], b["hit"], round(b["sum"] / b["n"], 2)))
        # 排序要确定 —— 提示词内容变了缓存就失效，顺序抖动会让回测无法复现
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    # ---------- 统计 ----------

    def stats(self) -> dict:
        ds, ls = self.decisions(), self.lessons()
        idx = {l.decision_id: l for l in ls}
        by_err: dict = {}
        for l in ls:
            b = by_err.setdefault(l.error_type, {"n": 0, "hit": 0, "sum": 0.0})
            b["n"] += 1
            b["hit"] += 1 if l.verdict == "hit" else 0
            b["sum"] += l.agent_ret_pct
        return {
            "decisions": len(ds),
            "resolved": sum(1 for d in ds if d.decision_id in idx),
            "open": sum(1 for d in ds if d.decision_id not in idx),
            "by_error": {k: {"n": v["n"], "hit": v["hit"],
                             "avg": round(v["sum"] / v["n"], 2)}
                         for k, v in sorted(by_err.items())},
            "symbols": len({d.symbol for d in ds}),
            "date_range": ([ds[0].trade_date, ds[-1].trade_date] if ds else ["", ""]),
        }


# ============================================================
# 模块级便捷入口（名字对齐调用方习惯：get_past_context(ticker, as_of=...))
# ============================================================

def get_past_context(symbol: str, as_of: Any,
                     store: Optional[MemoryStore] = None, **kw) -> MemoryContext:
    st = store if store is not None else MemoryStore()
    return st.get_past_context(symbol, as_of, **kw)


def render_memory_audit(ctx: MemoryContext) -> list:
    """把一次记忆注入的情况转成审计行，供日志与报告使用。"""
    if not ctx.has_store:
        return ["记忆库尚未建立（首次运行，无历史判断可参考）"]
    if ctx.empty:
        return [f"记忆 | {ctx.symbol} @ {ctx.as_of}：无已知历史（首次或全部在验证期）"]
    out = [f"记忆 | {ctx.symbol} @ {ctx.as_of}：可见判断 {len(ctx.decisions)} 条、"
           f"已结算教训 {len(ctx.lessons)} 条、规律 {len(ctx.patterns)} 类"]
    if ctx.blocked:
        out.append(f"  扣留 {ctx.blocked} 条已结算教训：结果窗口延伸到 "
                   f"{max(ctx.blocked_until)}，该时点尚不可知")
    if ctx.pending:
        out.append(f"  {ctx.pending} 条判断仍在验证期内，未作为失误呈现")
    return out


# ============================================================
# 工具
# ============================================================

def _decision_id(signal) -> str:
    raw = (f"{getattr(signal, 'date', '')}|{getattr(signal, 'symbol', '')}|"
           f"{getattr(signal, 'source', '')}|{getattr(signal, 'prompt_version', '')}|"
           f"{_norm_action(getattr(signal, 'action', ''))}")
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _digest(facts: Optional[dict]) -> str:
    if not facts:
        return ""
    try:
        return hashlib.md5(json.dumps(facts, sort_keys=True, ensure_ascii=False,
                                      default=str).encode("utf-8")).hexdigest()[:12]
    except (TypeError, ValueError):
        return ""


def _snap_dict(snap: Any) -> Optional[dict]:
    if snap is None:
        return None
    if isinstance(snap, dict):
        return snap
    fn = getattr(snap, "to_prompt_dict", None)
    return fn() if callable(fn) else None


def _gist(snap: Any, facts: Optional[dict]) -> dict:
    """
    留一手「当时的市场状态」。用途是事后归因：判断错了是**方向**问题
    还是**位置**问题（超买区追高），没有这几个数就永远分不清。
    """
    g: dict = {}
    if snap is not None and hasattr(snap, "rsi14"):
        for k, attr in (("rsi14", "rsi14"), ("pos_in_60d", "pos_in_60d"),
                        ("vol_ratio", "vol_ratio"), ("momentum_20d", "momentum_20d"),
                        ("atr_pct", "atr_pct"), ("close", "close"),
                        ("ma20", "ma20"), ("ma60", "ma60")):
            v = getattr(snap, attr, None)
            if v is not None:
                g[k] = round(float(v), 3)
        return g
    f = facts or {}
    for k, key in (("rsi14", "RSI14"), ("pos_in_60d", "60日区间位置"),
                   ("vol_ratio", "量比"), ("momentum_20d", "20日涨幅%"),
                   ("atr_pct", "ATR占价格比%"), ("close", "收盘价")):
        if f.get(key) is not None:
            g[k] = f[key]
    return g


def _index_of(bars: list, date: str) -> Optional[int]:
    for i, b in enumerate(bars):
        if b.date == date:
            return i
    return None


def _f(v) -> Optional[float]:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
