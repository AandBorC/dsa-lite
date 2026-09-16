#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多空辩论 —— 把「多视角对抗」做成可回测、可证伪的东西
============================================================
形态参考 TradingAgents（arXiv 2412.20138）的多空辩论，但只抄它最硬的两处，
并且把它的「软承诺」升级成「硬检查」：

1. **计数器硬终止** —— 轮数由 `turn_limit = 2 * max_rounds` 决定，
   模型无权宣布「聊完了」。理由有三层：
     - 交给模型自认结束 = 可能无限循环，每多跑一轮都是钱
     - 轮数不定 ⇒ 调用次数不定 ⇒ 成本无法预算
     - 轮数不定 ⇒ 每次跑出不同的辩论过程 ⇒ 不可复现
   本项目的立场更硬一点：**连「认输」也不缩短辩论**。
   `concede` 只作为记录，轮数照跑满 —— 否则「辩论多长」这个变量
   就又交回到模型手里了，而且是以最隐蔽的方式。

2. **不受发言顺序影响** —— TradingAgents 的做法是在裁决者提示词里写一句
   「不受发言顺序影响」。那只是承诺，而承诺不能当证据。
   所以这里加一道**顺序置换检验**：同一份辩论记录，正序喂一次、反序再喂一次，
   比较两次裁决。
     - 一致   → 顺序无关，结论可用
     - 不一致 → **该结论不可信**，显式降级为观望，并写明两个值
   代价是多一次裁决调用（+1，不是翻倍），换来一个能写进报告的判据。

顺手修的两处（同一类问题的另两面）：
3. **首轮注入「对手尚未发言」占位符** —— 不给占位符，首轮模型会凭空虚构
   对手观点再加以反驳，整场辩论建立在幻觉上。
4. **裁决失败/动作非法 → 显式标记放弃，不伪造方向**。Action.parse 对
   无法识别的输入会静默回落到 HOLD，那在这里是不可接受的：
   一次失败会被记录成一次正常的「观望判断」。失败伪装成正常，比失败更糟。

与项目其余部分的关系
------------------------------------------------------------
辩论只是**另一种分析器**：`DebateAnalyzer.analyze()` 与单轮分析器同形，
于是记忆门控、分数闸门、台账、prompt 哈希缓存全部照用，不必各写一遍。

它同时是**一个可以被回测回答的问题**：`debate` 与 `llm` 作为两个并列策略，
用同一段行情跑 compare —— 「多视角到底有没有增量」应该被证伪，而不是被相信。

成本：单次决策 = 2*rounds 次发言 + 2 次裁决。rounds=2 → 6 次调用。
所以它是 analyze 的可选路径（--debate），回测里作为独立策略与单轮 llm 对照。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .llm import DEBATE_PROMPT_VERSION, dict_to_signal, extract_json
from .signals import Action, Signal

log = logging.getLogger("debate")

# ------------------------------------------------------------
# 常量
# ------------------------------------------------------------

ROLE_BULL = "bull"
ROLE_BEAR = "bear"
ROLE_JUDGE = "judge"

ORDER_BULL_FIRST = "bull_first"
ORDER_BEAR_FIRST = "bear_first"
ORDERS = (ORDER_BULL_FIRST, ORDER_BEAR_FIRST)

DEFAULT_MAX_ROUNDS = 2

# 裁决权重漂移超过这个幅度，就算动作没变也判为「顺序敏感」。
# 只看动作会漏掉一类情形：两次都给 HOLD，但一次是被多头说服、一次是被空头说服。
# 阈值是主观的，但**必须是个具名常量**——否则它就是一个没人知道的隐含规则。
ORDER_WEIGHT_TOLERANCE = 0.20

# 首轮的占位符。少了它，模型会自己虚构一个对手。
NO_OPPONENT_YET = "（对手尚未发言）"

# 单条发言进入记录时的截断长度。留足信息，同时防止某次啰嗦把 prompt 撑爆。
MAX_TEXT_IN_TRANSCRIPT = 400


# ------------------------------------------------------------
# 三个角色的系统提示词
# ------------------------------------------------------------
#
# 三份提示词里都保留了「时点纪律」段：辩论多绕几个模型调用，
# 但喂进去的事实始终是同一份 as_of 快照，绕路不产生新信息。
# 这一点必须写在提示词里而不只是写在代码里 —— 模型不知道代码。

_TIME_DISCIPLINE = """时点纪律（优先于以下所有规则）：
你只活在用户给出的那个快照日期。你不知道那个日期之后发生的任何事。
- 禁止引用训练数据里关于这段时期的记忆（"后来它涨了""后来它出事了"）
- 禁止假设后续行情、政策、业绩、公告
- 若某个结论只能靠"事后已知的结果"得出，就不要给出这个结论"""

BULL_SYSTEM = f"""你是多空辩论中的**多头一方**。你的任务是把「买入/持有」这一侧的论证做到最强，但不得编造事实。

{_TIME_DISCIPLINE}

辩论规则：
1. 只依据用户给出的指标快照。严禁引用新闻、财报、传闻、基本面——你没有这些数据。
2. 如果用户标注「对手尚未发言」，就**不要假设**对手会说什么，也不要提前反驳它。
   凭空虚构对手观点再加以反驳，会让整场辩论建立在幻觉之上。
3. 引用对方发言时不得断章取义地改写它的结论。
4. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。

输出结构：
{{"action":"BUY|HOLD|AVOID","score":0-100,"point":"不超过60字的核心论点","concede":true|false}}

score 是你这一侧视角下的多空强度，0=极度看空，100=极度看多。
concede 表示你是否承认对方论证成立、放弃本方立场。
注意：concede **不改变辩论轮数**，只作为记录。
不要为了结束辩论而给出 concede，也不要为了显得坚定而拒绝承认明显事实。"""

BEAR_SYSTEM = f"""你是多空辩论中的**空头一方**。你的任务是把「卖出/回避」这一侧的论证做到最强，但不得编造事实。

{_TIME_DISCIPLINE}

辩论规则：
1. 只依据用户给出的指标快照。严禁引用新闻、财报、传闻、基本面——你没有这些数据。
2. 如果用户标注「对手尚未发言」，就**不要假设**对手会说什么，也不要提前反驳它。
   凭空虚构对手观点再加以反驳，会让整场辩论建立在幻觉之上。
3. 引用对方发言时不得断章取义地改写它的结论。
4. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。

输出结构：
{{"action":"SELL|HOLD|AVOID","score":0-100,"point":"不超过60字的核心论点","concede":true|false}}

score 是你这一侧视角下的多空强度，0=极度看空，100=极度看多。
concede 表示你是否承认对方论证成立、放弃本方立场。
注意：concede **不改变辩论轮数**，只作为记录。
不要为了结束辩论而给出 concede，也不要为了显得坚定而拒绝承认明显事实。"""

JUDGE_SYSTEM = f"""你是多空辩论的**裁决者**。你只依据双方论证与指标快照做判断，禁止引用任何外部信息。

{_TIME_DISCIPLINE}

两条最重要的规则：
1. **你的判断不受发言顺序影响。** 记录里的先后只是**记录顺序**，
   先出现的一方不因为先出现而占优。请对双方论证做同等严格的评估。
2. 若双方论证强度接近、或指标本身矛盾，就给 HOLD ——
   不要为了显得果断而硬造一个方向。观望是合法且常见的结论。

只输出一个 JSON 对象，不要 markdown 代码块：
{{"action":"BUY|SELL|HOLD|AVOID","score":0-100,"confidence":0-1,
 "stop":数字或null,"target":数字或null,
 "reason":"不超过60字，须说明采纳哪一方、为什么",
 "weight_bull":0-1,"weight_bear":0-1}}

weight_bull / weight_bear 是你对双方论证的采纳权重，两者之和为 1。
给 BUY 时必须给 stop 和 target，且 stop < 当前价 < target。"""


# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------

@dataclass
class DebateConfig:
    """
    辩论配置。三个字段都会进入每次调用的 prompt，所以它们天然被缓存哈希覆盖 ——
    改配置等于换一整套缓存，不会出现「改了参数却命中旧结果」。
    """
    max_rounds: int = DEFAULT_MAX_ROUNDS
    order: str = ORDER_BULL_FIRST
    order_check: bool = True        # 顺序置换检验；关掉可省一次裁决调用

    def turn_limit(self) -> int:
        """发言总次数 = 每轮多空各一次。下限 2 —— 少于一个完整回合就不叫辩论。"""
        return max(2, int(self.max_rounds) * 2)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DebateConfig":
        d = d or {}
        return cls(
            max_rounds=int(d.get("max_rounds", DEFAULT_MAX_ROUNDS) or DEFAULT_MAX_ROUNDS),
            order=str(d.get("order", ORDER_BULL_FIRST) or ORDER_BULL_FIRST),
            order_check=bool(d.get("order_check", True)),
        )


def order_sequence(order: str, max_rounds: int) -> list[str]:
    """
    发言顺序。**认不出就抛错** —— 静默回落到默认顺序，
    会让「我明明设了 bear_first」变成一句没人发现的谎话。
    """
    if order not in ORDERS:
        raise ValueError(f"未知发言顺序: {order}（可选 {ORDERS}）")
    first = ROLE_BULL if order == ORDER_BULL_FIRST else ROLE_BEAR
    second = ROLE_BEAR if first == ROLE_BULL else ROLE_BULL
    n = max(1, int(max_rounds))
    return [r for _ in range(n) for r in (first, second)]


# ------------------------------------------------------------
# 记录结构
# ------------------------------------------------------------

@dataclass
class Turn:
    """一次发言。seq 是**全局计数器**，不是模型说的第几轮。"""
    seq: int
    role: str
    text: str
    parsed: Optional[dict] = None

    @property
    def action(self) -> Optional[str]:
        if not self.parsed:
            return None
        v = str(self.parsed.get("action") or "").strip().upper()
        return v or None

    @property
    def score(self) -> Optional[float]:
        return _num((self.parsed or {}).get("score"))

    @property
    def point(self) -> str:
        return str((self.parsed or {}).get("point") or "").strip()

    @property
    def conceded(self) -> bool:
        return bool((self.parsed or {}).get("concede"))

    def as_dict(self) -> dict:
        return {"seq": self.seq, "role": self.role, "text": self.text,
                "action": self.action, "score": self.score, "concede": self.conceded}


@dataclass
class DebateResult:
    symbol: str
    date: str
    order: str
    max_rounds: int
    turns: list = field(default_factory=list)
    judge: Optional[dict] = None
    judge_reversed: Optional[dict] = None
    order_sensitive: bool = False
    order_note: str = ""
    order_checked: bool = False
    calls: int = 0

    # ---------- 派生 ----------

    @property
    def turn_limit(self) -> int:
        return max(2, int(self.max_rounds) * 2)

    @property
    def completed(self) -> bool:
        """是否跑满计数器。永远为 True 才算「硬终止」生效。"""
        return len(self.turns) >= self.turn_limit

    @property
    def transcript_hash(self) -> str:
        """
        辩论过程的内容指纹。用于「同输入 → 同过程」的可复现性断言，
        也用于落盘去重：同一个 hash 存两次没有意义。
        """
        payload = json.dumps([[t.role, t.text] for t in self.turns], ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def judge_action(self) -> Optional[str]:
        return _norm_action((self.judge or {}).get("action"))

    @property
    def judge_reversed_action(self) -> Optional[str]:
        return _norm_action((self.judge_reversed or {}).get("action"))

    @property
    def judge_weight_bull(self) -> Optional[float]:
        """裁决者对多头论证的采纳权重。只看动作会漏掉「两次都 HOLD 但偏向相反」。"""
        return _num((self.judge or {}).get("weight_bull"))

    @property
    def judge_reversed_weight_bull(self) -> Optional[float]:
        return _num((self.judge_reversed or {}).get("weight_bull"))

    @property
    def failed(self) -> bool:
        return self.verdict()[2].get("debate_failed", False)

    def verdict(self) -> tuple[Action, str, dict]:
        """
        把辩论收敛成一个可执行动作，返回 (action, reason, meta_flags)。

        三种情况都**不伪装**：
          - 裁决没解析出来        → 显式放弃，说明这是"没产出决策"而非"决定观望"
          - 动作字段非法/缺失    → 同上（Action.parse 会静默回落 HOLD，这里不能忍）
          - 顺序敏感             → 结论不可信，降级观望并写明两个值
        """
        if self.judge is None:
            return (Action.HOLD,
                    "辩论裁决无法解析，本次显式放弃（这不是观望判断，是这次决策没有产出）",
                    {"debate_failed": True, "debate_fail_reason": "judge_unparsed"})

        act = self.judge_action
        if act is None:
            return (Action.HOLD,
                    "辩论裁决的动作字段非法或缺失，显式放弃（不猜、不回落成观望判断）",
                    {"debate_failed": True, "debate_fail_reason": "judge_action_invalid"})

        if self.order_sensitive:
            a, b = self.judge_action, self.judge_reversed_action
            return (Action.HOLD,
                    f"辩论结论对发言顺序敏感（正序 {a} / 反序 {b}；{self.order_note}），"
                    f"该结论不可信，降为观望",
                    {"debate_order_sensitive": True,
                     "debate_forward": a, "debate_reversed": b})

        reason = str(self.judge.get("reason") or "").strip() or f"辩论裁决 {act}"
        flags = {}
        if not self.order_checked:
            # 没做检验 ≠ 检验通过。这两件事混在一起，报告就会给出虚假保证。
            flags["debate_order_unchecked"] = True
        return Action.parse(act), reason, flags

    def summary(self) -> dict:
        """进台账 meta 的精简摘要 —— 全文另存，摘要保证 CSV 不被撑爆。"""
        return {
            "max_rounds": self.max_rounds,
            "turn_limit": self.turn_limit,
            "turns": len(self.turns),
            "completed": self.completed,
            "order": self.order,
            "judge_action": self.judge_action,
            "judge_reversed_action": self.judge_reversed_action,
            # 权重一起存：只看动作会把"两次都 HOLD 但一次偏多一次偏空"当成顺序无关
            "judge_weight_bull": self.judge_weight_bull,
            "judge_reversed_weight_bull": self.judge_reversed_weight_bull,
            "order_checked": self.order_checked,
            "order_sensitive": self.order_sensitive,
            "calls": self.calls,
            "transcript_hash": self.transcript_hash,
            "bull_actions": [t.action for t in self.turns if t.role == ROLE_BULL],
            "bear_actions": [t.action for t in self.turns if t.role == ROLE_BEAR],
            "concedes": [t.seq for t in self.turns if t.conceded],
        }

    def as_record(self) -> dict:
        """落盘用：摘要 + 每一步发言原文。"""
        return {**self.summary(), "symbol": self.symbol, "date": self.date,
                "judge": self.judge, "judge_reversed": self.judge_reversed,
                "order_note": self.order_note,
                "transcript": [t.as_dict() for t in self.turns]}

    def render(self) -> str:
        """人读的辩论报告。"""
        L = [f"## 多空辩论 {self.symbol} @ {self.date}",
             f"- 配置：{self.max_rounds} 轮 × 2 = {self.turn_limit} 次发言"
             f"（实际 {len(self.turns)} 次），先手 {self.order}，共 {self.calls} 次调用",
             f"- 过程指纹：`{self.transcript_hash}`", ""]
        for t in self.turns:
            who = "多头" if t.role == ROLE_BULL else "空头"
            a = t.action or "?"
            s = f" score={t.score}" if t.score is not None else ""
            note = "（承认对方论证）" if t.conceded else ""
            L.append(f"**[{t.seq + 1}] {who}** {a}{s}{note}")
            L.append(f"　{t.point or t.text[:120]}")
        L.append("")
        L.append(f"**裁决（正序 {self.order}）**：{self.judge_action or '解析失败'}")
        if self.judge and self.judge.get("reason"):
            L.append(f"　{self.judge['reason']}")
        if self.order_checked:
            L.append(f"**裁决（反序）**：{self.judge_reversed_action or '解析失败'}")
            mark = "⚠️ 顺序敏感" if self.order_sensitive else "✅ 顺序无关"
            L.append(f"**顺序置换检验**：{mark} —— {self.order_note}")
        else:
            L.append("**顺序置换检验**：未执行（`--no-order-check`）—— "
                     "因此「不受发言顺序影响」这句话本次**未经检验**")
        return "\n".join(L)


# ------------------------------------------------------------
# 提示词构造
# ------------------------------------------------------------

def facts_header(snap) -> str:
    """
    事实块。**复用单轮路径的 render_text()** ——
    辩论不得引入任何单轮路径看不到的数据，否则两边就不再可比，
    回测比出来的差异也就分不清是"多视角的功劳"还是"多喂了数据"。
    """
    return (f"标的：{snap.symbol}\n"
            f"以下是指标快照（截至 {snap.date} 收盘，均为已发生事实）：\n\n"
            f"{snap.render_text()}")


def _role_cn(role: str) -> str:
    return {ROLE_BULL: "多头", ROLE_BEAR: "空头", ROLE_JUDGE: "裁决者"}.get(role, role)


def render_entries(turns: list) -> str:
    """
    把发言渲染成记录。

    **不标注轮次**，只编连续序号 —— 顺序置换检验要求「顺序」是唯一的变量，
    留着"第2轮"的字样会让反序版本显得自相矛盾，等于把变量弄脏了。
    """
    if not turns:
        return NO_OPPONENT_YET
    lines = []
    for n, t in enumerate(turns, 1):
        a = t.action or "?"
        s = f" score={t.score}" if t.score is not None else ""
        body = t.point or t.text[:MAX_TEXT_IN_TRANSCRIPT]
        lines.append(f"{n}. [{_role_cn(t.role)}] {a}{s} 论点：{body}")
    return "\n".join(lines)


def build_turn_user(snap, memory_block: str, prior_turns: list) -> str:
    """某位辩手的 user 消息。首轮与后续轮的差别只有一处：有没有对手发言。"""
    parts = [facts_header(snap)]
    if memory_block and memory_block.strip():
        parts.append(memory_block.strip())
    parts.append(f"现在是 {snap.date} 收盘之后，{snap.date} 之后发生的事你一无所知。")
    if not prior_turns:
        parts.append(f"# 辩论记录\n\n{NO_OPPONENT_YET}\n\n"
                     "你是先发言的一方，请独立陈述你的论证。")
    else:
        parts.append("# 辩论记录\n\n" + render_entries(prior_turns))
        parts.append("现在轮到你发言。请针对以上论证给出你的观点。")
    parts.append("记住：只输出 JSON。")
    return "\n\n".join(parts)


def build_judge_user(snap, memory_block: str, turns: list, reverse: bool = False) -> str:
    """裁决者的 user 消息。reverse=True 时只把记录顺序倒过来，别的一字不改。"""
    seq = list(reversed(turns)) if reverse else list(turns)
    parts = [facts_header(snap)]
    if memory_block and memory_block.strip():
        parts.append(memory_block.strip())
    parts.append("# 辩论发言（按发言先后排列）\n\n" + render_entries(seq))
    parts.append("请裁决。记住：只输出 JSON。")
    return "\n\n".join(parts)


# ------------------------------------------------------------
# 顺序置换检验
# ------------------------------------------------------------

def compare_verdicts(forward: Optional[dict], reverse: Optional[dict]) -> tuple[bool, str]:
    """
    比较正序与反序两次裁决，返回 (是否顺序敏感, 说明)。

    只比动作会漏一类情形：两次都判 HOLD，但一次是被多头说服、一次是被空头说服。
    所以动作相同时也看权重漂移；权重缺失就只比动作，并在说明里讲清楚
    —— 检验力度不足必须说出来，不能默默按"通过"记。
    """
    if forward is None or reverse is None:
        return False, "有一次裁决未能解析，无法比较（不构成顺序敏感的证据）"

    fa, ra = _norm_action(forward.get("action")), _norm_action(reverse.get("action"))
    if fa != ra:
        return True, f"两次裁决动作不同（正序 {fa} / 反序 {ra}）"

    wf, wr = _num(forward.get("weight_bull")), _num(reverse.get("weight_bull"))
    if wf is None or wr is None:
        return False, f"动作一致（{fa}）；权重缺失，本次仅比较动作"
    drift = abs(wf - wr)
    if drift > ORDER_WEIGHT_TOLERANCE:
        return True, (f"动作一致（{fa}）但多头权重漂移 {drift:.2f} "
                      f"（正序 {wf:.2f} / 反序 {wr:.2f}）超过 {ORDER_WEIGHT_TOLERANCE}")
    return False, f"动作与权重均一致（{fa}，多头权重漂移 {drift:.2f}）"


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def run_debate(ask: Callable[[str, str], str], snap, memory_block: str = "",
               cfg: Optional[DebateConfig] = None) -> DebateResult:
    """
    跑一场辩论。

    ask(system, user) -> str 是唯一的对外依赖 —— 缓存、重试、限流都由调用方
    在 ask 里解决，于是本函数是纯逻辑，可以用一个假函数完整测试。
    """
    c = cfg or DebateConfig()
    limit = c.turn_limit()
    seq = order_sequence(c.order, c.max_rounds)
    turns: list = []
    calls = 0

    # ---- 计数器硬终止：循环条件只看 len(turns)，模型没有任何发言权 ----
    # 解析失败也照样 append（parsed=None），因为**发言发生过**是事实，
    # 而轮数必须可预测。若把解析失败算作"这一步不算"，轮数就又不确定了。
    while len(turns) < limit:
        role = seq[len(turns)]
        system = BULL_SYSTEM if role == ROLE_BULL else BEAR_SYSTEM
        user = build_turn_user(snap, memory_block, turns)
        text = ask(system, user)
        calls += 1
        turns.append(Turn(seq=len(turns), role=role, text=text or "",
                          parsed=extract_json(text or "")))

    res = DebateResult(symbol=snap.symbol, date=snap.date, order=c.order,
                       max_rounds=c.max_rounds, turns=turns)

    # ---- 裁决 + 顺序置换检验 ----
    res.judge = extract_json(ask(JUDGE_SYSTEM, build_judge_user(snap, memory_block, turns)))
    calls += 1
    if c.order_check:
        res.judge_reversed = extract_json(
            ask(JUDGE_SYSTEM, build_judge_user(snap, memory_block, turns, reverse=True)))
        calls += 1
        res.order_checked = True
        res.order_sensitive, res.order_note = compare_verdicts(res.judge, res.judge_reversed)
    else:
        res.order_note = "未执行顺序置换检验，结论未经检验"
    res.calls = calls
    return res


# ------------------------------------------------------------
# 落盘：辩论过程（append-only JSONL）
# ------------------------------------------------------------

def default_root() -> Path:
    """
    DSA_DEBATE_DIR 可覆盖位置。
    CI 里导到临时目录，避免"假辩论"被提交进仓库 ——
    辩论记录是证据，往里塞测试数据等于污染证据。
    """
    env = os.environ.get("DSA_DEBATE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "debate"


class TranscriptStore:
    """
    辩论过程落盘。

    为什么要存全文：裁决只给一个动作，事后复盘时唯一能回答
    「它为什么这么判」的就是辩论过程本身。只存动作，等于把过程丢了。
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_root() / "transcripts.jsonl"
        self._hashes: Optional[set] = None

    def _load_hashes(self) -> set:
        if self._hashes is None:
            self._hashes = set()
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._hashes.add(json.loads(line).get("transcript_hash"))
                    except Exception:  # noqa: BLE001
                        continue
        return self._hashes

    def append(self, res: DebateResult) -> bool:
        """按过程指纹去重后追加。返回是否真的写入。"""
        h = res.transcript_hash
        if h in self._load_hashes():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.as_record(), ensure_ascii=False) + "\n")
        self._hashes.add(h)
        return True

    def read(self, symbol: Optional[str] = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if symbol and rec.get("symbol") != symbol:
                continue
            out.append(rec)
        return out


# ------------------------------------------------------------
# 适配成分析器：与单轮分析器同形
# ------------------------------------------------------------

class DebateAnalyzer:
    """
    把多空辩论包成 `analyze(snap, ...) -> Signal`。

    与 LLMAnalyzer 同形，所以 strategies.LLMStrategy 可以原样复用 ——
    记忆门控、分数闸门、台账、prompt 哈希缓存都不必再写一遍。
    这也保证了 A/B 对比的公平：两条路径的**周边机制完全相同**，
    唯一的差别就是"单轮判断"还是"多空辩论"。

    缓存：每一轮发言、每一次裁决都各是一次独立调用，各自进 prompt 哈希缓存。
    所以回测重跑是免费的，而且逐轮可复现 —— 辩论这种"多步生成"最容易
    变成"每次跑都不一样"，缓存是唯一能挡住它的东西。

    落盘：transcript_store 只由 analyze/CLI 传入。回测路径不传，
    这样跑回测不会往仓库里灌辩论记录。
    """

    def __init__(self, analyzer, cfg: Optional[DebateConfig] = None,
                 transcript_store: Optional[TranscriptStore] = None,
                 use_cache: bool = True):
        self.analyzer = analyzer
        self.cfg = cfg or DebateConfig()
        self.store = transcript_store
        self.use_cache = use_cache
        self.model = getattr(analyzer, "model", "")
        self.prompt_version = DEBATE_PROMPT_VERSION
        self.last: Optional[DebateResult] = None
        # 给统计打上归属标记。validator 靠它把辩论统计与底层分析器的 token 统计
        # 分开 —— 两者的键名有重叠（都有 calls），混在一起会算错。
        self.stats_scope = "debate"
        self.stats = {"debates": 0, "turns": 0, "calls": 0,
                      "judge_failed": 0, "order_sensitive": 0,
                      "order_checked": 0, "order_unchecked": 0,
                      "flagged": 0}

    @property
    def available(self) -> bool:
        return bool(getattr(self.analyzer, "available", True))

    def _ask(self, system: str, user: str) -> str:
        """所有模型调用都从这里走，于是缓存与重试集中在底层分析器一处。"""
        raw = getattr(self.analyzer, "raw_chat", None)
        if raw is None:
            raise RuntimeError(f"{type(self.analyzer).__name__} 不支持 raw_chat，无法辩论")
        return raw(system, user, use_cache=self.use_cache)

    def analyze(self, snap, position=None, holding: bool = False,
                memory_block: str = "") -> Signal:
        res = run_debate(self._ask, snap, memory_block, self.cfg)
        self.last = res

        act, reason, flags = res.verdict()

        # 复用单轮路径的转换器：字段夹取、BUY 才能挂价位、stop>=target 丢弃，
        # 规则全部一致。差异只应来自判断本身，不应来自装配方式的区别。
        d = dict(res.judge or {})
        d["action"] = act.value
        d["reason"] = reason
        if act == Action.BUY and (d.get("stop") is None or d.get("target") is None):
            # 裁决没给价位时的兜底。标记出来，免得和"裁决自己给的价位"混为一谈。
            atr = getattr(snap, "atr14", None) or snap.close * 0.02
            d["stop"] = round(snap.close - 2.5 * atr, 2)
            d["target"] = round(snap.close + 5 * atr, 2)
            flags["levels_from_atr"] = True

        sig = dict_to_signal(d, snap.symbol, snap.date, self.model,
                             DEBATE_PROMPT_VERSION, source="debate",
                             ref_price=snap.close)
        sig.meta.update(res.summary())
        sig.meta.update(flags)
        # meta 里**只留摘要与指纹，不留全文**。理由：meta 会写进 signals.csv 的每一行，
        # 把四段发言塞进去会让台账迅速膨胀到没法用编辑器打开。
        # 全文落在 TranscriptStore，meta 里的 transcript_hash 就是查它的钥匙 ——
        # 与记忆库同构：台账存索引，正文另存，两边都不许改。

        self.stats["debates"] += 1
        self.stats["turns"] += len(res.turns)
        self.stats["calls"] += res.calls
        self.stats["judge_failed"] += 1 if res.failed else 0
        self.stats["order_sensitive"] += 1 if res.order_sensitive else 0
        self.stats["order_checked"] += 1 if res.order_checked else 0
        self.stats["order_unchecked"] += 0 if res.order_checked else 1
        self.stats["flagged"] += 1 if (res.failed or res.order_sensitive) else 0

        if self.store is not None:
            self.store.append(res)
        return sig

    def cost_report(self) -> dict:
        base = {}
        if hasattr(self.analyzer, "cost_report"):
            try:
                base = self.analyzer.cost_report()
            except Exception:  # noqa: BLE001
                base = {}
        return {**base, "debate": dict(self.stats)}


def render_debate_audit(dab: "DebateAnalyzer", rows: Optional[list] = None) -> list[str]:
    """
    辩论审计行。理由与记忆审计完全一样：
    一个「顺序没检验、裁决全失败」的静默辩论，在最终信号上和没开辩论长得一样。
    过程必须自报，否则「我们做了顺序检验」就成了一句没人能核实的话。
    """
    s = dab.stats
    c = dab.cfg
    out = [f"共 {s['debates']} 次辩论、{s['turns']} 次发言、{s['calls']} 次模型调用"
           f"（{c.max_rounds} 轮 × 2 次发言 + 裁决）"]
    if s["order_checked"]:
        out.append(f"顺序置换检验 {s['order_checked']} 次，判为顺序敏感 "
                   f"{s['order_sensitive']} 次")
    else:
        out.append("⚠️ 未做顺序置换检验 —— 「不受发言顺序影响」本次**未经检验**。"
                   "没验证不等于验证通过")
    if s["judge_failed"]:
        out.append(f"⚠️ 裁决解析失败 {s['judge_failed']} 次（已显式降级，未伪造方向）")
    for r in (rows or []):
        if r.order_checked:
            mark = "⚠️ 顺序敏感" if r.order_sensitive else "✅ 顺序无关"
        else:
            mark = "— 未检验"
        out.append(f"{r.symbol}：正序 {r.order} → {r.judge_action or '解析失败'}；"
                   f"反序 → {r.judge_reversed_action or '未检验'}；{mark}"
                   f"（过程指纹 {r.transcript_hash}）")
    return out


# ------------------------------------------------------------
# 工具
# ------------------------------------------------------------

def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_action(v) -> Optional[str]:
    """把动作字段规范化。**认不出返回 None**，而不是回落成 HOLD —— 见 verdict()。"""
    s = str(v or "").strip().upper()
    return s if s in ("BUY", "SELL", "HOLD", "AVOID") else None
