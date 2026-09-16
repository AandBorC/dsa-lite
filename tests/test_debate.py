#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多空辩论验收 —— 重点是证明「辩论不会失控」，不是证明「辩论有用」
====================================================================

这个模块抄了 TradingAgents 的两条做法，但把它们从**承诺**变成了**判据**。
所以测试的重心也就不是"辩论能不能跑"，而是：

    1. 轮数真的不受模型影响吗？
       —— 模型说「我认输」、说「还没聊完」、输出一坨无法解析的废话，
          都必须**改变不了发言次数**。任何一条漏了，辩论长度就回到了模型手里。
    2. 「不受发言顺序影响」这句话，到底有没有被检查过？
       —— 必须有对照：一个顺序无关的裁决者要被判「无关」，
          一个先发言者占优的裁决者要被判「敏感**并降级**」。
          只有第一条会通过一个永远返回 False 的空实现，
          只有第二条会通过一个永远返回 True 的过敏实现。两条都要有。
    3. 失败有没有被伪装成正常判断？
       —— 裁决解析失败、动作字段非法，都必须显式标记；
          Action.parse 会把无法识别的输入静默变成 HOLD，那在这里不可接受。

余下的条目（可复现、缓存、只读记忆、不引入未来数据）与项目其余部分同规：
**每条判据都配对照组**，证明场景本身不是空转。

纯离线，零网络，零依赖。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestConfig                              # noqa: E402
from core.debate import (BEAR_SYSTEM, BULL_SYSTEM, JUDGE_SYSTEM,      # noqa: E402
                         NO_OPPONENT_YET, ORDER_BEAR_FIRST, ORDER_BULL_FIRST,
                         ORDER_WEIGHT_TOLERANCE, DebateAnalyzer, DebateConfig,
                         TranscriptStore, build_judge_user, default_root,
                         order_sequence, render_debate_audit, run_debate)
from core.fetchers import Bar                                         # noqa: E402
from core.indicators import build_snapshot                            # noqa: E402
from core.llm import LLMAnalyzer, mock_role_of                        # noqa: E402
from core.memory import MemoryStore                                   # noqa: E402
from core.signals import SignalCache                                  # noqa: E402
from core.strategies import LLMStrategy                               # noqa: E402
from core.validator import StrategyValidator, render_validation        # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, got, want, tol: float = 1e-6) -> None:
    global PASS, FAIL
    ok = False
    if isinstance(want, bool):
        ok = bool(got) is want
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


def check_true(name: str, cond, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f": {extra}" if extra else ""))


def check_in(name: str, text: str, needle: str, want: bool = True) -> None:
    """断言 needle 是否出现在文本里。用于提示词内容检查。"""
    got = needle in (text or "")
    check_true(name, got is want,
               f"出现={got}（期望 {want}）needle={needle[:34]!r}")


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ============================================================
# 造数据
# ============================================================

def _dates(n: int, start: str = "2025-01-02") -> list[str]:
    """n 个交易日日期（跳周末）。"""
    out, d = [], datetime.strptime(start, "%Y-%m-%d")
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def make_bars(prices: list, start: str = "2025-01-02") -> list:
    ds, bars, prev = _dates(len(prices), start), [], prices[0]
    for d, c in zip(ds, prices):
        o = prev
        bars.append(Bar(date=d, open=o, close=c, high=max(o, c) * 1.004,
                        low=min(o, c) * 0.996, volume=1e6, amount=0.0,
                        pct_chg=0.0, turnover=0.0, amplitude=0.0))
        prev = c
    return bars


PRICES = [100 + i * 0.35 for i in range(110)]
BARS = make_bars(PRICES)
SNAP = build_snapshot("sh600519", BARS, len(BARS) - 1)


# ============================================================
# 假分析器：把「模型说的话」写死
# ============================================================

def bull_reply(action: str = "BUY", score: float = 70.0,
               point: str = "多头论点-ABC", concede: bool = False) -> str:
    return json.dumps({"action": action, "score": score, "point": point,
                       "concede": concede}, ensure_ascii=False)


def bear_reply(action: str = "HOLD", score: float = 35.0,
               point: str = "空头论点-XYZ", concede: bool = False) -> str:
    return json.dumps({"action": action, "score": score, "point": point,
                       "concede": concede}, ensure_ascii=False)


def judge_reply(action: str = "HOLD", reason: str = "按指标综合裁决",
                wb: float = 0.5) -> str:
    return json.dumps({"action": action, "score": 55.0, "confidence": 0.5,
                       "reason": reason, "weight_bull": wb,
                       "weight_bear": round(1 - wb, 2)}, ensure_ascii=False)


class FakeAnalyzer:
    """
    可编程的假分析器。

    真实模型不可控，而这里要测的恰恰是「当模型说 X 时系统会做什么」——
    所以模型的话必须能写死。replies[role] 可以是字符串，也可以是
    callable(role, user, 第几次调用) 用来模拟"看人下菜碟"的裁决者。
    """

    def __init__(self, replies: dict | None = None, default: str = "",
                 model: str = "fake-llm"):
        self.replies = replies or {}
        self.default = default
        self.model = model
        self.available = True
        self.stats = {"calls": 0}
        self.seen: list[tuple[str, str]] = []

    def role_calls(self, role: str) -> int:
        return sum(1 for r, _ in self.seen if r == role)

    def prompts(self, role: str) -> list[str]:
        return [u for r, u in self.seen if r == role]

    def raw_chat(self, system: str, user: str, use_cache: bool = True) -> str:
        role = mock_role_of(system)
        self.seen.append((role, user))
        self.stats["calls"] += 1
        spec = self.replies.get(role, self.default)
        if callable(spec):
            return spec(role, user, self.role_calls(role))
        return spec


def run_with(replies: dict, default: str = "", cfg: DebateConfig | None = None,
             memory: str = "", snap=None):
    """跑一场辩论并同时拿回假分析器（要看它收到了什么 prompt）。"""
    fa = FakeAnalyzer(replies, default)
    res = run_debate(lambda s, u: fa.raw_chat(s, u), snap or SNAP, memory,
                     cfg or DebateConfig())
    return fa, res


BASIC = {"bull": bull_reply(), "bear": bear_reply(), "judge": judge_reply("HOLD")}

TMP = Path(tempfile.mkdtemp(prefix="dsa_debate_test_"))

try:
    # ========================================================
    section("1. 计数器硬终止 —— 轮数由循环条件定，模型说什么都不算")

    fa, res = run_with(BASIC)
    check("1a 正常跑满计数器", len(res.turns), 4)
    check("1a completed", res.completed, True)
    check("1a 调用数 = 发言 + 2 次裁决", res.calls, 6)
    check("1a 发言序号是 0..3 的全局计数", [t.seq for t in res.turns], [0, 1, 2, 3])

    # 1b 模型全程认输
    fa, res = run_with({"bull": bull_reply(concede=True),
                        "bear": bear_reply(concede=True), "judge": judge_reply("HOLD")})
    check("1b 全程认输仍跑满 4 次发言", len(res.turns), 4)
    check_true("1b 认输被如实记录下来", all(t.conceded for t in res.turns))
    check_true("1b 认输不缩短辩论 —— 否则辩论长度又回到模型手里", res.completed)

    # 1c 模型宣称"还没聊完" + 输出无法解析的废话
    fa, res = run_with({"judge": judge_reply("HOLD")},
                       default="我们还没聊完，请不要结束，继续辩论。")
    check("1c 宣称继续也不延长", len(res.turns), 4)
    check("1c 无法解析的发言照样占一次计数",
          sum(1 for t in res.turns if t.parsed is None), 4)
    check("1c 调用数仍然可预测", res.calls, 6)

    # 1d 轮数随配置线性
    for mr, want in ((1, 2), (2, 4), (3, 6)):
        _, r = run_with({"judge": judge_reply("HOLD")}, "{}",
                        DebateConfig(max_rounds=mr))
        check(f"1d max_rounds={mr} → turn_limit={want}", r.turn_limit, want)
        check(f"1d max_rounds={mr} → 实际发言 {want} 次", len(r.turns), want)

    # 1e 关掉顺序检验 → 少一次调用
    _, res = run_with({"judge": judge_reply("HOLD")}, "",
                      DebateConfig(order_check=False))
    check("1e 少了反序裁决，调用数 = 发言 + 1", res.calls, 5)
    check("1e order_checked=False", res.order_checked, False)

    # ========================================================
    section("2. 首轮占位符 —— 不给它，模型会凭空虚构一个对手再反驳")

    fa, res = run_with({"bull": bull_reply(point="多头论点-ABC"),
                        "bear": bear_reply(point="空头论点-XYZ"),
                        "judge": judge_reply("HOLD")})
    bu, be = fa.prompts("bull"), fa.prompts("bear")
    check("2a 多头收到 2 次发言请求", len(bu), 2)
    check_in("2b 多头首轮含「对手尚未发言」占位符", bu[0], NO_OPPONENT_YET, True)
    check_in("2c 首轮不得出现对手论点（否则就是让它对着幻觉反驳）",
             bu[0], "空头论点-XYZ", False)
    check_in("2d 对照：多头第二轮确实拿到了对手论点", bu[1], "空头论点-XYZ", True)
    check_in("2e 首轮不含编号记录条目", bu[0], "1. [", False)
    # 占位符只给「整场辩论第一个开口的人」，而不是「每个角色的第一次发言」。
    # bull_first 下空头第一次开口时，对手其实已经讲完了 —— 这时再给它"对手尚未发言"
    # 反而是错的：它会对着一个不存在的沉默对手说话。
    check_in("2f bull_first 下空头首答不该有占位符（对手已发言）",
             be[0], NO_OPPONENT_YET, False)
    check_in("2f 而它确实拿到了多头的论点", be[0], "多头论点-ABC", True)

    fa_bf, _ = run_with({"bull": bull_reply(point="多头论点-ABC"),
                         "bear": bear_reply(point="空头论点-XYZ"),
                         "judge": judge_reply("HOLD")}, "",
                        DebateConfig(order=ORDER_BEAR_FIRST))
    check_in("2g 对照：换成 bear_first，占位符就该轮到空头",
             fa_bf.prompts("bear")[0], NO_OPPONENT_YET, True)
    check_in("2g 而多头这次首答不该有占位符",
             fa_bf.prompts("bull")[0], NO_OPPONENT_YET, False)
    check_in("2h 多头这次拿到的对手论点是空头的",
             fa_bf.prompts("bull")[0], "空头论点-XYZ", True)
    check("2i 一场辩论里占位符只出现一次",
          sum(1 for _, u in fa.seen if NO_OPPONENT_YET in u), 1)
    check_in("2j 首轮提示词仍含事实快照", bu[0], "收盘价:", True)

    # ========================================================
    section("3. 顺序置换检验 —— 把「不受发言顺序影响」从承诺变成判据")

    def first_speaker(user: str) -> str:
        """从裁决者的 user 里读出「谁先发言」。"""
        head = user.split("# 辩论发言")[-1]
        m = re.search(r"\d+\.\s*\[(多头|空头)\]", head)
        return m.group(1) if m else "?"

    def biased_judge(role, user, n):
        """模拟「先发言的一方被系统性偏袒」的裁决者。"""
        return judge_reply("BUY", "先发言方占优", 0.7) if first_speaker(user) == "多头" \
            else judge_reply("SELL", "先发言方占优", 0.3)

    def mild_judge(role, user, n):
        """有顺序依赖，但幅度在容差内。"""
        return judge_reply("HOLD", "略有偏向", 0.52 if first_speaker(user) == "多头" else 0.45)

    # 3a 对照一：顺序无关的裁决者必须被判「无关」
    fa, res = run_with(BASIC)
    check("3a 无偏裁决 → 顺序无关", res.order_sensitive, False)
    check("3a 检验确实执行了", res.order_checked, True)
    check_in("3a 说明里给出漂移量", res.order_note, "漂移", True)
    act, reason, flags = res.verdict()
    check("3a 采用裁决动作", act.value, "HOLD")
    check_true("3a 既不标失败也不标敏感",
               not flags.get("debate_failed") and not flags.get("debate_order_sensitive"))

    # 3b 对照二：有偏的裁决者必须被判「敏感」并降级
    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply(), "judge": biased_judge})
    check("3b 正序（多头先）裁决=BUY", res.judge_action, "BUY")
    check("3b 反序（空头先）裁决=SELL", res.judge_reversed_action, "SELL")
    check("3b 判为顺序敏感", res.order_sensitive, True)
    act, reason, flags = res.verdict()
    check("3b 降级为 HOLD —— 敏感结论不得给出方向", act.value, "HOLD")
    check_true("3b meta 标了 order_sensitive", flags.get("debate_order_sensitive") is True)
    check_in("3b reason 明说不可信", reason, "不可信", True)
    check_in("3b reason 同时给出两个值（正序）", reason, "正序 BUY", True)
    check_in("3b reason 同时给出两个值（反序）", reason, "反序 SELL", True)

    # 3c 动作相同但权重漂移 —— 只看动作会漏掉的那一类
    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply(),
                        "judge": lambda r, u, n: judge_reply(
                            "HOLD", "同一动作不同权重",
                            0.8 if first_speaker(u) == "多头" else 0.3)})
    check("3c 两次裁决动作都是 HOLD", (res.judge_action, res.judge_reversed_action),
          ("HOLD", "HOLD"))
    check("3c 只看动作会漏 —— 权重漂移 0.50 仍判敏感", res.order_sensitive, True)
    check("3c verdict 同样降级", res.verdict()[0].value, "HOLD")
    check_in("3c 说明指出是权重漂移", res.order_note, "权重漂移", True)

    # 3d 对照：漂移在容差内 → 判无关
    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply(), "judge": mild_judge})
    check("3d 漂移 0.07 在容差 0.20 内 → 顺序无关", res.order_sensitive, False)
    check_true("3d 对照：场景确实含顺序依赖（权重随顺序变），只是幅度可接受",
               res.judge_weight_bull != res.judge_reversed_weight_bull)

    # 3e 关掉检验 —— 「没检验」不能被当成「通过」
    fa, res = run_with(BASIC, "", DebateConfig(order_check=False))
    act, reason, flags = res.verdict()
    check("3e 未执行检验", res.order_checked, False)
    check_true("3e 必须显式标出「未经检验」",
               flags.get("debate_order_unchecked") is True)
    check("3e 未标为敏感（没证据 ≠ 有证据）", res.order_sensitive, False)

    # 3f compare_verdicts 的解析失败分支
    from core.debate import compare_verdicts
    sens, note = compare_verdicts(None, {"action": "BUY"})
    check("3f 一侧未解析时不判敏感", sens, False)
    check_in("3f 并说明这不是敏感的证据", note, "不构成", True)
    sens, note = compare_verdicts({"action": "BUY"}, {"action": "BUY"})
    check("3f 权重缺失时只比动作", sens, False)
    check_in("3f 并坦白检验力度不足", note, "权重缺失", True)

    # ========================================================
    section("4. 反序是置换，不是改写 —— 否则检验的是别的东西")

    _, res = run_with(BASIC)
    fwd = build_judge_user(SNAP, "", res.turns, reverse=False)
    rev = build_judge_user(SNAP, "", res.turns, reverse=True)
    f_seq = re.findall(r"\[(多头|空头)\]", fwd.split("# 辩论发言")[-1])
    r_seq = re.findall(r"\[(多头|空头)\]", rev.split("# 辩论发言")[-1])
    check("4a 正序角色序列", f_seq, ["多头", "空头", "多头", "空头"])
    check("4b 反序 = 正序的严格逆序", r_seq, list(reversed(f_seq)))
    check("4c 两份 prompt 长度完全相同（同一批内容，只换顺序）",
          len(fwd), len(rev))
    check_true("4d 两份 prompt 内容确实不同（对照，防「reverse 没生效」）", fwd != rev)
    check_in("4e 事实块在两份里都在记录之前（模拟器取指标取第一处）",
             fwd.split("# 辩论发言")[0], "收盘价:", True)
    check_in("4f 反序版事实块与正序一致", rev.split("# 辩论发言")[0],
             SNAP.render_text().split("\n")[0], True)

    _, rb = run_with(BASIC, "", DebateConfig(order=ORDER_BEAR_FIRST))
    check("4g bear_first 角色序列", [t.role for t in rb.turns],
          ["bear", "bull", "bear", "bull"])
    check_true("4h 两种顺序的发言角色互为置换（不是少发或多发）",
               sorted(t.role for t in res.turns) == sorted(t.role for t in rb.turns))

    try:
        order_sequence("random_idea", 2)
        check_true("4i 未知顺序必须抛错，不许静默回落", False)
    except ValueError:
        check_true("4i 未知顺序抛错，不许静默回落", True)

    # ========================================================
    section("5. 裁决失败必须显式 —— 伪装成正常判断比失败更糟")

    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply()},
                       default="这不是 JSON：我觉得还是再观察观察比较稳妥。")
    check("5a 裁决未解析", res.judge, None)
    act, reason, flags = res.verdict()
    check("5a 降级 HOLD", act.value, "HOLD")
    check_true("5a 显式标记失败", flags.get("debate_failed") is True)
    check_in("5a 说明这是「没有产出」而不是「决定观望」", reason, "没有产出", True)

    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply(),
                        "judge": json.dumps({"action": "MAYBE", "score": 55,
                                             "reason": "说不清"}, ensure_ascii=False)})
    check("5b 非法动作字段", res.judge_action, None)
    check("5b 判为失败（Action.parse 会静默吃成 HOLD，这里不能忍）", res.failed, True)
    act, reason, _ = res.verdict()
    check_in("5b 说明动作字段非法", reason, "非法", True)
    check("5b 同样降级 HOLD", act.value, "HOLD")

    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply(),
                        "judge": json.dumps({"score": 50}, ensure_ascii=False)})
    check("5c 动作字段缺失也算失败", res.failed, True)
    check_in("5c 说明里点出「缺失」", res.verdict()[1], "缺失", True)

    fa, res = run_with({"bull": bull_reply(), "bear": bear_reply(),
                        "judge": judge_reply("BUY", "采纳多头")})
    check("5d 对照：合法裁决不算失败", res.failed, False)
    check("5d 采用方向", res.verdict()[0].value, "BUY")

    # ========================================================
    section("6. 可复现与缓存 —— 多步生成最怕每次跑都不一样")

    _, r1 = run_with(BASIC)
    _, r2 = run_with(BASIC)
    check("6a 两次跑的过程指纹相同", r1.transcript_hash, r2.transcript_hash)
    check("6a 两次跑的发言数相同", len(r1.turns), len(r2.turns))
    check("6a 两次跑的调用数相同", r1.calls, r2.calls)

    fa, r3 = run_with(BASIC, "", None, memory="## 历史判断回顾\n- 上次高位追高亏了 5%")
    check_in("6b 记忆进入辩手上下文", fa.prompts("bull")[0], "上次高位追高亏了 5%", True)
    check_in("6b 记忆也进入裁决上下文", fa.prompts("judge")[0], "上次高位追高亏了 5%", True)

    _, r4 = run_with(BASIC, "", DebateConfig(max_rounds=3))
    check("6c 轮数变 → 发言数变", len(r4.turns), 6)
    check_true("6c 指纹随之变化", r4.transcript_hash != r1.transcript_hash)

    # 6d 真缓存：把 LLMAnalyzer 的最底层网络调用换成可数的假响应，
    # 这样测到的是**真的 raw_chat 缓存代码**，而不是我另写一份缓存来测自己。
    class CountingLLM(LLMAnalyzer):
        def __init__(self, cache_path):
            super().__init__(api_key="test-key", base_url="https://example.invalid/v1",
                             cache=SignalCache(path=cache_path))
            self.chat_calls = 0

        def _chat(self, system: str, user: str) -> str:
            self.chat_calls += 1
            role = mock_role_of(system)
            if role == "bull":
                return bull_reply()
            if role == "bear":
                return bear_reply()
            return judge_reply("HOLD")

    llm = CountingLLM(TMP / "llm_cache.json")
    ask = lambda s, u: llm.raw_chat(s, u)            # noqa: E731
    c1 = run_debate(ask, SNAP, "", DebateConfig())
    calls_after_first = llm.chat_calls
    check("6d 首次全部落到真实调用", calls_after_first, 6)
    check("6d 缓存条目数 = 调用数", len(llm.cache), 6)
    c2 = run_debate(ask, SNAP, "", DebateConfig())
    check("6d 第二次零真实调用（逐轮命中缓存）", llm.chat_calls, calls_after_first)
    check("6d 缓存命中 6 次", llm.stats["cache_hits"], 6)
    check("6d 两次裁决一致", c1.judge_action, c2.judge_action)
    check("6d 两次指纹一致（回测重跑零成本且可复现）",
          c1.transcript_hash, c2.transcript_hash)

    # 6e 对照：缓存键必须含 system，否则三种角色会互相污染
    before_calls, before_hits = llm.chat_calls, llm.stats["cache_hits"]
    llm.raw_chat(BULL_SYSTEM, "同一份 user 文本")
    llm.raw_chat(BEAR_SYSTEM, "同一份 user 文本")
    check("6e 同 user 不同角色 → 各自真实调用（键含 system）",
          llm.chat_calls, before_calls + 2)

    # ========================================================
    section("7. 不变式与集成")

    # 7a 辩论不得引入未来数据
    fb = make_bars([100 + i * 0.35 for i in range(100)])
    prev = fb[-1].close
    for d in _dates(3, "2031-01-06"):
        c = prev * 1.01
        fb.append(Bar(date=d, open=prev, close=c, high=c * 1.004, low=prev * 0.996,
                      volume=1e6, amount=0.0, pct_chg=0.0, turnover=0.0,
                      amplitude=0.0))
        prev = c
    i_cut = 99
    snap_f = build_snapshot("sh600519", fb, i_cut)
    check_true("7a 对照：K线里确实存在三根未来K线（否则下面这条是空转）",
               sum(1 for b in fb[i_cut + 1:] if b.date.startswith("2031")) == 3)
    check_true("7a 快照日期落在未来尾巴之前",
               snap_f.date < "2031-01-01", snap_f.date)
    fa, _ = run_with(BASIC, snap=snap_f)
    leaked = [u for _, u in fa.seen if "2031" in u]
    check("7a 任何一次发言的上下文都不得出现未来日期", len(leaked), 0)

    # 7b 回测路径只读记忆，绝不写
    mstore = MemoryStore(root=TMP / "mem")
    fa = FakeAnalyzer(BASIC)
    strat = LLMStrategy(DebateAnalyzer(fa), cache=None, prompt_version="d1",
                        memory=mstore)
    for k in range(75, len(BARS)):
        strat.decide("sh600519", BARS, k)
    check("7b 回测跑了几十根K线后，记忆库仍是空的", len(mstore.decisions()), 0)
    check_true("7b 对照：确实产生了判断（否则「没写」无从谈起）",
               strat.analyzer.stats["debates"] > 10)

    # 7c 辩论统计进体检报告
    fa2 = FakeAnalyzer(BASIC)
    strat2 = LLMStrategy(DebateAnalyzer(fa2), cache=None, prompt_version="d1")
    v = StrategyValidator(BacktestConfig(), is_ratio=0.7, min_trades_for_verdict=5)
    bt_result, rep = v.validate(strat2, {"sh600519": BARS}, None)
    db = rep.strategy_stats.get("debate")
    check_true("7c 报告里带上辩论统计", isinstance(db, dict))
    check_true("7c 辩论次数 > 0", (db or {}).get("debates", 0) > 0,
               f"{db.get('debates')}")
    check("7c 发言数 = 辩论次数 × 4", db["turns"], db["debates"] * 4)
    check("7c 调用数 = 辩论次数 × 6", db["calls"], db["debates"] * 6)
    check("7c 策略名跟着辩论走（否则两份报告分不清）", strat2.name, "debate")
    md = render_validation(rep, bt_result)
    check_in("7c 报告正文出现辩论小节", md, "多空辩论", True)
    check_in("7c 报告正文给出顺序检验结果", md, "顺序置换检验", True)

    # 7d meta 只存索引不存正文
    sig = DebateAnalyzer(FakeAnalyzer(BASIC)).analyze(SNAP)
    check_true("7d meta 不含辩论全文", "debate_transcript" not in sig.meta)
    check_true("7d meta 里有过程指纹（全文的钥匙）", "transcript_hash" in sig.meta)
    check_true("7d meta 体积受控（台账不会被撑爆）",
               len(json.dumps(sig.meta, ensure_ascii=False)) < 2500,
               f"{len(json.dumps(sig.meta, ensure_ascii=False))} 字节")
    check("7d 信号来源标记", sig.source, "debate")
    check("7d 提示词版本独立于单轮", sig.prompt_version, "d1")

    # 7e 角色标识：离线模拟器靠提示词里的关键词分流，这层耦合必须被守住
    check("7e BULL_SYSTEM → bull", mock_role_of(BULL_SYSTEM), "bull")
    check("7e BEAR_SYSTEM → bear", mock_role_of(BEAR_SYSTEM), "bear")
    check("7e JUDGE_SYSTEM → judge", mock_role_of(JUDGE_SYSTEM), "judge")

    # 7f 时点纪律必须写进提示词（模型不知道代码）
    for nm, sp in (("多头", BULL_SYSTEM), ("空头", BEAR_SYSTEM), ("裁决", JUDGE_SYSTEM)):
        check_in(f"7f {nm}提示词含时点纪律", sp, "时点纪律", True)
    check_in("7f 裁决提示词明令不受发言顺序影响", JUDGE_SYSTEM, "不受发言顺序影响", True)
    check_in("7f 裁决提示词允许 HOLD（不硬造方向）", JUDGE_SYSTEM, "硬造", True)
    check_in("7f 多头提示词写明认输不改变轮数", BULL_SYSTEM, "不改变辩论轮数", True)

    # 7g 分析器自报统计与归属标记
    da = DebateAnalyzer(FakeAnalyzer(BASIC))
    da.analyze(SNAP)
    check("7g stats_scope 标记（validator 靠它认领）", da.stats_scope, "debate")
    check("7g debates", da.stats["debates"], 1)
    check("7g turns", da.stats["turns"], 4)
    check("7g calls", da.stats["calls"], 6)
    check("7g order_checked", da.stats["order_checked"], 1)
    check("7g 无失败无敏感 → flagged=0", da.stats["flagged"], 0)
    audit = render_debate_audit(da, [da.last])
    check_true("7g 审计行非空", len(audit) >= 3, f"{len(audit)} 行")
    check_in("7g 审计行给出两个顺序的裁决", "\n".join(audit), "正序", True)

    # 7h 落盘：指纹去重 + 全文可取回 + 位置可覆盖
    os.environ["DSA_DEBATE_DIR"] = str(TMP / "deb")
    check("7h 默认根目录跟随 DSA_DEBATE_DIR", str(default_root()), str(TMP / "deb"))
    store = TranscriptStore()
    _, r5 = run_with(BASIC)
    check("7h 首次写入成功", store.append(r5), True)
    check("7h 同指纹不重复写（过程指纹就是主键）", store.append(r5), False)
    recs = store.read("sh600519")
    check("7h 可取回 1 条", len(recs), 1)
    check("7h 落盘记录含四段发言全文", len(recs[0].get("transcript") or []), 4)
    check("7h 落盘记录含两次裁决", recs[0].get("judge_reversed") is not None, True)
    check("7h 落盘记录带过程指纹", recs[0].get("transcript_hash"), r5.transcript_hash)
    del os.environ["DSA_DEBATE_DIR"]

finally:
    shutil.rmtree(TMP, ignore_errors=True)

# ============================================================

print()
print("=" * 60)
print(f"多空辩论验收：{PASS}/{PASS + FAIL} 通过" +
      ("，全部通过 ✅" if FAIL == 0 else f"，失败 {FAIL} 项 ❌"))
print("=" * 60)
sys.exit(1 if FAIL else 0)
