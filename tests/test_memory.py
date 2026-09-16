#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反思记忆验收 —— 重点是证明「记忆不漏未来」，不是证明「记忆能用」
====================================================================

记忆是这套系统里唯一**跨时间**的组件，也是最容易引入未来函数的组件：
一条"事后复盘写下的教训"，本身就是用决策之后的信息生产出来的。
所以这个测试文件的重心不是"记忆功能是否正常"，而是：

    这条教训，在它不该被看见的那一天，是不是真的看不见？

写法沿用 test_lookahead.py 的两条规矩：
  1. 先构造一个**真的含泄漏**的场景，再验证门控拦住了它
  2. 每个判据都配**对照组** —— 证明这个场景不是空转
     （比如"扣留时不能泄露数字"，必须同时证明"不扣留时那个数字真的会出现"，
      否则第一条断言可能只是因为它压根就没这个数字）

纯离线，零网络，零依赖。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.fetchers import Bar                                         # noqa: E402
from core.llm import (PROMPT_VERSION, SYSTEM_PROMPT, build_prompt)     # noqa: E402
from core.memory import (MIN_SAMPLE_FOR_PATTERN, DecisionRecord,      # noqa: E402
                         LessonRecord, MemoryStore, Outcome, classify)
from core.signals import Action, Signal                                # noqa: E402

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


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ============================================================
# 造数据：确定性K线（跳周末），价格完全由入参决定
# ============================================================

def _dates(n: int, start: str = "2025-01-02") -> list[str]:
    """生成 n 个交易日日期（跳过周六周日）。"""
    out, d = [], datetime.strptime(start, "%Y-%m-%d")
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def make_bars(prices: list, start: str = "2025-01-02",
              wick: float = 0.001, lows: list | None = None,
              highs: list | None = None) -> list:
    """
    按给定收盘价序列造K线。

    lows/highs 可覆盖 —— 测「止损被扫后行情回来」必须能精确控制影线，
    否则那条分支永远跑不到。
    """
    ds = _dates(len(prices), start)
    bars = []
    prev = prices[0]
    for i, (d, c) in enumerate(zip(ds, prices)):
        o = prev
        h = highs[i] if highs and highs[i] is not None else max(o, c) * (1 + wick)
        low = lows[i] if lows and lows[i] is not None else min(o, c) * (1 - wick)
        bars.append(Bar(date=d, open=o, close=c, high=h, low=low,
                        volume=1e6, amount=0.0, pct_chg=0.0,
                        turnover=0.0, amplitude=0.0))
        prev = c
    return bars


def up_prices(n: int = 60, step: float = 0.6, base: float = 100.0) -> list:
    """稳定上涨 —— 用来构造"判断成立"的对照场景。"""
    return [round(base + i * step, 2) for i in range(n)]


def down_prices(n: int = 45, step: float = 0.6, base: float = 100.0) -> list:
    """
    稳定下跌 —— 用来构造"每一条都在跌"的场景。

    为什么需要它：如果用一个先平后跌的序列，晚一点的判断会因为
    入场时价格已经跌到底而得到 0% 收益，落进"噪声区间"分类。
    那样统计出来的就不是 5 条同类错误，测试也就测不到想测的东西 ——
    样本必须真的同质，规律检验才有意义。
    """
    return [round(base - i * step, 2) for i in range(n)]


class StoreFixture:
    """临时记忆库，用完清理。"""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="dsa_mem_"))
        self.store = MemoryStore(self.dir)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def mk_decision(store: MemoryStore, symbol: str, trade_date: str,
                action: str = "BUY", horizon: int = 5,
                stop: float | None = None, target: float | None = None,
                rsi: float | None = None, pos: float | None = None,
                source: str = "llm", prompt_version: str = "v3") -> DecisionRecord:
    """
    直接写一条判断记录。

    刻意不走 LLMAnalyzer：这个文件测的是记忆的时间语义，
    不该被模型调用的任何不确定性干扰。
    """
    sig = Signal(date=trade_date, symbol=symbol, action=Action.parse(action),
                 score=70.0, confidence=0.7, entry=None, stop=stop,
                 target=target, horizon_days=horizon, source=source,
                 model="test-model", prompt_version=prompt_version, reason="测试")
    facts = {}
    if rsi is not None:
        facts["RSI14"] = rsi
    if pos is not None:
        facts["60日区间位置"] = pos
    return store.record_decision(sig, facts=facts, visible_lessons=[])


# ============================================================
# 1. 门控：教训必须等到「可知日」才可见（带对照组）
# ============================================================

def test_gate_visibility() -> None:
    section("1. 时点门控：教训在「可知日」之前必须不可见")
    fx = StoreFixture()
    try:
        prices = up_prices(40)
        bars = make_bars(prices)
        # 先验证日期生成本身没错（否则后面所有断言都建立在错误的日期上）
        check("交易日序列跳过周末 [0]", bars[0].date, "2025-01-02")
        check("交易日序列跳过周末 [1]", bars[1].date, "2025-01-03")
        check("交易日序列跳过周末 [2]", bars[2].date, "2025-01-06")

        mk_decision(fx.store, "sz300750", bars[0].date, horizon=5)
        learned = bars[5].date
        check("可知日 = 决策日后第 horizon 个交易日", learned, "2025-01-09")

        fx.store.resolve({"sz300750": bars})

        # ---- 实验组：可知日之前 ----
        before = fx.store.get_past_context("sz300750", bars[3].date)
        check("可知日之前：可见教训数", len(before.lessons), 0)
        # 对照组：证明这 1 条教训确实存在，只是被时点挡住了（不是压根没有）
        check("可知日之前：被扣留计数（对照组）", before.blocked, 1)
        check("可知日之前：扣留的可知日", before.blocked_until[0], learned)

        # ---- 对照组：可知日当天起可见 ----
        at = fx.store.get_past_context("sz300750", bars[4].date)
        check("可知日前一天仍不可见（边界）", len(at.lessons), 0)
        on = fx.store.get_past_context("sz300750", bars[5].date)
        check("可知日当天即可见（边界）", len(on.lessons), 1)
        check("可知日当天：无扣留", on.blocked, 0)
    finally:
        fx.close()


# ============================================================
# 2. 当天的判断不进当天的上下文
# ============================================================

def test_self_exclusion() -> None:
    section("2. 判断不进自己的上下文（严格早于）")
    fx = StoreFixture()
    try:
        prices = up_prices(30)
        bars = make_bars(prices)
        d0 = mk_decision(fx.store, "sh600519", bars[0].date, horizon=5)

        same_day = fx.store.get_past_context("sh600519", bars[0].date)
        check("同一天：该判断不可见（否则它会拿自己当先例）", len(same_day.decisions), 0)

        next_day = fx.store.get_past_context("sh600519", bars[1].date)
        check("次日：该判断可见（对照组）", len(next_day.decisions), 1)
        check("次日：可见的正是同一条", next_day.decisions[0].decision_id, d0.decision_id)
    finally:
        fx.close()


# ============================================================
# 3. 不进早结算
# ============================================================

def test_no_early_resolution() -> None:
    section("3. 验证期没走完，不许结算")
    fx = StoreFixture()
    try:
        prices = up_prices(40)
        bars = make_bars(prices)
        mk_decision(fx.store, "sz000858", bars[0].date, horizon=10)

        # 只给到第 5 根 —— 距离 10 日验证期还差一半
        got = fx.store.resolve({"sz000858": bars[:6]})
        check("窗口未走完：不产出教训", len(got), 0)
        check("窗口未走完：判断仍为待结算", len(fx.store.open_decisions()), 1)

        # 用 as_of 卡住结算同样不行（结果窗口延伸到 as_of 之后）
        got2 = fx.store.resolve({"sz000858": bars}, as_of=bars[8].date)
        check("as_of 卡在窗口中段：仍不结算", len(got2), 0)

        # 对照组：走完就能结算
        got3 = fx.store.resolve({"sz000858": bars})
        check("窗口走完：结算出 1 条（对照组）", len(got3), 1)
        check("结算后可追溯的可知日", got3[0].learned_date, bars[10].date)
    finally:
        fx.close()


# ============================================================
# 4. 日期戳诚实：可知日 ≠ 写入日
# ============================================================

def test_learned_date_vs_created_on() -> None:
    section("4. 日期戳是「可知日」而不是「写入日」")
    fx = StoreFixture()
    try:
        prices = [100, 100, 100, 100, 100, 118.3] + [118.3] * 20
        bars = make_bars(prices)
        mk_decision(fx.store, "sh601318", bars[0].date, horizon=5)

        got = fx.store.resolve({"sh601318": bars})
        l = got[0]
        today = datetime.now().strftime("%Y-%m-%d")

        check("ret 与价格一致", round(l.ret_pct, 1), 18.3, tol=0.05)
        check("写入日 = 今天（这条教训是现在才写下的）", l.created_on, today)
        check("可知日 = 2025-01-09（远早于写入日）", l.learned_date, "2025-01-09")
        check("可知日 != 写入日（这是整套门控成立的前提）",
              l.learned_date != l.created_on, True)

        # 核心性质：今天写下的教训，在三个月前的时点就该可见 ——
        # 因为它的信息窗口整段落在那边。
        old = fx.store.get_past_context("sh601318", "2025-01-10")
        check("今天写的教训在 2025-01-10 时点可见", len(old.lessons), 1)
        check("  且内容确实是那条 +18.3%", "+18.3%" in old.render(), True)

        # 而早一天就不行
        earlier = fx.store.get_past_context("sh601318", "2025-01-08")
        check("2025-01-08 时点不可见（边界另一侧）", len(earlier.lessons), 0)
    finally:
        fx.close()


# ============================================================
# 5. 扣留必须发声，且不得泄露结果数字
# ============================================================

def test_withheld_must_speak_no_leak() -> None:
    section("5. 扣留要发声，且不许泄露结果")
    fx = StoreFixture()
    try:
        prices = [100, 100, 100, 100, 100, 118.3] + [118.3] * 20
        bars = make_bars(prices)
        mk_decision(fx.store, "sh600036", bars[0].date, horizon=5)
        fx.store.resolve({"sh600036": bars})

        # 可知日（2025-01-09）之前，站在 2025-01-06 看
        ctx = fx.store.get_past_context("sh600036", bars[2].date)
        text = ctx.render()
        check("扣留计数不为零", ctx.blocked, 1)
        check("输出了扣留说明", "还不知道" in text or "扣留" in text, True)
        check("明确要求不许当成失误", "不要当成失误" in text, True)
        # ⚠️ 关键断言：不能泄露结果数字，否则门控等于没做
        check("不泄露被扣留结果的数字", "18.3" not in text, True)
        check("不泄露被扣留结果的文本", "判断成立" not in text, True)

        # ---- 对照组：证明那个数字本来是存在的，第一条断言不是空转 ----
        after = fx.store.get_past_context("sh600036", "2025-01-09").render()
        check("对照组：过了可知日就真的会出现 18.3", "18.3" in after, True)
        check("对照组：过了可知日就真的会出现结论文本", "判断成立" in after, True)
        # 反向对照：有教训时开场白才说"及已结算结果"，并给出使用纪律
        check("对照组：有教训时说「及已结算结果」", "及已结算结果" in after, True)
        check("对照组：有教训时摆出「使用纪律」", "使用纪律" in after, True)

        # 扣留说明不能出现在没有扣留的时候
        check("无扣留时不出现扣留段", "扣留" not in after, True)
    finally:
        fx.close()


# ============================================================
# 6. 未结算的判断不得被当成失误
# ============================================================

def test_open_is_not_a_failure() -> None:
    section("6. 验证期内的判断不得呈现为失误")
    fx = StoreFixture()
    try:
        prices = up_prices(40)
        bars = make_bars(prices)
        mk_decision(fx.store, "sz300750", bars[0].date, horizon=10)
        # 不结算，直接看
        ctx = fx.store.get_past_context("sz300750", bars[5].date)
        text = ctx.render()
        check("计入验证期内", ctx.pending, 1)
        check("可见教训仍为 0", len(ctx.lessons), 0)
        check("提到「验证期」", "验证期" in text, True)
        check("不出现「未成立」这种判决（对照组性质）", "未成立" not in text, True)
        # 开场白必须与事实相符：没有已结算教训时不能写"及已结算结果"
        check("没有教训时说「尚无已结算结果」", "尚无已结算结果" in text, True)
        check("没有教训时不写「及已结算结果」", "及已结算结果" in text, False)
        check("没有教训时不摆「使用纪律」小节", "使用纪律" in text, False)
        check("改为提示「都还没有结果」", "都还没有结果" in text, True)
    finally:
        fx.close()


# ============================================================
# 7. 错误归因（纯函数，逐类钉死）
# ============================================================

def _d(action="BUY", horizon=5, stop=95.0, target=110.0, rsi=None, pos=None):
    return DecisionRecord(decision_id="x", symbol="T", trade_date="2025-01-02",
                          action=action, horizon_days=horizon, stop=stop,
                          target=target, facts_gist={"rsi14": rsi, "pos_in_60d": pos})


def _o(ret, mfe=0.0, mae=0.0, stop_hit=False, stop_first=None, action="BUY"):
    return Outcome(bars_used=5, ret_pct=ret, mfe_pct=mfe, mae_pct=mae,
                   agent_ret_pct=(ret if action == "BUY" else -ret),
                   stop_hit=stop_hit, stop_first=stop_first)


def test_error_taxonomy() -> None:
    section("7. 错误归因（每条带一个反向对照）")
    cases = [
        ("成立", _d(), _o(5.0, mfe=6.0, mae=-1.0), "hit", "right_call"),
        ("高位追高", _d(rsi=78.0, pos=92.0), _o(-4.2, mfe=0.5, mae=-5.1), "miss", "chased_high"),
        ("入场即套", _d(rsi=55.0), _o(-4.0, mfe=1.0, mae=-5.5), "miss", "false_breakout"),
        ("止损被扫后回来", _d(), _o(3.0, mfe=5.0, mae=-6.0, stop_hit=True, stop_first=True),
         "miss", "stop_shaken"),
        ("方向看错", _d(rsi=55.0), _o(-4.0, mfe=1.5, mae=-3.0), "miss", "wrong_direction"),
        ("噪声区间", _d(), _o(1.0, mfe=1.5, mae=-1.0), "flat", "flat"),
        ("卖飞", _d(action="SELL"), _o(4.0, mfe=5.0, mae=-0.5, action="SELL"),
         "miss", "sold_too_early"),
        ("离场正确", _d(action="SELL"), _o(-4.0, mfe=0.5, mae=-5.0, action="SELL"),
         "hit", "right_call"),
        ("过度保守", _d(action="HOLD", stop=None, target=None), _o(6.0, mfe=7.0, mae=-0.5),
         "miss", "missed_opportunity"),
        ("回避正确", _d(action="AVOID", stop=None, target=None), _o(-6.0, mfe=0.5, mae=-7.0),
         "hit", "correct_avoid"),
        ("观望落在噪声区", _d(action="HOLD", stop=None, target=None), _o(1.0, mfe=1.2, mae=-1.0),
         "flat", "flat"),
    ]
    for label, d, o, want_v, want_e in cases:
        v, e, text = classify(d, o)
        check(f"{label} → 判定", v, want_v)
        check(f"{label} → 归因", e, want_e)
        check(f"{label} → 教训非空", len(text) > 8, True)

    # 关键对照：同样是亏，有高位特征 / 无高位特征必须分到不同桶，
    # 否则"高位追进"这个画像永远统计不出来。
    _, e_high, _ = classify(_d(rsi=80.0), _o(-4.0, mfe=0.5, mae=-5.1))
    _, e_low, _ = classify(_d(rsi=45.0), _o(-4.0, mfe=0.5, mae=-3.0))
    check("高位与低位亏损失败必须分开归因", e_high != e_low, True)


# ============================================================
# 7b. 非买入方向的记账口径
# ============================================================

def test_direction_accounting() -> None:
    section("7b. 记账口径：买入记收益，其余记「不做多的价值」")
    fx = StoreFixture()
    try:
        # 观望/回避用的是 ±5% 带宽，比持仓的 ±2% 宽 ——
        # 所以这里的跌幅必须足够陡（-7.5%），否则会正确地落进「无信息」。
        dn = make_bars(down_prices(30, step=1.5))
        up = make_bars(up_prices(30))

        mk_decision(fx.store, "BUY_T", dn[0].date, action="BUY", horizon=5)
        mk_decision(fx.store, "HOLD_T", dn[0].date, action="HOLD", horizon=5)
        mk_decision(fx.store, "SELL_T", up[0].date, action="SELL", horizon=5)
        fx.store.resolve({"BUY_T": dn, "HOLD_T": dn, "SELL_T": up})

        ls = {l.symbol: l for l in fx.store.lessons()}
        check("三条都已结算", len(ls), 3)

        # 下跌行情
        b, h = ls["BUY_T"], ls["HOLD_T"]
        check("买入（跌）：agent_ret 与区间收益同号", b.agent_ret_pct < 0, True)
        check("买入：agent_ret == ret", round(b.agent_ret_pct, 3), round(b.ret_pct, 3))
        # 关键：观望必须是 -ret，不能是 0 —— 否则「回避正确」那一行的
        # 平均影响永远是 0.00%，统计列变成装饰
        check("观望（跌 7.5%）：agent_ret 为正（不做多是赚的）", h.agent_ret_pct > 0, True)
        check("观望：agent_ret == -ret", round(h.agent_ret_pct, 3), round(-h.ret_pct, 3))
        check("观望归因为回避正确", h.error_type, "correct_avoid")
        check("  且不是零影响（旧口径的 bug 就长这样）", h.agent_ret_pct != 0.0, True)

        # 卖出：上涨行情下的"卖飞"
        s = ls["SELL_T"]
        check("卖出（涨）：agent_ret 为负（不做多是亏的）", s.agent_ret_pct < 0, True)
        check("卖出归因为卖飞", s.error_type, "sold_too_early")

        # 观望的带宽：小幅下跌不该被算成"回避正确"，否则事后怎么看都对
        v_flat, e_flat, _ = classify(_d(action="HOLD", stop=None, target=None),
                                     _o(-3.0, mfe=0.5, mae=-3.5))
        check("观望遇 -3%：判为无信息（不吹成回避正确）", (v_flat, e_flat), ("flat", "flat"))
    finally:
        fx.close()


# ============================================================
# 8. 样本不足不给出「规律」
# ============================================================

def test_pattern_sample_guard() -> None:
    section("8. 样本不足不列为规律")
    fx = StoreFixture()
    try:
        bars = make_bars(down_prices(45))       # 每一条判断后面都在跌

        # 造 3 条同类错误（< MIN_SAMPLE）
        for k in range(3):
            mk_decision(fx.store, "T", bars[k].date, horizon=5, rsi=78.0, pos=92.0)
        fx.store.resolve({"T": bars})
        ctx3 = fx.store.get_past_context("T", bars[-1].date)
        check("已结算 3 条", len(ctx3.lessons), 3)
        check(f"3 条 < {MIN_SAMPLE_FOR_PATTERN}：不列规律", len(ctx3.patterns), 0)

        # 再补 2 条凑到 5
        for k in range(3, 5):
            mk_decision(fx.store, "T", bars[k].date, horizon=5, rsi=78.0, pos=92.0)
        fx.store.resolve({"T": bars})
        ctx5 = fx.store.get_past_context("T", bars[-1].date)
        check("已结算 5 条", len(ctx5.lessons), 5)
        check(f"5 条 ≥ {MIN_SAMPLE_FOR_PATTERN}：列出规律（对照组）", len(ctx5.patterns), 1)
        check("规律类型正确", ctx5.patterns[0][0], "chased_high")
        check("规律计数正确", ctx5.patterns[0][1], 5)
    finally:
        fx.close()


# ============================================================
# 9. 确定性与容错
# ============================================================

def test_determinism_and_robustness() -> None:
    section("9. 可复现性与坏行容错")
    fx = StoreFixture()
    try:
        bars = make_bars(down_prices(45))
        for k in range(6):
            mk_decision(fx.store, "T", bars[k].date, horizon=5, rsi=78.0, pos=90.0)
        fx.store.resolve({"T": bars})

        a = fx.store.get_past_context("T", bars[-1].date).render()
        b = fx.store.get_past_context("T", bars[-1].date).render()
        # 不确定的渲染顺序会让 prompt 哈希抖动 → 缓存全废、回测不可复现
        check("两次渲染完全一致", a == b, True)

        # 坏行：半截写入或手工编辑都会留下。不许炸，也不许静默吞掉。
        n_before = len(fx.store.decisions())
        with open(fx.store.decisions_path, "a", encoding="utf-8") as f:
            f.write("{这行JSON是坏的\n")
        check("坏行不影响已解析条数", len(fx.store.decisions()), n_before)
        check("坏行不影响渲染", len(fx.store.get_past_context("T", bars[-1].date).render()) > 0, True)
    finally:
        fx.close()


# ============================================================
# 10. LLM 层：记忆进提示词 + 提示词版本
# ============================================================

class FakeSnap:
    symbol = "sz300750"
    date = "2025-06-20"
    close = 200.0

    def render_text(self) -> str:
        return "收盘价: 200.0\nMA20: 195.0"

    def to_prompt_dict(self) -> dict:
        return {"收盘价": 200.0}


def test_prompt_carries_memory() -> None:
    section("10. 提示词携带记忆（升版 v3）")
    check("PROMPT_VERSION 已升到 v3（改提示词必须升版）", PROMPT_VERSION, "v3")
    check("系统提示词含记忆使用纪律", "历史判断回顾的使用规则" in SYSTEM_PROMPT, True)
    check("系统提示词要求可推翻但须说理由", "可以推翻它" in SYSTEM_PROMPT, True)
    check("系统提示词禁止把验证期判断当失误",
          "正在验证期内的判断不是失误" in SYSTEM_PROMPT, True)

    snap = FakeSnap()
    plain = build_prompt(snap.symbol, snap)
    # 用块首标记判定，不用裸词 —— 系统提示词的规则说明里本来就会提到
    # 「历史判断回顾」这几个字，拿裸词当判据会永远为真（第一版就踩了这个坑）。
    check("不给记忆时不出现记忆段", "# 历史判断回顾" in plain, False)

    block = "# 历史判断回顾（时点：2025-06-20）\n\n测试用教训正文 XXX"
    with_mem = build_prompt(snap.symbol, snap, memory_block=block)
    check("给记忆时出现记忆段", "# 历史判断回顾" in with_mem, True)
    check("记忆正文被完整带入", "测试用教训正文 XXX" in with_mem, True)
    check("记忆改变了提示词（缓存因此自动失效）", plain != with_mem, True)
    check("空记忆段等于没有记忆",
          build_prompt(snap.symbol, snap, memory_block="   ") == plain, True)

    # 时点纪律不能因为加了记忆就失效：提示词里不该出现真实今天
    today = datetime.now().strftime("%Y-%m-%d")
    check("提示词不出现真实今天", today not in with_mem, True)
    check("提示词锚定在快照日期", "2025-06-20" in with_mem, True)


# ============================================================
# 11. 策略层：逐根门控 + 回测只读不写
# ============================================================

class RecordingAnalyzer:
    """记录每次调用收到的 memory_block，用来验证逐根门控。"""

    model = "recording"

    def __init__(self):
        self.blocks: list[str] = []

    def analyze(self, snap, position=None, holding=False, memory_block=""):
        self.blocks.append(memory_block or "")
        return Signal(date=snap.date, symbol=snap.symbol, action=Action.HOLD,
                      score=50.0, source="llm", model=self.model)


def test_strategy_per_bar_gating() -> None:
    section("11. 策略层逐根门控，且回测只读不写")
    from core.strategies import LLMStrategy
    fx = StoreFixture()
    try:
        prices = up_prices(60)
        bars = make_bars(prices)
        mk_decision(fx.store, "sz300750", bars[0].date, horizon=5)
        fx.store.resolve({"sz300750": bars})
        learned = bars[5].date

        n_dec_before = len(fx.store.decisions())
        analyzer = RecordingAnalyzer()
        st = LLMStrategy(analyzer, prompt_version="v3", memory=fx.store)

        st.decide("sz300750", bars, 2)     # 可知日之前
        st.decide("sz300750", bars, 5)     # 可知日当天
        st.decide("sz300750", bars, 6)     # 可知日之后

        check("第 2 根：不注入教训正文", "判断成立" in analyzer.blocks[0], False)
        check("第 2 根：但仍说明在验证中", "验证" in analyzer.blocks[0], True)
        check("第 5 根（可知日）：注入教训", "判断成立" in analyzer.blocks[1], True)
        check("第 6 根：持续注入", "判断成立" in analyzer.blocks[2], True)

        check("统计到注入次数", st.stats["memory_injected"], 3)
        check("统计到扣留次数", st.stats["blocked"], 1)
        check("统计到可见教训数", st.stats["lessons_shown"], 2)

        # ⭐ 回测绝不能写记忆：写了就污染库，之后拿它回测变成自我循环
        check("回测未向记忆库写入任何判断", len(fx.store.decisions()), n_dec_before)
        check("可知日与策略判定一致", learned, "2025-01-09")
    finally:
        fx.close()


# ============================================================

def main() -> int:
    print("dsa-lite 反思记忆验收 —— 重点是证明它不漏未来")
    test_gate_visibility()
    test_self_exclusion()
    test_no_early_resolution()
    test_learned_date_vs_created_on()
    test_withheld_must_speak_no_leak()
    test_open_is_not_a_failure()
    test_error_taxonomy()
    test_direction_accounting()
    test_pattern_sample_guard()
    test_determinism_and_robustness()
    test_prompt_carries_memory()
    test_strategy_per_bar_gating()
    total = PASS + FAIL
    print(f"\n{'=' * 66}")
    print(f"反思记忆验收：{PASS}/{total} 通过"
          + (f"，失败 {FAIL}" if FAIL else "，全部通过 ✅"))
    print("=" * 66)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
