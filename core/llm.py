#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 分析层 —— 把模型包成可回测的信号源
============================================================
这里只干一件事：给模型一份「事实快照」，要回一个 JSON 信号。

三条防线，对应 LLM 炒股最容易翻车的三个地方：

1. 幻觉 → 只喂算好的指标（Snapshot），不喂原始K线让它自己算。
   提示词里明确写「字段没给就不要推断」，并禁止它引用训练数据里的
   "公司基本面"（回测时那等于未来函数）。

2. 格式失控 → 要求纯 JSON，解析失败一律降级为 HOLD。
   宁可少交易，也不能让脏数据污染回测结果。

3. 成本与复现 → 按 prompt 哈希缓存。同一份提示词只问一次，
   重跑回测零成本，且结果完全可复现 —— 没有缓存就没法做参数调优。

支持任何 OpenAI 兼容接口：DeepSeek / 通义 / Ollama 本地 / Gemini 兼容端点。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

from .signals import Action, Signal, SignalCache

log = logging.getLogger("llm")

# ------------------------------------------------------------
# 提示词模板（prompt_version 变了就必须重新回测）
# ------------------------------------------------------------

# v2 起加入「时点纪律」段。v3 起加入「历史判断回顾」（反思记忆）槽位。
# 改提示词必须同步升版本号 ——
# 缓存按 prompt 内容哈希，但版本号是台账里唯一的可追溯线索：
# 不升版，三个月后没人说得清那次回测用的是哪一版 prompt。
PROMPT_VERSION = "v3"

# 多空辩论是**另一套**提示词（三个角色各自的 system），且决策链路完全不同，
# 所以给它独立的版本号，而不是把 v3 往上堆。
# 这样台账能直接区分「单轮判断」与「辩论判断」，两者的回测结论不能混着比。
DEBATE_PROMPT_VERSION = "d1"

SYSTEM_PROMPT = """你是一名严格的A股短线交易分析师。你只能基于用户提供的量化指标做判断，禁止引用任何外部信息。

时点纪律（比下面所有规则都优先）：
你只活在用户给出的那个快照日期。你不知道那个日期之后发生的任何事。
- 禁止引用你在训练数据里关于这段时期的记忆（"后来它涨了""这家公司后来出事了"）
- 禁止假设后续的行情、政策、业绩、公告
- 如果某个结论只能靠"事后已知的结果"才能得出，那就不要给出这个结论 —— 宁可 HOLD

历史判断回顾的使用规则（如果提供了这一段）：
那里面是**你自己**过去的判断和已结算结果，用途只有一个：别重复同一个错误。
- 它是参考，不是命令。如果你认为本次情况确实不同，可以推翻它，但必须在 reason 里写明理由
- 只被"已结算"的结果影响。正在验证期内的判断不是失误，不要为了纠正它而反向操作
- 历史模式不构成对本次走势的预知。它说明过去，不说明这一次
- 如果本次事实与历史教训冲突，以本次事实为准

硬性规则：
1. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。
2. 严禁引用公司基本面、财报、新闻、传闻——你没有这些数据，编造即为严重错误。
3. 如果指标显示多空矛盾或信息不足，必须给 HOLD 或 AVOID，不要勉强给方向。
4. score 是 0-100 的多空强度：0=极度看空，50=中性，100=极度看多。
5. 给 BUY 时必须同时给出 stop（止损价）和 target（目标价），且 stop < 当前价 < target。
6. confidence 是你对本次判断的把握程度 0-1。不确定就调低，不要一律给 0.8。

输出 JSON 结构：
{"action":"BUY|SELL|HOLD|AVOID","score":0-100,"confidence":0-1,"stop":数字或null,"target":数字或null,"horizon_days":整数,"reason":"不超过40字的中文理由"}"""

USER_TEMPLATE = """标的：{symbol}
以下是指标快照（截至 {date} 收盘，均为已发生事实）：

{facts}

当前状态：{position_desc}
{memory}
现在是 {date} 收盘之后，{date} 之后发生的事你一无所知。
请判断下一个交易日之后的操作方向。记住：只输出 JSON。"""


def build_prompt(symbol: str, snap, holding: bool = False,
                 pnl_pct: Optional[float] = None,
                 memory_block: str = "") -> str:
    """
    构造完整 prompt（system + user 拼接后用于缓存哈希）。

    memory_block 是「历史判断回顾」。它天然被缓存哈希覆盖 ——
    记忆一变，哈希就变，缓存自动失效。这一点很重要：
    否则加了记忆却命中旧缓存，等于加了没生效，而且完全看不出来。
    """
    pos_desc = "空仓"
    if holding:
        pos_desc = f"已持仓，当前浮动盈亏 {pnl_pct:+.2f}%" if pnl_pct is not None else "已持仓"
    mem = f"\n{memory_block.strip()}\n" if memory_block and memory_block.strip() else ""
    return (SYSTEM_PROMPT + "\n---\n" +
            USER_TEMPLATE.format(symbol=symbol, date=snap.date,
                                 facts=snap.render_text(), position_desc=pos_desc,
                                 memory=mem))


# ------------------------------------------------------------
# 解析
# ------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """
    从一段文本里抠出 JSON 对象。容忍三种常见脏格式：
    裸 JSON / ```json 包裹 / 前后有废话。

    抽成独立函数是因为多空辩论的每一轮发言也要走这里。
    解析器必须全项目只有一份实现 —— 否则「解析失败就降级」这条规则
    会在两条路径上各自演化，最后变成两种行为。
    """
    if not text:
        return None
    t = text.strip()

    # 去掉 markdown 代码块
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.S)
    if m:
        t = m.group(1).strip()

    # 直接解析
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # 退而求其次：找第一个平衡的 {...}
    start = t.find("{")
    if start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
    log.warning("无法解析模型输出: %s", text[:200])
    return None


def parse_signal(text: str, symbol: str, date: str, score: float = 50.0) -> Optional[dict]:
    """解析单轮分析器的输出。语义等同 extract_json，保留原签名以免调用方改动。"""
    return extract_json(text)


def dict_to_signal(d: dict, symbol: str, date: str, model: str,
                   prompt_version: str, source: str = "llm",
                   ref_price: Optional[float] = None) -> Signal:
    """
    把模型 JSON 转成 Signal，字段缺失/越界一律夹到安全范围。

    ref_price 是信号生成日的收盘价，写入 entry 字段仅作**参考基准**，
    用来算盈亏比、算距止损的百分比 —— 它绝不是成交价。
    回测引擎一律按 T+1 开盘价成交，模型无权指定价格。
    """
    action = Action.parse(d.get("action"))
    score = _num(d.get("score"), 50.0)
    score = max(0.0, min(100.0, score))
    conf = _num(d.get("confidence"), 0.5)
    conf = max(0.0, min(1.0, conf))
    stop, target = _num(d.get("stop")), _num(d.get("target"))
    horizon = int(_num(d.get("horizon_days"), 10) or 10)
    horizon = max(1, min(120, horizon))

    # 只有买入信号才谈止损/目标。A股不能做空，卖出信号是"离场"而非"开仓"，
    # 给它挂止损目标只会让报告看起来专业、实际全是噪声。
    if action != Action.BUY:
        stop = target = None
    elif stop is not None and target is not None and stop >= target:
        log.warning("%s %s 的 stop(%s) >= target(%s)，丢弃价位", symbol, date, stop, target)
        stop = target = None

    return Signal(
        date=date, symbol=symbol, action=action, score=score, confidence=conf,
        entry=ref_price, stop=stop, target=target, horizon_days=horizon,
        source=source, model=model, prompt_version=prompt_version,
        reason=str(d.get("reason", ""))[:200],
        meta={"raw": d, "entry_is_reference": True})


# ------------------------------------------------------------
# 分析器
# ------------------------------------------------------------

class LLMAnalyzer:
    """
    OpenAI 兼容接口的分析器。DeepSeek / 通义 / Ollama 本地都能接。

    环境变量：
        LLM_BASE_URL   默认 https://api.deepseek.com/v1
        LLM_API_KEY    云端必填；本地端点（localhost）可留空
        LLM_MODEL      默认 deepseek-chat
        LLM_MAX_TOKENS 可选。本地模型强烈建议设，否则小模型容易啰嗦到跑不动
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, temperature: float = 0.2,
                 timeout: Optional[int] = None, retries: int = 2,
                 max_tokens: Optional[int] = None,
                 cache: Optional[SignalCache] = None,
                 use_cache: bool = True):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://api.deepseek.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.temperature = temperature
        self.retries = retries
        # 本地模型要两样东西：不要 Key，以及一个足够长的超时。
        # 冷启动要把权重读进内存，第一次调用比热态慢好几倍。
        self.timeout = timeout or (240 if _is_local_url(self.base_url) else 60)
        # 本地小模型没有 provider 侧的 output 上限，不掐着点能自己跑个一两分钟
        mt = max_tokens if max_tokens is not None else os.environ.get("LLM_MAX_TOKENS")
        self.max_tokens = int(mt) if mt else None
        self.cache = cache if cache is not None else SignalCache()
        self.use_cache = use_cache
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0}

    @property
    def available(self) -> bool:
        # 本地端点不需要 API Key（Ollama 的 OpenAI 兼容层不校验它）
        return bool(self.api_key) or _is_local_url(self.base_url)

    def analyze(self, snap, position=None, holding: bool = False,
                memory_block: str = "") -> Signal:
        prompt = build_prompt(
            snap.symbol, snap, holding=holding,
            pnl_pct=(position or {}).get("pnl_pct") if position else None,
            memory_block=memory_block)

        # 1) 缓存
        if self.use_cache:
            hit = self.cache.get(self.model, prompt)
            if hit is not None:
                self.stats["cache_hits"] += 1
                s = dict_to_signal(hit, snap.symbol, snap.date, self.model,
                                   PROMPT_VERSION, ref_price=snap.close)
                s.meta["cached"] = True
                return s

        # 2) 调模型
        raw_text = self._chat(SYSTEM_PROMPT, prompt.split("\n---\n", 1)[-1])
        parsed = parse_signal(raw_text, snap.symbol, snap.date)

        # 3) 解析失败 → 降级 HOLD（绝不让脏数据进回测）
        if parsed is None:
            self.stats["errors"] += 1
            if self.use_cache:
                self.cache.put(self.model, prompt, {
                    "action": "HOLD", "score": 50, "confidence": 0,
                    "reason": "解析失败降级", "_raw": raw_text[:500]})
                self.cache.flush()
            return Signal.hold(snap.date, snap.symbol, source="llm", model=self.model,
                               prompt_version=PROMPT_VERSION,
                               reason="LLM输出无法解析，降级观望")

        if self.use_cache:
            self.cache.put(self.model, prompt, parsed)
            self.cache.flush()
        return dict_to_signal(parsed, snap.symbol, snap.date, self.model,
                              PROMPT_VERSION, ref_price=snap.close)

    def raw_chat(self, system: str, user: str, use_cache: bool = True) -> str:
        """
        裸调用：自定义 system，取回模型原文，不做 JSON 校验、不构造 Signal。

        供多空辩论这类「同一份事实、多种角色提示词」的场景使用。
        缓存键含 system 全文 —— 辩论的每一轮、每一种发言顺序各算一次独立调用，
        这正是「重跑回测零成本且结果可复现」在辩论路径上依然成立的原因。
        """
        prompt = f"{system}\n---\n{user}"
        if self.use_cache and use_cache:
            hit = self.cache.get(self.model, prompt)
            # 必须检查 _text：analyze() 往同一张缓存里写的是信号字典，
            # 万一键撞上（同 system 同 user），也不能把信号当发言文本返回。
            if hit is not None and "_text" in hit:
                self.stats["cache_hits"] += 1
                return hit["_text"]
        text = self._chat(system, user)
        if self.use_cache and use_cache:
            self.cache.put(self.model, prompt, {"_text": text})
            self.cache.flush()
        return text

    def _chat(self, system: str, user: str) -> str:
        """
        底层调用。system 与 user 分开传 ——
        辩论的三个角色各有自己的 system，不能再共用一个模块级常量。
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"

        last_err = None
        for attempt in range(self.retries + 1):
            self.stats["calls"] += 1
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode("utf-8"),
                    headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                usage = data.get("usage") or {}
                self.stats["tokens_in"] += usage.get("prompt_tokens", 0) or 0
                self.stats["tokens_out"] += usage.get("completion_tokens", 0) or 0
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                last_err = f"HTTP {exc.code}: {detail}"
                # 401/402 是账号问题，重试一百次也不会变。给出能直接照做的下一步。
                if exc.code == 401:
                    last_err = "API Key 无效或已吊销（401）"
                    break
                if exc.code == 402:
                    last_err = ("账户余额不足（402）。云端模型需先充值；"
                                "或改用本地 Ollama（见 .env.example，零成本）")
                    break
                # 唯一值得自动降级的 4xx：部分本地模型不认 response_format。
                # 提示词里本来就写了「只输出 JSON」，去掉该字段照样能解析。
                if "response_format" in detail and "response_format" in payload:
                    log.warning("%s 不支持 response_format，改用提示词约束", self.model)
                    payload.pop("response_format", None)
                    continue
                # 其余 4xx 都是配置或参数问题，重试无意义
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))

        self.stats["errors"] += 1
        raise RuntimeError(f"LLM 调用失败（{self.model}@{self.base_url}）: {last_err}")

    def cost_report(self) -> dict:
        if _is_local_url(self.base_url):
            return {**self.stats, "est_cost_cny": 0.0, "note": "本地模型，无 API 费用"}
        # DeepSeek 参考价：输入 ¥1/百万token，输出 ¥2/百万token（按需改）
        cost = self.stats["tokens_in"] / 1e6 * 1.0 + self.stats["tokens_out"] / 1e6 * 2.0
        return {**self.stats, "est_cost_cny": round(cost, 4)}


class MockAnalyzer:
    """
    离线分析器：用规则模拟一个「LLM 的判断」，用于无 API Key 时跑通全流程。

    它的存在不只是为了测试 —— 更重要的是提供一条对照线：
    如果真 LLM 跑不过这个粗糙的规则版，说明模型没带来增量。
    """

    def __init__(self, model: str = "mock-rule-v1"):
        self.model = model
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0}

    @property
    def available(self) -> bool:
        return True

    def analyze(self, snap, position=None, holding: bool = False,
                memory_block: str = "") -> Signal:
        # memory_block 刻意不参与规则计算：MockAnalyzer 是「不用 LLM 能拿到什么」
        # 的对照线。让它也读记忆，这条对照线就被污染了 —— 那样再拿 LLM 跟它比，
        # 比的就不是"模型增量"，而是"记忆增量"。
        self.stats["calls"] += 1
        f = snap.to_prompt_dict()
        score = 50.0
        reasons = []

        ma20, ma60 = f.get("MA20"), f.get("MA60")
        close = f["收盘价"]
        if ma20 and close > ma20:
            score += 12
            reasons.append("站上MA20")
        else:
            score -= 12
        if ma60 and close > ma60:
            score += 8
            reasons.append("站上MA60")
        else:
            score -= 8
        slope = f.get("MA20近5日斜率%")
        if slope is not None:
            score += max(-10, min(10, slope * 2))
        m20 = f.get("20日涨幅%")
        if m20 is not None:
            score += max(-12, min(12, m20 * 0.6))
        vr = f.get("量比")
        if vr is not None and vr > 1.5:
            score += 6
        r = f.get("RSI14")
        if r is not None:
            if r > 75:
                score -= 12
                reasons.append("RSI超买")
            elif r < 30:
                score += 10
                reasons.append("RSI超卖")
        pos = f.get("60日区间位置")
        if pos is not None and pos > 92:
            score -= 8

        score = max(0.0, min(100.0, score))
        atr = f.get("ATR14") or close * 0.02
        if score >= 62:
            action = Action.BUY
        elif score <= 38:
            action = Action.SELL
        else:
            action = Action.HOLD
        # 只有买入方向才给止损/目标；卖出是离场信号（A股不能做空）
        stop = round(close - 2.5 * atr, 2) if action == Action.BUY else None
        target = round(close + 4 * atr, 2) if action == Action.BUY else None
        return Signal(
            date=snap.date, symbol=snap.symbol, action=action, score=round(score, 1),
            confidence=round(min(0.85, abs(score - 50) / 50 + 0.3), 2),
            entry=close, stop=stop, target=target,
            horizon_days=15, source="llm", model=self.model,
            prompt_version="mock",
            reason="规则模拟：" + (", ".join(reasons) if reasons else "中性"),
            meta={"snapshot": f, "entry_is_reference": True})

    def raw_chat(self, system: str, user: str, use_cache: bool = True) -> str:
        """
        离线模拟的辩论发言。

        必须是**确定性**的：一个每次给出不同答案的模拟器，会让
        「同一输入 → 同一结论」这条测试变成空转。所以这里只根据
        系统提示词里的角色 + 用户文本里的指标做纯函数推导，不引入随机数。
        """
        self.stats["calls"] += 1
        return _mock_debate_reply(system, user)


# ------------------------------------------------------------

def build_analyzer(kind: str = "auto", **kw):
    """工厂：auto 优先真 LLM，没有 Key 就退回离线规则模拟器。"""
    if kind == "mock":
        return MockAnalyzer()
    if kind == "llm":
        a = LLMAnalyzer(**kw)
        if not a.available:
            raise RuntimeError("未配置 LLM_API_KEY")
        return a
    a = LLMAnalyzer(**kw)
    if a.available:
        log.info("使用 LLM 分析器: %s @ %s", a.model, a.base_url)
        return a
    log.warning("未检测到 LLM_API_KEY，退回离线规则模拟器（MockAnalyzer）")
    return MockAnalyzer()


# ------------------------------------------------------------
# 离线辩论回复
# ------------------------------------------------------------

_ROLE_MARKERS = (("多头", "bull"), ("空头", "bear"), ("裁决", "judge"))


def mock_role_of(system: str) -> str:
    """从系统提示词判断这是哪个角色的调用。离线与测试都靠它分流。"""
    for marker, role in _ROLE_MARKERS:
        if marker in system:
            return role
    return "unknown"


def _facts_from_text(text: str) -> dict:
    """
    从提示词里抠回几个关键指标。

    取**第一处**匹配 —— 事实块永远排在发言记录之前，所以先匹配到的就是事实，
    不会被某位辩手在论点里复述的数字覆盖。
    """
    out: dict = {}
    for key, pat in (
        ("收盘价", r"收盘价:\s*(-?[\d.]+)"),
        ("MA20", r"MA20:\s*(-?[\d.]+)"),
        ("MA60", r"MA60:\s*(-?[\d.]+)"),
        ("RSI14", r"RSI14:\s*(-?[\d.]+)"),
        ("60日区间位置", r"60日区间位置:\s*(-?[\d.]+)"),
    ):
        m = re.search(pat, text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def _mock_debate_reply(system: str, user: str) -> str:
    """
    纯函数式的离线辩论回复：同样的输入永远给出同样的输出。

    多空两侧刻意朝各自方向偏，裁决者则**只看指标、完全不看发言顺序** ——
    这正好让它成为顺序置换检验的对照线：机制正常时它必须被判为「顺序无关」。
    """
    role = mock_role_of(system)
    f = _facts_from_text(user)
    close, ma20, ma60 = f.get("收盘价"), f.get("MA20"), f.get("MA60")
    rsi, pos = f.get("RSI14"), f.get("60日区间位置")

    base = 50.0
    base += 10 if (ma20 and close and close > ma20) else -10
    base += 8 if (ma60 and close and close > ma60) else -8
    if rsi is not None:
        base -= max(0.0, (rsi - 60) * 0.8)     # 超买压分
        base += max(0.0, (40 - rsi) * 0.8)     # 超卖加分
    if pos is not None and pos > 90:
        base -= 6                              # 贴近 60 日高点，追高扣分

    if role == "bull":
        s = min(100.0, base + 14)
        return json.dumps({"action": "BUY" if s >= 62 else "HOLD",
                           "score": round(s, 1), "concede": False,
                           # 只陈述读到的数字，不下"占优/不足"这类结论 ——
                           # 模拟器一旦输出与事实相反的断言，读演示的人就被误导了
                           "point": f"多头视角：收盘{close}、MA20 {ma20}、RSI {rsi}，评分{s:.0f}"},
                          ensure_ascii=False)
    if role == "bear":
        s = max(0.0, base - 14)
        return json.dumps({"action": "SELL" if s <= 38 else "HOLD",
                           "score": round(s, 1), "concede": False,
                           "point": f"空头视角：RSI {rsi}、60日位置 {pos}、MA60 {ma60}，评分{s:.0f}"},
                          ensure_ascii=False)
    if role == "judge":
        act = "BUY" if base >= 62 else ("SELL" if base <= 38 else "HOLD")
        wb = max(0.0, min(1.0, base / 100))
        return json.dumps({"action": act, "score": round(base, 1), "confidence": 0.5,
                           "reason": f"按指标综合分{base:.0f}裁决，不依赖发言顺序",
                           "weight_bull": round(wb, 2), "weight_bear": round(1 - wb, 2)},
                          ensure_ascii=False)
    return json.dumps({"action": "HOLD", "score": 50.0, "point": "未识别角色"},
                      ensure_ascii=False)


def _is_local_url(url: str) -> bool:
    """本地推理端点：不需要 API Key，也不产生费用。"""
    u = (url or "").lower()
    return any(h in u for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


def _num(v, default=None) -> Optional[float]:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
