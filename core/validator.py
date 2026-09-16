#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略验证层 —— 自动体检与过拟合检测
============================================================
代码化 a-share-master《验证层-回测框架》里的判定标准：

    指标        及格线      优秀线
    胜率        >45%       >55%
    盈亏比      >1.5:1     >2.5:1
    最大回撤    <20%       <12%
    夏普        >1.0       >1.5
    交易频率    5-20笔/月  8-15笔/月

三层验证在这里的落地方式：
    第一层 逻辑一致性  → 信号分布检查（模型是不是永远在观望？）
    第二层 规则回测    → 回测引擎跑出的五大指标
    第三层 完整回测    → 样本内/外切分 + 显著性检验（本文件的重点）

第三层才是真正区分「有 alpha」和「运气好」的地方：
    · 样本外指标崩了  → 过拟合，参数是在拟合噪声
    · p 值不显著      → 高胜率可能只是样本量太小
    · 换手率过低      → 结论没有统计意义（3 笔交易说明不了任何事）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .backtest import BacktestConfig, BacktestEngine, BacktestResult

log = logging.getLogger("validator")

# 与《验证层-回测框架》完全对齐的阈值
# direction=up   : 越大越好，pass/good 是下限
# direction=down : 越小越好，pass/good 是上限
# direction=range: 落在区间内才好，good_lo/good_hi 是优秀区间
THRESHOLDS = {
    "win_rate_pct":     {"pass": 45.0, "good": 55.0, "direction": "up"},
    "profit_loss_ratio": {"pass": 1.5, "good": 2.5, "direction": "up"},
    "max_drawdown_pct": {"pass": 20.0, "good": 12.0, "direction": "down"},
    "sharpe":           {"pass": 1.0,  "good": 1.5,  "direction": "up"},
    "trades_per_month": {"pass_lo": 5.0, "pass_hi": 20.0,
                         "good_lo": 8.0, "good_hi": 15.0, "direction": "range"},
}

# 指标不达标 → 问题定位到哪一层（对应验证层文档的层级诊断表）
DIAGNOSIS = {
    "win_rate_pct":     ("选股层 / 执行层", "入场时机质量不足：放宽入场条件或提高评分门槛"),
    "profit_loss_ratio": ("执行层-退出框架", "盈利时跑太早或亏损时扛太久：检查止盈/止损参数"),
    "max_drawdown_pct": ("风控层", "仓位过重或缺少止损：降 per_trade_pct 或收紧止损"),
    "sharpe":           ("战略层-周期", "波动过大收益不稳：加趋势过滤，或降低交易频率"),
    "trades_per_month": ("战略层 / 频率", "频率偏离区间：过高说明过度交易，过低说明样本不足"),
}


@dataclass
class CheckResult:
    name: str
    value: float
    verdict: str          # PASS / GOOD / FAIL / NA
    threshold: str
    note: str = ""


@dataclass
class ValidationReport:
    strategy_name: str = ""
    period: tuple = ("", "")
    checks: list[CheckResult] = field(default_factory=list)
    grade: str = "NA"                     # 🟢 全达标 / 🟡 部分 / 🔴 需检修
    overfit_verdict: str = ""
    is_metrics: Optional[dict] = None
    oos_metrics: Optional[dict] = None
    significance_note: str = ""
    signal_distribution: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    benchmark_note: str = ""


# ============================================================
# 主验证器
# ============================================================

class StrategyValidator:
    """
    一个策略的完整体检流程：
        回测 → 五大指标打分 → 样本内外对比 → 显著性检验 → 逐层诊断
    """

    def __init__(self, config: Optional[BacktestConfig] = None,
                 is_ratio: float = 0.7, min_trades_for_verdict: int = 20):
        self.cfg = config or BacktestConfig()
        self.is_ratio = is_ratio
        self.min_trades = min_trades_for_verdict

    # ---------- 主入口 ----------

    def validate(self, strategy, bars_by_symbol: dict, benchmark: Optional[list] = None,
                 name: Optional[str] = None) -> tuple[BacktestResult, ValidationReport]:
        engine = BacktestEngine(strategy, self.cfg)
        result = engine.run(bars_by_symbol, benchmark)
        report = self._build_report(result, bars_by_symbol, benchmark, name)
        return result, report

    # ---------- 报告构建 ----------

    def _build_report(self, r: BacktestResult, bars_by_symbol: dict,
                      benchmark: Optional[list], name: Optional[str]) -> ValidationReport:
        m = r.metrics
        rep = ValidationReport(
            strategy_name=name or r.strategy_name,
            period=r.period,
            warnings=list(r.warnings),
        )

        # ---- 五大指标 ----
        for key, th in THRESHOLDS.items():
            val = float(getattr(m, key, 0.0) or 0.0)
            if th["direction"] == "range":
                lo_ok = th["pass_lo"] <= val <= th["pass_hi"]
                good_ok = th["good_lo"] <= val <= th["good_hi"]
                verdict = "GOOD" if good_ok else "PASS" if lo_ok else "FAIL"
                thr_text = f"{th['pass_lo']:.0f}-{th['pass_hi']:.0f} 笔/月（优秀 {th['good_lo']:.0f}-{th['good_hi']:.0f}）"
            elif th["direction"] == "up":
                verdict = "GOOD" if val >= th["good"] else "PASS" if val >= th["pass"] else "FAIL"
                thr_text = f"≥{th['pass']}（优秀 ≥{th['good']}）"
            else:
                verdict = "GOOD" if val <= th["good"] else "PASS" if val <= th["pass"] else "FAIL"
                thr_text = f"≤{th['pass']}（优秀 ≤{th['good']}）"
            if m.trades == 0:
                verdict = "NA"
            rep.checks.append(CheckResult(
                name=key, value=round(val, 2), verdict=verdict, threshold=thr_text))

        # ---- 这些结论无论如何都要给，不能因为样本少就留空 ----
        # 基准对比
        if m.benchmark_return_pct:
            verb = "跑赢" if m.excess_return_pct > 0 else "跑输"
            rep.benchmark_note = (f"策略 {m.total_return_pct:+.2f}% vs 基准 "
                                  f"{m.benchmark_return_pct:+.2f}% → {verb} "
                                  f"{abs(m.excess_return_pct):.2f}%")
        else:
            rep.benchmark_note = "未取得基准指数数据，无法判断超额收益（强烈建议补上）"

        # 样本内外一致性
        ins, oos = r.in_sample_split(self.is_ratio)
        rep.is_metrics = ins.metrics.as_dict()
        rep.oos_metrics = oos.metrics.as_dict()
        rep.overfit_verdict = self._overfit_check(ins.metrics, oos.metrics)

        # 显著性
        rep.significance_note = self._significance_note(m)

        # 信号分布
        rep.signal_distribution = dict(r.signal_counts)

        # ---- 样本不足：结论到此为止，但上面的字段都已填好 ----
        if m.trades < self.min_trades:
            rep.grade = "🔴 样本不足"
            rep.issues.append(
                f"仅 {m.trades} 笔交易，低于判定门槛 {self.min_trades} 笔 —— "
                f"此结果不具备统计意义，任何结论都是噪声")
            rep.advice.append(
                "不要据此调参。先让策略产生足够样本：放宽买入阈值、拉长回测区间、"
                "扩大标的池，或提高卖出阈值以加快持仓周转")
            if rep.benchmark_note:
                rep.issues.append("基准对比：" + rep.benchmark_note)
            if r.warnings:
                rep.advice.append("回测引擎告警：" + "；".join(r.warnings))
            return rep

        # ---- 评级 ----
        verdicts = [c.verdict for c in rep.checks if c.verdict != "NA"]
        n_fail = verdicts.count("FAIL")
        if n_fail == 0 and verdicts.count("GOOD") >= 3:
            rep.grade = "🟢 全达标"
        elif n_fail <= 2:
            rep.grade = "🟡 部分不达标"
        else:
            rep.grade = "🔴 需检修"

        # ---- 层级诊断 ----
        for c in rep.checks:
            if c.verdict == "FAIL" and c.name in DIAGNOSIS:
                layer, why = DIAGNOSIS[c.name]
                rep.issues.append(f"{c.name}={c.value} 不达标 → 定位【{layer}】：{why}")
                rep.advice.append(f"【{layer}】{why}")

        if rep.overfit_verdict.startswith("🔴"):
            rep.advice.append("【第三层验证】样本外表现崩塌，说明参数在拟合噪声："
                              "减少可调参数、拉长回测区间、或降低策略复杂度")

        # ---- HOLD 占比检查 ----
        total_sig = sum(r.signal_counts.values()) or 1
        hold_ratio = r.signal_counts.get("HOLD", 0) / total_sig
        if hold_ratio > 0.9:
            rep.advice.append(
                f"【第一层验证】HOLD 占比 {hold_ratio:.0%}，策略绝大多数时间无所作为 —— "
                f"先确认这是设计意图（只做极少数高确定性机会），还是阈值过严")

        if m.benchmark_return_pct and m.excess_return_pct < -5 and rep.grade.startswith("🟢"):
            rep.advice.append("绝对收益达标但显著跑输基准 —— 同样的钱买指数更省心，"
                              "需要提高门槛或换赛道")

        if r.warnings:
            rep.advice.append("回测过程中发现以下问题，结论需打折扣："
                              + "；".join(r.warnings))
        return rep

    # ---------- 过拟合检测 ----------

    @staticmethod
    def _overfit_check(ism, oosm) -> str:
        if not oosm.trades:
            return "⚠️ 样本外无交易，无法判断（样本外区间太短或标的在该段无信号）"

        thin = ism.trades < 10 or oosm.trades < 10
        tail = (f"（但样本内 {ism.trades} 笔 / 样本外 {oosm.trades} 笔，"
                f"分段样本偏薄，衰减方向可信、幅度不可信）") if thin else ""

        ratio_is, ratio_oos = ism.win_rate_pct, oosm.win_rate_pct
        exp_is, exp_oos = ism.expectancy_pct, oosm.expectancy_pct

        if exp_oos < 0 and exp_is > 0:
            return (f"🔴 样本内为正、样本外转负：单笔期望 {exp_is:+.2f}% → {exp_oos:+.2f}%"
                    f"{tail}")
        if exp_is > 0 and exp_oos < exp_is * 0.4:
            return (f"🔴 期望衰减超 60%：{exp_is:+.2f}% → {exp_oos:+.2f}%"
                    f"，典型的过拟合特征{tail}")
        if abs(ratio_is - ratio_oos) > 15:
            return (f"🟡 胜率不稳：{ratio_is:.1f}% → {ratio_oos:.1f}%"
                    f"，波动超 15 个百分点{tail}")
        return (f"🟢 样本内外一致：单笔期望 {exp_is:+.2f}% → {exp_oos:+.2f}%，"
                f"胜率 {ratio_is:.1f}% → {ratio_oos:.1f}%")

    # ---------- 显著性 ----------

    @staticmethod
    def _significance_note(m) -> str:
        if m.trades < 10:
            return "交易笔数不足以做显著性检验（建议 ≥30 笔）"
        if m.alpha_p_value <= 0.01 and m.alpha_t_stat > 0:
            return (f"✅ 高度显著：t={m.alpha_t_stat}，p={m.alpha_p_value}（<0.01）—— "
                    f"日收益均值显著为正，不太可能是纯随机")
        if m.is_significant:
            return (f"✅ 显著：t={m.alpha_t_stat}，p={m.alpha_p_value}（<0.05）")
        if m.alpha_t_stat < 0:
            return (f"🔴 负向显著：t={m.alpha_t_stat}，p={m.alpha_p_value} —— "
                    f"策略的期望收益为负，反向做可能更好")
        return (f"⚠️ 不显著：t={m.alpha_t_stat}，p={m.alpha_p_value}（>0.05）—— "
                f"当前收益与随机波动无法区分，样本量或优势都不够")

    # ---------- 多策略横向对比 ----------

    def compare(self, strategies: list, bars_by_symbol: dict,
                benchmark: Optional[list] = None) -> list[tuple[str, BacktestResult, ValidationReport]]:
        out = []
        for st in strategies:
            log.info("评估策略: %s", st.name)
            r, rep = self.validate(st, bars_by_symbol, benchmark)
            out.append((st.name, r, rep))
        out.sort(key=lambda x: x[1].metrics.sharpe, reverse=True)
        return out


# ============================================================
# 渲染：把体检报告输出成 Markdown（可直接贴进季度体检表）
# ============================================================

def render_validation(rep: ValidationReport, result: BacktestResult) -> str:
    m = result.metrics
    L = []
    A = L.append
    A(f"## 策略体检报告 · {rep.strategy_name}")
    A("")
    A(f"- 回测区间：{rep.period[0]} ~ {rep.period[1]}")
    A(f"- 综合评级：**{rep.grade}**")
    A(f"- 基准对比：{rep.benchmark_note}")
    A("")
    if "样本不足" in rep.grade:
        A("> ⚠️ **交易样本不足，下表仅供参考，不构成对策略有效性的判断。**")
        A("> 统计上，交易笔数少于 20 笔时，胜率和盈亏比的置信区间宽到没有意义 ——")
        A("> 连续盈利 3 笔和连续亏损 3 笔，在抛硬币里都是常见事件。")
        A("")
    A("### 五大指标（对齐 a-share-master 验证层）")
    A("")
    A("| 指标 | 实测 | 及格线 | 判定 |")
    A("|------|------|--------|------|")
    label = {
        "win_rate_pct": "胜率", "profit_loss_ratio": "盈亏比",
        "max_drawdown_pct": "最大回撤%", "sharpe": "夏普比率",
        "trades_per_month": "交易频率",
    }
    icon = {"GOOD": "🟢 优秀", "PASS": "🟡 及格", "FAIL": "🔴 不达标", "NA": "— 无数据"}
    for c in rep.checks:
        unit = "%" if c.name in ("win_rate_pct", "max_drawdown_pct") else ""
        A(f"| {label.get(c.name, c.name)} | {c.value}{unit} | {c.threshold} | {icon[c.verdict]} |")
    A("")
    A("### 收益与风险")
    A("")
    A("| 项目 | 数值 |")
    A("|------|------|")
    A(f"| 累计收益 | {m.total_return_pct:+.2f}% |")
    A(f"| 年化收益 | {m.annual_return_pct:+.2f}% |")
    A(f"| 基准收益 | {m.benchmark_return_pct:+.2f}% |")
    A(f"| 超额收益 | {m.excess_return_pct:+.2f}% |")
    A(f"| 期末权益 | ¥{m.final_equity:,.2f} |")
    A(f"| 交易笔数 | {m.trades}（胜 {m.wins} / 负 {m.losses}） |")
    A(f"| 单笔期望 | {m.expectancy_pct:+.3f}% |")
    A(f"| 盈利因子（总盈利/总亏损） | {m.profit_factor} |")
    A(f"| 平均持仓 | {m.avg_hold_days} 个交易日 |")
    A(f"| 年化波动 | {m.volatility_pct}% |")
    A(f"| 索提诺 | {m.sortino} |")
    A(f"| 卡玛比率 | {m.calmar} |")
    A(f"| 持仓天数占比 | {m.exposure_pct}% |")
    A("")
    A("### 第三层：样本内外一致性（过拟合检测）")
    A("")
    A(rep.overfit_verdict or "— 数据不足，未能判定")
    A("")
    if rep.is_metrics and rep.oos_metrics and (rep.is_metrics.get("trades") or
                                              rep.oos_metrics.get("trades")):
        A("| 分组 | 笔数 | 胜率% | 盈亏比 | 单笔期望% |")
        A("|------|------|-------|--------|-----------|")
        for tag, mm in (("样本内(前段)", rep.is_metrics), ("样本外(后段)", rep.oos_metrics)):
            A(f"| {tag} | {mm['trades']} | {mm['win_rate_pct']} | "
              f"{mm['profit_loss_ratio']} | {mm['expectancy_pct']} |")
        A("")
    A("### 统计显著性")
    A("")
    A(rep.significance_note or "— 未能计算")
    A("")
    A("### 信号分布")
    A("")
    dist = rep.signal_distribution or {}
    tot = sum(dist.values()) or 1
    A("| 信号 | 次数 | 占比 |")
    A("|------|------|------|")
    for k in ("BUY", "SELL", "HOLD", "AVOID"):
        if k in dist:
            A(f"| {k} | {dist[k]} | {dist[k]/tot:.1%} |")
    A("")
    if rep.issues:
        A("### 诊断")
        A("")
        for i in rep.issues:
            A(f"- {i}")
        A("")
    if rep.advice:
        A("### 改进方向")
        A("")
        for a in rep.advice:
            A(f"- {a}")
        A("")
    if rep.warnings:
        A("### 回测引擎告警")
        A("")
        for w in rep.warnings:
            A(f"- ⚠️ {w}")
        A("")
    return "\n".join(L)
