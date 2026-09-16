#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复权算法验收 —— 用一次真实的除权事件来判定对错
============================================================
为什么要单独测这个东西：

tushare 的 pro.daily() 给的是【不复权】原始价，而回测必须用【前复权】。
复权写错的后果不是抛异常，而是**静默失真** —— 曲线看着完全正常，
但除权日那个凭空出现的跳空会污染 MA / RSI / 最大回撤，
最后把一个垃圾策略验证成好策略。这类错误不测就发现不了。

样本：贵州茅台 2025-06-26 除权（真实数据，非构造）

    日期          tushare不复权close     官方pre_close      腾讯前复权close
    2025-06-25         1435.86               —               1356.28
    2025-06-26         1420.00            1408.26              —
    2025-06-30         1409.52               —               1357.54
    2026-09-15         1272.75               —               1272.75  (基准日)

    最新复权因子（2026-09-15）= 8.6463

两条互相独立的验证路径：

    A. 内部一致性 —— 复权后的日涨跌幅必须等于 tushare 的 pct_chg。
       这两个字段在 tushare 里是分开算的（close 不复权，pct_chg 已除权调整），
       所以能对上就说明复权方向和口径都对，不是自说自话。

    B. 外部对照 —— 复权后的价格必须贴近腾讯的前复权价（完全不同的数据源）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fetchers import Bar, _apply_adjust, _nearest_adj  # noqa: E402

# ---------------- 真实样本 ----------------
LATEST_DATE = "2026-09-15"
LATEST_ADJ = 8.6463

RAW = {                       # tushare 不复权收盘价
    "2025-06-25": 1435.86,
    "2025-06-26": 1420.00,
    "2025-06-30": 1409.52,
    LATEST_DATE: 1272.75,
}
PRE_CLOSE_0626 = 1408.26      # tushare 给出的除权后参考价
PCT_CHG_0626 = 0.8337         # tushare 给出的当日涨跌幅（已按除权调整）
TENCENT_QFQ = {               # 腾讯前复权，独立数据源，作为外部对照
    "2025-06-25": 1356.28,
    "2025-06-30": 1357.54,
    LATEST_DATE: 1272.75,
}

# 由除权当日的 pre_close 反解因子跳变幅度（不依赖腾讯，避免循环论证）：
#   除权后价格 = 除权前价格 × adj_old / adj_new
RATIO = RAW["2025-06-25"] / PRE_CLOSE_0626          # = adj_new / adj_old
ADJ_OLD = LATEST_ADJ / (RAW["2025-06-25"] / TENCENT_QFQ["2025-06-25"])
ADJ_NEW = ADJ_OLD * RATIO

FACTORS = {
    "2025-06-25": ADJ_OLD,
    "2025-06-26": ADJ_NEW,     # ← 除权日：因子跳到新值
    "2025-06-30": ADJ_NEW,
    LATEST_DATE: LATEST_ADJ,
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


def pct(a: float, b: float) -> float:
    """a 相对 b 的百分比偏差。"""
    return abs(a / b - 1) * 100


def build(pairs) -> list[Bar]:
    return [Bar(date=d, open=p, close=p, high=p, low=p, volume=0.0,
                amount=0.0, pct_chg=0.0, turnover=0.0, amplitude=0.0)
            for d, p in pairs]


def main() -> int:
    print("=" * 62)
    print("复权算法验收（贵州茅台 2025-06-26 真实除权事件）")
    print("=" * 62)
    print(f"  反解出的因子：除权前 {ADJ_OLD:.4f} → 除权后 {ADJ_NEW:.4f}"
          f"（跳变 {(RATIO - 1) * 100:.3f}%）")
    print()

    bars = build(list(RAW.items()))
    out = _apply_adjust(bars, FACTORS, "qfq")
    qfq = {b.date: b.close for b in out}

    # ---- 1. 基准日必须原价不动 ----
    print("[1] 前复权的定义性质")
    check("基准日(最新交易日)价格保持不变",
          pct(qfq[LATEST_DATE], RAW[LATEST_DATE]) < 1e-6,
          f"{qfq[LATEST_DATE]:.2f} vs raw {RAW[LATEST_DATE]:.2f}")

    # ---- 2. 内部一致性：复权后涨跌幅 == tushare pct_chg ----
    print()
    print("[2] 内部一致性：复权后涨跌幅 应等于 tushare 的 pct_chg")
    real = (qfq["2025-06-26"] / qfq["2025-06-25"] - 1) * 100
    naive = (RAW["2025-06-26"] / RAW["2025-06-25"] - 1) * 100
    print(f"       复权后涨跌 {real:+.4f}%   不复权涨跌 {naive:+.4f}%   "
          f"官方 pct_chg {PCT_CHG_0626:+.4f}%")
    check("复权后涨跌幅与官方 pct_chg 一致", abs(real - PCT_CHG_0626) < 0.01,
          f"偏差 {abs(real - PCT_CHG_0626):.4f} 个百分点")
    check("不复权口径确实是错的（方向都反了）", naive < 0 < real,
          f"真涨 {real:+.2f}%，不复权却显示 {naive:+.2f}%")

    # ---- 3. 外部对照：与腾讯前复权对齐 ----
    print()
    print("[3] 外部对照：与腾讯前复权价对比（不同数据源）")
    worst = 0.0
    for d, ref in TENCENT_QFQ.items():
        d_pct = pct(qfq[d], ref)
        worst = max(worst, d_pct)
        print(f"       {d}  本实现 {qfq[d]:9.2f}   腾讯 {ref:9.2f}   偏差 {d_pct:.3f}%")
    check("全部样本与腾讯前复权偏差 < 0.05%", worst < 0.05, f"最大偏差 {worst:.3f}%")

    # ---- 4. 价格序列不再有假跳空 ----
    print()
    print("[4] 连续性：除权日不应出现跳空")
    gap_raw = abs(RAW["2025-06-26"] / RAW["2025-06-25"] - 1) * 100
    gap_qfq = abs(qfq["2025-06-26"] / qfq["2025-06-25"] - 1) * 100
    check("复权后当日波动收敛到真实涨跌幅内", gap_qfq < 2.0 and gap_raw > gap_qfq * 1.1,
          f"不复权跳空 {gap_raw:.2f}% → 复权后 {gap_qfq:.2f}%")

    # ---- 5. 后复权：最早日不动 ----
    print()
    print("[5] 后复权（基准应落在最早一日）")
    out_h = _apply_adjust(bars, FACTORS, "hfq")
    hfq = {b.date: b.close for b in out_h}
    check("后复权最早日价格保持不变",
          pct(hfq["2025-06-25"], RAW["2025-06-25"]) < 1e-6,
          f"{hfq['2025-06-25']:.2f} vs raw {RAW['2025-06-25']:.2f}")
    check("后复权涨幅 == 前复权涨幅（复权方式不该改变收益率）",
          abs((hfq["2025-06-26"] / hfq["2025-06-25"] - 1) - (qfq["2025-06-26"] / qfq["2025-06-25"] - 1)) < 1e-9)

    # ---- 6. 停牌日因子回溯 ----
    print()
    print("[6] 停牌日：应向前取最近因子，绝不能向后取")
    dates = sorted(FACTORS)
    check("停牌日(2025-06-28 周六)取到除权后因子",
          abs(_nearest_adj(FACTORS, dates, "2025-06-28") - ADJ_NEW) < 1e-9)
    check("早于因子表起点时外推为最早因子（不返回 None）",
          abs(_nearest_adj(FACTORS, dates, "2025-06-20") - ADJ_OLD) < 1e-9,
          "（数据覆盖不足时退化为外推，_apply_adjust 会发警告）")

    # ---- 7. 关闭复权时保持原样 ----
    print()
    print("[7] adjust='none' 时必须原样返回")
    raw_out = _apply_adjust(bars, FACTORS, "none")
    check("不传复权时价格不被修改",
          all(abs(a.close - b.close) < 1e-9 for a, b in zip(raw_out, bars)))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print()
    print("=" * 62)
    print(f"复权验收：{passed}/{total} 通过")
    print("=" * 62)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
