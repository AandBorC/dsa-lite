#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时点门控验收 —— 用「真的会泄漏的场景」来判定，而不是读一遍代码觉得对
============================================================

回测里最容易骗过自己的事：策略代码明明只读了 bars[:i+1]，结论却还是错的 ——
因为**进来的数据本身**就带着 as_of 之后的信息。这类错误不抛异常、不打日志，
只是安静地把一个垃圾策略验证成好策略。

所以这个测试不给被测对象配合作弊的机会，四条主线全部构造真实泄漏场景：

    [1] 请求不发       下游收到的 end 参数必须已经被钉在 as_of（相当于
                      TradingAgents 的 mock.assert_not_called()：未来数据
                      连请求都不该发出去）
    [2] 出口过滤       故意让上游吐回 as_of 之后的 K 线（模拟不听话的数据源），
                      出口必须裁掉 —— 这测的是「不信任上游」而不是「上游自觉」
    [3] 降级路径       过期缓存回退这条路径最容易漏，单独测
    [4] 复权基准日     A 股特有的一维：as_of 落在除权日之前时，
                      那次除权不能参与历史价复权

另外三条辅线：

    [5] 带时间戳内容   晚于 as_of 的丢弃；无时间戳的在历史运行里丢弃
    [6] LLM prompt     时点纪律必须写进提示词，且**不能泄露真实今天**
    [7] 门控未开也发声 不指定 as_of 时必须在 warnings 里说明，不能静默

每条都带一个「对照组」，证明测试场景本身真的含泄漏 —— 否则一个空转的
测试会给人虚假的安全感，那比没有测试更糟。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.fetchers as F                                          # noqa: E402
from core import asof                                              # noqa: E402
from core.backtest import BacktestConfig, BacktestEngine           # noqa: E402
from core.fetchers import Bar, CsvCache, FetcherChain              # noqa: E402
from core.fetchers import _apply_adjust                            # noqa: E402
from core.indicators import Snapshot                               # noqa: E402
from core.llm import build_prompt                                  # noqa: E402
from core.strategies import FiveDimStrategy                        # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


# ============================================================
# 测试替身：一个「不听话」的数据源
# ============================================================

class SpyFetcher:
    """
    记录收到的请求参数，并**故意无视 end 参数**吐回整段数据。

    这不是刁难 —— 真实数据源就是这样：你请求 2025-06-20 为终点，
    它顺手把最新一段也带上。门控必须扛得住这种上游。
    """
    name = "spy"
    calls: list[dict] = []
    payload: list[Bar] = []

    def available(self) -> bool:
        return True

    def fetch(self, symbol, start, end, adjust="qfq", is_index=False):
        type(self).calls.append({"symbol": symbol, "start": start, "end": end,
                                 "adjust": adjust})
        return list(type(self).payload)


def gen_bars(start: str = "2025-01-01", n: int = 100) -> list[Bar]:
    d0 = date.fromisoformat(start)
    out = []
    for k in range(n):
        px = 100.0 + k * 0.5
        out.append(Bar(date=(d0 + timedelta(days=k)).isoformat(),
                       open=px, close=px, high=px + 1, low=px - 1,
                       volume=1_000_000.0, amount=100_000_000.0,
                       pct_chg=0.0, turnover=1.0, amplitude=2.0))
    return out


ALL_BARS = gen_bars()                 # 2025-01-01 ~ 2025-04-10
AS_OF = "2025-02-28"                  # 探针时点
TAIL = ALL_BARS[-1].date              # 2025-04-10


def main() -> int:
    print("=" * 66)
    print("时点门控验收（as_of=2025-02-28，数据实际覆盖到 2025-04-10）")
    print("=" * 66)

    F._REGISTRY["spy"] = SpyFetcher
    SpyFetcher.calls, SpyFetcher.payload = [], list(ALL_BARS)

    # ========================================================
    # [1] 请求不发到未来
    # ========================================================
    print()
    print("[1] 请求区间必须钉在 as_of 内 —— 未来数据连请求都不发")
    chain = FetcherChain(priority=["spy"], use_cache=False)
    got, src = chain.fetch("sh600519", "2025-01-01", "2025-04-10", as_of=AS_OF)
    last_call = SpyFetcher.calls[-1]
    print(f"       下游实收 end = {last_call['end']}（as_of = {AS_OF}）")
    check("下游收到的 end 被压到 as_of", last_call["end"] == AS_OF,
          f"收到 {last_call['end']}")
    check("返回的 K 线末根不晚于 as_of", got and got[-1].date <= AS_OF,
          f"末根 {got[-1].date if got else '—'}")

    # 对照组：证明这个场景本身真的会泄漏
    SpyFetcher.calls = []
    ungated, _ = chain.fetch("sh600519", "2025-01-01", "2025-04-10")
    check("对照组：不门控时确实会拿到未来数据（测试非空转）",
          ungated and ungated[-1].date == TAIL and ungated[-1].date > AS_OF,
          f"未门控末根 {ungated[-1].date if ungated else '—'}，"
          f"多出 {len(ungated) - len(got)} 根")

    # ========================================================
    # [2] 出口过滤 —— 不信任上游
    # ========================================================
    print()
    print("[2] 出口过滤：即使上游无视 end 参数吐回未来数据，也必须裁掉")
    SpyFetcher.calls = []
    got2, _ = chain.fetch("sh600519", "2025-01-01", "2025-04-10", as_of=AS_OF)
    check("上游确实吐回了越界数据（证明场景有效）",
          len(SpyFetcher.payload) > len(got2),
          f"上游 {len(SpyFetcher.payload)} 根 → 出口 {len(got2)} 根")
    check("出口按 as_of 裁掉越界部分", all(b.date <= AS_OF for b in got2),
          f"裁掉 {len(SpyFetcher.payload) - len(got2)} 根")
    check("裁剪被记账，不是静默丢弃",
          chain.audit is not None and chain.audit.dirty
          and sum(chain.audit.clipped.values()) == len(SpyFetcher.payload) - len(got2),
          f"audit={chain.audit.clipped if chain.audit else None}")

    # ========================================================
    # [3] 降级路径：过期缓存回退
    # ========================================================
    print()
    print("[3] 过期缓存回退 —— 最容易漏的一条降级路径")
    tmp = Path(tempfile.mkdtemp(prefix="dsa_lite_asof_"))
    cache = CsvCache(cache_dir=tmp, ttl_days=1.0)
    cache.write("sh600519", "qfq", ALL_BARS)
    p = cache.path("sh600519", "qfq")
    old = time.time() - 3 * 86400
    os.utime(p, (old, old))                        # 假装缓存是三天前的 → 已过期
    check("前置条件：缓存已过期", not cache.is_fresh("sh600519", "qfq"))

    SpyFetcher.payload = []                        # 数据源全部失败
    chain3 = FetcherChain(priority=["spy"], cache=cache)
    got3, src3 = chain3.fetch("sh600519", "2025-01-01", "2025-04-10", as_of=AS_OF)
    check("走了过期缓存回退路径", src3 == "cache(stale)", f"source={src3}")
    check("过期缓存也被裁到 as_of", got3 and got3[-1].date <= AS_OF,
          f"末根 {got3[-1].date if got3 else '—'}，共 {len(got3)} 根")
    check("过期缓存也被切到请求区间（原实现会原样返回整段）",
          len(got3) < len(ALL_BARS),
          f"{len(ALL_BARS)} 根 → {len(got3)} 根")

    SpyFetcher.payload = list(ALL_BARS)            # 复原

    # ========================================================
    # [4] 复权基准日的时点正确性
    # ========================================================
    print()
    print("[4] 前复权基准日：as_of 落在除权日之前时，那次除权不得参与复权")
    # 构造：2025-03-10 除权，因子从 1.0 跳到 1.2（除权前价格 × 1/1.2）
    factors = {"2025-03-01": 1.0, "2025-03-10": 1.2, "2025-06-30": 1.2}
    bars = [Bar(date="2025-03-05", open=120.0, close=120.0, high=120.0, low=120.0,
                volume=0.0, amount=0.0, pct_chg=0.0, turnover=0.0, amplitude=0.0),
            Bar(date="2025-06-30", open=100.0, close=100.0, high=100.0, low=100.0,
                volume=0.0, amount=0.0, pct_chg=0.0, turnover=0.0, amplitude=0.0)]

    p_asof = _apply_adjust(bars, factors, "qfq", as_of="2025-03-05")[0].close
    p_today = _apply_adjust(bars, factors, "qfq", as_of=None)[0].close
    print(f"       2025-03-05（除权前）收盘：as_of 口径 {p_asof:.4f} / "
          f"今天口径 {p_today:.4f}")
    check("as_of 早于除权日时，历史价保持原值不变",
          abs(p_asof - 120.0) < 1e-9, f"{p_asof:.4f}")
    check("对照组：用最新因子会把未来的除权缩进历史价（默认行为）",
          abs(p_today - 100.0) < 1e-9, f"{p_today:.4f}（=120/1.2）")
    check("两种口径确实不同（证明这一维真实存在）",
          abs(p_asof - p_today) > 1e-6,
          f"偏差 {(p_today / p_asof - 1) * 100:+.2f}%")

    # as_of 晚于除权日 → 与缺省行为一致（不能因为加了参数就改变正确场景）
    p_after = _apply_adjust(bars, factors, "qfq", as_of="2025-04-01")[0].close
    check("as_of 晚于除权日时，结果与『用最新因子』一致",
          abs(p_after - p_today) < 1e-9, f"{p_after:.4f}")
    # 后复权基准固定在最早一日，天然不随时间漂移
    h1 = _apply_adjust(bars, factors, "hfq", as_of="2025-03-05")[0].close
    h2 = _apply_adjust(bars, factors, "hfq", as_of=None)[0].close
    check("后复权不受 as_of 影响（基准固定最早日，不漂移）",
          abs(h1 - h2) < 1e-9 and abs(h1 - 120.0) < 1e-9, f"{h1:.4f} == {h2:.4f}")

    # ========================================================
    # [5] 带时间戳内容 / 无时间戳内容
    # ========================================================
    print()
    print("[5] 带时间戳内容：晚于 as_of 的丢弃；无时间戳的在历史运行里丢弃")
    rows = [{"date": "2025-02-27", "title": "冲突缓和"},      # 早于 as_of → 留
            {"date": "2025-02-28", "title": "业绩快报"},      # 正好 as_of → 留
            {"date": "2025-03-01", "title": "监管问询"},      # 晚于 → 丢
            {"date": "2025-03-02", "title": "股东减持"},      # 晚于 → 丢
            {"date": None, "title": "来源未标注时间"}]        # 无时间戳 → 历史丢

    kept, dropped = asof.filter_dated(rows, AS_OF)
    check("早于/等于 as_of 的保留", len([k for k in kept if k["date"]]) == 2,
          f"保留 {len([k for k in kept if k['date']])} 条带时间戳的")
    check("晚于 as_of 的丢弃", len(dropped) == 3, f"丢弃 {len(dropped)} 条")
    check("无时间戳内容在历史运行里被丢弃",
          not any(k["date"] is None for k in kept))

    kept_live, _ = asof.filter_dated(rows, asof.today())
    check("实时运行保留无时间戳内容（不能一刀切）",
          any(k["date"] is None for k in kept_live))

    check("窗口边界：end 当天算在内、次日排除",
          asof.in_window("2025-02-28", "2025-02-01", AS_OF)
          and not asof.in_window("2025-03-01", "2025-02-01", AS_OF))

    check("扣留时给出说明文本，而不是留白",
          all(s in asof.withheld_notice("新闻", AS_OF)
              for s in ("新闻", AS_OF, "扣留", "不要")))

    # ========================================================
    # [6] LLM prompt 的时点纪律
    # ========================================================
    print()
    print("[6] LLM prompt：写明时点纪律，且不泄露『真实今天是哪天』")
    snap = Snapshot(symbol="sh600519", date="2025-02-28", close=1400.0, pct_chg=1.2)
    prompt = build_prompt("sh600519", snap, holding=False)
    check("prompt 带上快照日期", "2025-02-28" in prompt)
    check("prompt 写明『你不知道该日期之后的事』",
          "不知道" in prompt and "时点纪律" in prompt)
    check("prompt 不泄露真实今天（回测里那就是未来信息）",
          asof.today() not in prompt,
          f"今天={asof.today()} 未出现在 prompt 中")

    # ========================================================
    # [7] 门控未开时也必须发声
    # ========================================================
    print()
    print("[7] 不指定 as_of 时不许静默 —— 沉默的『没有门控』最危险")
    engine = BacktestEngine(FiveDimStrategy(), BacktestConfig())
    r_none = engine.run({"sh600519": ALL_BARS})
    check("未指定 as_of 时 warnings 明确提示",
          any("未指定 as_of" in w for w in r_none.warnings),
          f"warnings={r_none.warnings[:1]}")

    r_gated = engine.run({"sh600519": ALL_BARS}, as_of=AS_OF)
    check("指定 as_of 后回测区间被真正收窄",
          r_gated.period[1] <= AS_OF and r_gated.as_of == AS_OF,
          f"区间 {r_gated.period[0]} ~ {r_gated.period[1]}，as_of={r_gated.as_of}")
    check("越界裁剪写进回测 warnings（有据可查）",
          any("时点门控生效" in w for w in r_gated.warnings),
          f"裁掉 {len(ALL_BARS) - sum(1 for b in ALL_BARS if b.date <= AS_OF)} 根")

    # 收尾：清理临时缓存目录
    shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print()
    print("=" * 66)
    print(f"时点门控验收：{passed}/{total} 通过")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
