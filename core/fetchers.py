#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多源行情获取层 —— 降级链设计
============================================================
工程结构参考 daily_stock_analysis：按优先级逐个尝试数据源，
任一成功即返回，并记录「实际命中的数据源」，便于排查数据来源。

与传统写法不同，这里把「东方财富裸 HTTP 接口」放在第一优先：
它只需标准库（urllib + json），不依赖 pandas/akshare，
因此在任何干净环境里都能跑通，也天然适配 GitHub Actions。

降级链（可通过 config.yaml 的 fetcher_priority 调整）：
    eastmoney  →  akshare  →  tushare  →  baostock  →  本地缓存

用法：
    from core.fetchers import get_bars, normalize_symbol
    bars, source = get_bars("sh600519", "20250101", "20260915")
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from . import asof

log = logging.getLogger("fetchers")

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# 指数代码 → 东财 secid 前缀
_INDEX_SECID = {
    "sh000001": "1.000001",   # 上证指数
    "sh000300": "1.000300",   # 沪深300（默认基准）
    "sh000905": "1.000905",   # 中证500
    "sz399001": "0.399001",   # 深证成指
    "sz399006": "0.399006",   # 创业板指
    "sh000688": "1.000688",   # 科创50
}

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Bar:
    """一根日线。字段名与东财顺序对齐，避免中间层反复改名。"""
    date: str          # YYYY-MM-DD
    open: float
    close: float
    high: float
    low: float
    volume: float      # 手
    amount: float      # 元
    pct_chg: float     # 涨跌幅 %
    turnover: float    # 换手率 %
    amplitude: float   # 振幅 %

    def as_dict(self) -> dict:
        return asdict(self)


class NoDataSourceError(RuntimeError):
    """所有数据源都失败时抛出，附带每个源的失败原因。"""

    def __init__(self, symbol: str, errors: dict):
        self.symbol = symbol
        self.errors = errors
        detail = "; ".join(f"{k}={v}" for k, v in errors.items())
        super().__init__(f"{symbol} 全部数据源失败 → {detail}")


# ============================================================
# 代码规范化
# ============================================================

def normalize_symbol(code: str) -> str:
    """
    把各种写法统一成 sh600519 / sz000001 / bj430047 形式。

    支持：600519 / SH600519 / 600519.SH / sh600519 / 000001.SZ
    指数：000300 / sh000300 → sh000300
    """
    s = str(code).strip().lower().replace(" ", "")
    if not s:
        raise ValueError("空代码")

    # 600519.SH → sh600519
    if "." in s:
        num, mkt = s.split(".", 1)
        return _prefix(mkt) + num.zfill(6)

    # 已带前缀
    for p in ("sh", "sz", "bj"):
        if s.startswith(p):
            return p + s[len(p):].zfill(6)

    num = s.zfill(6)
    if num in _INDEX_SECID or num.startswith(("000", "999")) and len(num) == 6 and num.startswith("000"):
        # 000xxx 可能是深市个股也可能是上证指数，交由调用方用 index=True 区分
        pass
    return _prefix_by_number(num) + num


def _prefix(mkt: str) -> str:
    mkt = mkt.lower()
    if mkt in ("sh", "ss", "xshg", "sse"):
        return "sh"
    if mkt in ("sz", "xshe", "szse"):
        return "sz"
    if mkt in ("bj", "bse"):
        return "bj"
    raise ValueError(f"未知市场后缀: {mkt}")


def _prefix_by_number(num: str) -> str:
    if num.startswith(("60", "68", "51", "58", "56", "11", "50")):
        return "sh"
    if num.startswith(("00", "30", "12", "15", "16", "18", "39")):
        return "sz"
    if num.startswith(("43", "83", "87", "92")):
        return "bj"
    return "sh"


def to_secid(symbol: str, is_index: bool = False) -> str:
    """转成东财 secid。sh→1.，sz→0.，bj→0."""
    sym = normalize_symbol(symbol)
    if is_index or sym in _INDEX_SECID:
        if sym in _INDEX_SECID:
            return _INDEX_SECID[sym]
    num = sym[2:]
    market = "1" if sym.startswith("sh") else "0"
    return f"{market}.{num}"


def to_ts_code(symbol: str) -> str:
    """转 Tushare 的 600519.SH 形式。"""
    sym = normalize_symbol(symbol)
    return f"{sym[2:]}.{sym[:2].upper()}"


# ============================================================
# 数据源 1：东方财富（标准库直连，第一优先）
# ============================================================

class EastMoneyFetcher:
    name = "eastmoney"

    BASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    # f51日期 f52开 f53收 f54高 f55低 f56量 f57额 f58振幅 f59涨跌幅 f60涨跌额 f61换手率
    FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"

    def __init__(self, timeout: int = 15, retries: int = 2, pause: float = 0.4):
        self.timeout = timeout
        self.retries = retries
        self.pause = pause

    def fetch(self, symbol: str, start: str, end: str,
              adjust: str = "qfq", is_index: bool = False) -> list[Bar]:
        # fqt: 0不复权 1前复权 2后复权
        fqt = {"": "0", "none": "0", "qfq": "1", "hfq": "2"}.get(adjust, "1")
        params = {
            "secid": to_secid(symbol, is_index),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": self.FIELDS2,
            "klt": "101",       # 日线
            "fqt": fqt,
            "beg": _compact(start),
            "end": _compact(end),
            "lmt": "100000",
        }
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                raw = self._get(url)
                return self._parse(raw)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.pause * (attempt + 1))
        raise RuntimeError(f"eastmoney 请求失败: {last_err}")

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        if not data.get("data"):
            raise RuntimeError("eastmoney 返回空 data（代码不存在或已退市）")
        return data

    @staticmethod
    def _parse(payload: dict) -> list[Bar]:
        data = payload["data"]
        out: list[Bar] = []
        for line in data.get("klines") or []:
            p = line.split(",")
            if len(p) < 11:
                continue
            try:
                out.append(Bar(
                    date=p[0],
                    open=float(p[1]),
                    close=float(p[2]),
                    high=float(p[3]),
                    low=float(p[4]),
                    volume=float(p[5]),
                    amount=float(p[6]),
                    amplitude=float(p[7]),
                    pct_chg=float(p[8]),
                    turnover=float(p[10]),
                ))
            except (ValueError, IndexError):
                continue
        if not out:
            raise RuntimeError("eastmoney 解析后无有效K线")
        return out

    def name_of(self, symbol: str) -> str:
        params = {"secid": to_secid(symbol), "fields1": "f1,f2", "fields2": "f51",
                  "klt": "101", "fqt": "1", "beg": "20260101", "end": _compact("20991231")}
        try:
            return self._get(f"{self.BASE}?{urllib.parse.urlencode(params)}")["data"].get("name", "")
        except Exception:  # noqa: BLE001
            return ""


# ============================================================
# 数据源 2：腾讯财经（标准库直连，东财被限流时的主力替补）
# ============================================================

class TencentFetcher:
    """
    东财的接口有 IP 级限流，批量抓取时容易被打回（实测连续几次后
    RemoteDisconnected）。腾讯这个接口同样支持前复权，且口径与东财一致，
    是目前最可靠的免费替补。
    """

    name = "tencent"

    BASE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self, timeout: int = 15, retries: int = 2, pause: float = 0.5):
        self.timeout = timeout
        self.retries = retries
        self.pause = pause

    def fetch(self, symbol: str, start: str, end: str,
              adjust: str = "qfq", is_index: bool = False) -> list[Bar]:
        sym = normalize_symbol(symbol)
        fq = {"qfq": "qfq", "hfq": "hfq", "": "", "none": ""}.get(adjust, "qfq")
        span = _days_between(_dash(start), _dash(end))
        count = max(80, int(span / 7 * 5) + 80)      # 交易日 ≈ 自然日 × 5/7，留足缓冲
        param = f"{sym},day,{_dash(start)},{_dash(end)},{count},{fq}"
        url = f"{self.BASE}?param={param}"

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                data = self._get(url)
                node = (data.get("data") or {}).get(sym)
                if not node:
                    raise RuntimeError("腾讯返回中无该代码")
                klines = node.get(f"{fq}day") or node.get("day") or node.get("qfqday")
                if not klines:
                    raise RuntimeError("腾讯返回中无K线数组")
                return self._parse(klines)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.pause * (attempt + 1))
        raise RuntimeError(f"tencent 请求失败: {last_err}")

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _parse(klines: list) -> list[Bar]:
        """
        腾讯字段顺序：[日期, 开, 收, 高, 低, 成交量(手), ...]
        涨跌幅腾讯不给，用前一交易日收盘价自行算（这样才有真实数值，
        而不是像某些实现那样直接填 0 —— 填 0 会让按涨跌幅过滤的逻辑失效）。
        """
        out: list[Bar] = []
        prev_close = None
        for k in klines:
            if len(k) < 6:
                continue
            try:
                o, c, h, low = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
                v = float(k[5])
            except (TypeError, ValueError):
                continue
            # 腾讯成交量单位是手；成交额在部分标的下为第 7 位（单位万元）
            amount = 0.0
            if len(k) > 6:
                try:
                    amount = float(k[6]) * 10000 if float(k[6]) < 1e6 else float(k[6])
                except (TypeError, ValueError):
                    amount = 0.0
            pct = 0.0 if not prev_close else (c / prev_close - 1) * 100
            amp = 0.0 if not prev_close else (h - low) / prev_close * 100
            out.append(Bar(date=str(k[0]), open=o, close=c, high=h, low=low,
                           volume=v, amount=amount, pct_chg=round(pct, 3),
                           turnover=0.0, amplitude=round(amp, 3)))
            prev_close = c
        if not out:
            raise RuntimeError("tencent 解析后无有效K线")
        return out


# ============================================================
# 数据源 3 / 4：AkShare、Tushare（可选依赖，装了才启用）
# ============================================================

class AkshareFetcher:
    name = "akshare"

    def available(self) -> bool:
        try:
            import akshare  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def fetch(self, symbol: str, start: str, end: str,
              adjust: str = "qfq", is_index: bool = False) -> list[Bar]:
        import akshare as ak
        sym = normalize_symbol(symbol)
        if is_index:
            df = ak.stock_zh_index_daily(symbol=sym)
            df = df[(df["date"].astype(str) >= _dash(start)) & (df["date"].astype(str) <= _dash(end))]
            return [Bar(date=str(r["date"]), open=float(r["open"]), close=float(r["close"]),
                        high=float(r["high"]), low=float(r["low"]), volume=float(r["volume"]),
                        amount=0.0, pct_chg=0.0, turnover=0.0, amplitude=0.0)
                    for _, r in df.iterrows()]
        df = ak.stock_zh_a_hist(symbol=sym[2:], period="daily",
                                start_date=_compact(start), end_date=_compact(end),
                                adjust=adjust)
        if df is None or df.empty:
            raise RuntimeError("akshare 返回空")
        out = []
        for _, r in df.iterrows():
            out.append(Bar(
                date=str(r["日期"]), open=float(r["开盘"]), close=float(r["收盘"]),
                high=float(r["最高"]), low=float(r["最低"]), volume=float(r["成交量"]),
                amount=float(r.get("成交额", 0) or 0), pct_chg=float(r.get("涨跌幅", 0) or 0),
                turnover=float(r.get("换手率", 0) or 0), amplitude=float(r.get("振幅", 0) or 0),
            ))
        return out


class TushareFetcher:
    """
    Tushare Pro。

    这个源有两个坑，都不是「写法问题」而是「数据正确性问题」，必须处理：

    1) pro.daily() 返回的是【不复权】原始价。
       实测茅台 2025-06-26 除权：前一日 close = 1435.86，当日 pre_close = 1408.26。
       注意 pre_close 是 tushare 已经按除权调整过的值 —— 换句话说，
       同一个接口里的 close 和 pre_close 分属两套口径。
       谁直接拿 close 序列做回测，就会在除权日看到一个凭空出现的 -2% 跳空，
       MA/RSI/回撤全部被污染（分红越勤的票污染越重）。
       必须用 adj_factor 自己算前复权：
           qfq(t) = raw(t) × adj(t) / adj(最新日)

    2) adj_factor 是低积分账号的限流重灾区。
       实测当前 token：1 次/小时（报错 "频率超限(1次/小时)"）。
       好在复权因子只在分红除权那天变，一年就一两次，
       所以把整段历史一次拉下来落盘缓存、按周刷新即可 ——
       日常跑日报根本不会再碰这个接口。

    指数走 index_daily（指数本身无复权概念）。token 读环境变量 TUSHARE_TOKEN。
    """

    name = "tushare"

    # 声明自己会消费 as_of（FetcherChain 据此决定是否传参）。
    # 只有 tushare 需要它：其余数据源的复权在服务端完成，本地不参与计算。
    supports_as_of = True

    # 复权因子缓存有效期（天）。由分红频率决定：一年一两次的事件，周级刷新足够，
    # 同时把「1 次/小时」的限流影响降到零。
    ADJ_TTL_DAYS = 7.0
    # 一次拉取复权因子的起点。11 年足以覆盖任何日线回测，且行数远低于单次返回上限，
    # 一个请求搞定（避免在限流接口上反复试探）。
    ADJ_START = "20150101"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.adj_dir = Path(cache_dir or CACHE_DIR) / "adj_factor"
        self.adj_dir.mkdir(parents=True, exist_ok=True)
        self.notes: list[str] = []

    def available(self) -> bool:
        return bool(os.environ.get("TUSHARE_TOKEN")) and _importable("tushare")

    # ---------------- 复权因子缓存 ----------------

    def _adj_path(self, code: str) -> Path:
        return self.adj_dir / f"{code.replace('.', '_')}.json"

    def _read_adj(self, code: str) -> Optional[dict]:
        p = self._adj_path(code)
        if not p.exists():
            return None
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # noqa: BLE001
            return None
        age = (time.time() - float(blob.get("_cached_at", 0))) / 86400.0
        if age > self.ADJ_TTL_DAYS:
            return None
        return {k: float(v) for k, v in blob.items() if not k.startswith("_")}

    def _write_adj(self, code: str, factors: dict) -> None:
        blob = dict(factors)
        blob["_cached_at"] = time.time()
        try:
            self._adj_path(code).write_text(
                json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            log.warning("复权因子缓存写入失败 %s: %s", code, exc)

    def _get_adj(self, pro, code: str) -> dict:
        """取覆盖到最新交易日的复权因子表 {YYYY-MM-DD: factor}。"""
        cached = self._read_adj(code)
        if cached:
            log.info("[tushare] 复权因子命中缓存 %s（%d 个交易日）", code, len(cached))
            return cached

        today = datetime.now().strftime("%Y%m%d")
        df = pro.adj_factor(ts_code=code, start_date=self.ADJ_START, end_date=today)
        if df is None or df.empty:
            raise RuntimeError("adj_factor 返回空")
        out = {}
        for _, r in df.iterrows():
            d = str(r["trade_date"])
            out[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = float(r["adj_factor"])
        self._write_adj(code, out)
        log.info("[tushare] 复权因子已落盘 %s（%d 个交易日）", code, len(out))
        return out

    # ---------------- 抓取 ----------------

    def fetch(self, symbol: str, start: str, end: str,
              adjust: str = "qfq", is_index: bool = False,
              as_of: Optional[str] = None) -> list[Bar]:
        import tushare as ts
        ts.set_token(os.environ["TUSHARE_TOKEN"])
        pro = ts.pro_api()
        code = to_ts_code(symbol)
        s, e = _compact(start), _compact(end)

        if is_index:
            df = pro.index_daily(ts_code=code, start_date=s, end_date=e)
            if df is None or df.empty:
                raise RuntimeError("tushare index_daily 返回空")
            return [self._row_to_bar(r)
                    for _, r in df.sort_values("trade_date").iterrows()]

        df = pro.daily(ts_code=code, start_date=s, end_date=e)
        if df is None or df.empty:
            raise RuntimeError("tushare daily 返回空")
        bars = [self._row_to_bar(r)
                for _, r in df.sort_values("trade_date").iterrows()]

        bars = self._fill_turnover(pro, code, bars, s, e)

        if adjust in ("qfq", "hfq"):
            adj = self._get_adj(pro, code)
            bars = _apply_adjust(bars, adj, adjust, as_of=as_of)
            log.info("[tushare] %s 复权口径 %s，as_of=%s", symbol, adjust, as_of or "未指定")
        return bars

    def _fill_turnover(self, pro, code: str, bars: list[Bar],
                       s: str, e: str) -> list[Bar]:
        """
        daily 不带换手率，从 daily_basic 补。
        拿不到就明确记一笔并降级为 0 —— 换手率是快照字段之一，
        与其假装有值，不如让上层知道它缺了。
        """
        try:
            db = pro.daily_basic(ts_code=code, start_date=s, end_date=e,
                                 fields="trade_date,turnover_rate")
        except Exception as exc:  # noqa: BLE001
            note = f"{code} 换手率缺失（daily_basic 失败: {str(exc)[:60]}）"
            self.notes.append(note)
            log.warning("[tushare] %s", note)
            return bars
        if db is None or db.empty:
            note = f"{code} 换手率缺失（daily_basic 返回空）"
            self.notes.append(note)
            log.warning("[tushare] %s", note)
            return bars

        tr = {}
        for _, r in db.iterrows():
            d = str(r["trade_date"])
            v = r.get("turnover_rate")
            if v is not None and v == v:          # 过滤 NaN
                tr[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = float(v)
        if not tr:
            return bars
        return [replace(b, turnover=tr.get(b.date, b.turnover)) for b in bars]

    @staticmethod
    def _row_to_bar(r) -> Bar:
        d = str(r["trade_date"])
        return Bar(
            date=f"{d[:4]}-{d[4:6]}-{d[6:]}",
            open=float(r["open"]), close=float(r["close"]),
            high=float(r["high"]), low=float(r["low"]),
            volume=float(r["vol"]),                        # 单位：手
            amount=float(r.get("amount", 0) or 0) * 1000,  # tushare 给千元，换算成元
            pct_chg=float(r.get("pct_chg", 0) or 0),
            turnover=0.0, amplitude=0.0,
        )


def _nearest_adj(adj: dict, sorted_dates: list[str], date: str) -> float:
    """
    取 <= date 的最近一个因子。

    为什么「向前取」而不是「向后取」：分红发生在某个交易日，停牌日没有记录。
    向后取会在除权日拿到除权之后的因子 —— 那就是未来函数。
    向前取最坏只是漏掉一次分红带来的偏差，方向是保守的。

    date 早于因子表起点时退回最早因子（外推），这属于数据覆盖不足，
    _apply_adjust 会另外发警告，不在这里静默吞掉。
    """
    lo, hi, best = 0, len(sorted_dates) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] <= date:
            best = sorted_dates[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        best = sorted_dates[0]
    return adj[best]


def _apply_adjust(bars: list[Bar], adj: dict, adjust: str,
                  as_of: Optional[str] = None) -> list[Bar]:
    """
    按复权因子重算价格序列。

        qfq(t) = raw(t) × adj(t) / adj(基准日)   回测用这个：
                                                 基准日当天保持原值，历史价按分红比例缩回去
        hfq(t) = raw(t) × adj(t) / adj(最早)

    【时点纪律】前复权的基准日不能无脑取「最新」。

    adj(最新) 里的「最新」是**今天**。若 as_of 落在某个分红除权日之前，
    那次除权在 as_of 当天还没有发生，却已经被用来缩放历史价格 ——
    这就是把未来的除权信息注入了历史价。传入 as_of 后，基准日会被
    锁到「as_of 当天能看到的最后一个因子」，与当时的真实价格口径一致。

    后复权天然没有这个问题（基准固定在最早一日，不随时间移动），
    所以严格回测也可以直接用 hfq —— 这是最省事的正确做法。

    为什么不对 pct_chg 重算：复权是等比例缩放，
    任何一天的涨跌幅在数学上都不变；而 tushare 的 pct_chg
    本身就是除权调整后的正确值，重算反而会引入舍入误差。

    成交量也不调整 —— 复权影响的是价格口径，
    量比、放量判定都基于成交量之间的比值，不受影响。
    """
    if not adj or adjust not in ("qfq", "hfq"):
        return bars
    dates = sorted(adj)

    if adjust == "qfq":
        base_date = dates[-1]
        if as_of is not None:
            a = asof.norm(as_of)
            visible = [d for d in dates if d <= a]
            # as_of 早于因子表起点：只能外推（下面会发警告），取最早因子
            base_date = visible[-1] if visible else dates[0]
            if base_date != dates[-1]:
                log.info(
                    "qfq 基准日按时点锁定为 %s（因子表最新为 %s，"
                    "as_of 之后发生的除权不参与本次复权）", base_date, dates[-1])
        base = adj[base_date]
    else:
        base = adj[dates[0]]
    if not base:
        return bars

    # 行情区间超出因子表覆盖范围时，端点之外的交易日只能外推。
    # 不报错，但必须让人看见 —— 否则"复权过了"会变成一句空话。
    if bars and bars[0].date < dates[0]:
        log.warning("复权因子表起点 %s 晚于行情起点 %s，前段为外推近似",
                    dates[0], bars[0].date)
    if bars and bars[-1].date > dates[-1]:
        log.warning("复权因子表终点 %s 早于行情终点 %s，后段为外推近似",
                    dates[-1], bars[-1].date)

    out = []
    for b in bars:
        f = adj.get(b.date)
        if f is None:
            f = _nearest_adj(adj, dates, b.date)
        k = f / base
        if abs(k - 1.0) < 1e-9:
            out.append(b)
            continue
        # 不做 round：复权是等比缩放，取整会破坏这个性质，
        # 让前复权和后复权算出的日收益率出现 1e-5 级的差异。
        # 精度损失对回测无意义，但"两种口径结果不一致"会让人怀疑实现有 bug。
        out.append(replace(
            b,
            open=b.open * k, close=b.close * k,
            high=b.high * k, low=b.low * k))
    return out


class BaostockFetcher:
    name = "baostock"

    def available(self) -> bool:
        return _importable("baostock")

    def fetch(self, symbol: str, start: str, end: str,
              adjust: str = "qfq", is_index: bool = False) -> list[Bar]:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败 {lg.error_msg}")
        try:
            sym = normalize_symbol(symbol)
            code = f"{sym[:2]}.{sym[2:]}"
            fq = {"qfq": "2", "hfq": "1", "": "3", "none": "3"}.get(adjust, "2")
            rs = bs.query_history_k_data_plus(
                code, "date,open,high,low,close,volume,amount,pctChg,turn",
                start_date=_dash(start), end_date=_dash(end),
                frequency="d", adjustflag=fq)
            out = []
            while rs.error_code == "0" and rs.next():
                r = rs.get_row_data()
                out.append(Bar(date=r[0], open=float(r[1]), high=float(r[2]), low=float(r[3]),
                               close=float(r[4]), volume=float(r[5] or 0),
                               amount=float(r[6] or 0), pct_chg=float(r[7] or 0),
                               turnover=float(r[8] or 0), amplitude=0.0))
            if not out:
                raise RuntimeError("baostock 返回空")
            return out
        finally:
            bs.logout()


# ============================================================
# 本地 CSV 缓存（离线兜底 + 回测加速）
# ============================================================

class CsvCache:
    """
    把抓到的日线落盘，回测重复运行时直接读缓存，不再打网络。

    注意一个容易被忽略的坑：**前复权价格会变**。
    每次分红除权后，交易所口径会把全部历史前复权价格整体重算，
    所以「昨天缓存的 qfq 序列」和「今天新抓的 qfq 序列」不是同一套数。
    如果缓存永不过期，跨分红日回测就会得到自相矛盾的曲线。

    因此这里带 TTL：默认 1 天。同一批回测任务内部可以复用，
    隔天自动失效重抓。
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_days: float = 1.0):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def path(self, symbol: str, adjust: str) -> Path:
        return self.dir / f"{normalize_symbol(symbol)}_{adjust or 'raw'}.csv"

    def is_fresh(self, symbol: str, adjust: str = "qfq") -> bool:
        p = self.path(symbol, adjust)
        if not p.exists():
            return False
        if self.ttl_days <= 0:
            return True
        age_days = (time.time() - p.stat().st_mtime) / 86400.0
        return age_days <= self.ttl_days

    def read(self, symbol: str, adjust: str = "qfq") -> list[Bar]:
        p = self.path(symbol, adjust)
        if not p.exists():
            return []
        out = []
        with open(p, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    out.append(Bar(
                        date=row["date"], open=float(row["open"]), close=float(row["close"]),
                        high=float(row["high"]), low=float(row["low"]),
                        volume=float(row["volume"]), amount=float(row["amount"]),
                        pct_chg=float(row["pct_chg"]), turnover=float(row["turnover"]),
                        amplitude=float(row["amplitude"]),
                    ))
                except (KeyError, ValueError):
                    continue
        return out

    def write(self, symbol: str, adjust: str, bars: Iterable[Bar]) -> None:
        p = self.path(symbol, adjust)
        cols = ["date", "open", "close", "high", "low", "volume",
                "amount", "pct_chg", "turnover", "amplitude"]
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for b in bars:
                w.writerow(b.as_dict())


# ============================================================
# 降级链调度器
# ============================================================

DEFAULT_PRIORITY = ["eastmoney", "tencent", "akshare", "tushare", "baostock"]
_REGISTRY = {
    "eastmoney": EastMoneyFetcher,
    "tencent": TencentFetcher,
    "akshare": AkshareFetcher,
    "tushare": TushareFetcher,
    "baostock": BaostockFetcher,
}


class FetcherChain:
    """
    按优先级逐个尝试，任一成功即返回。

    与 DSA 的差异：每次调用都会返回 (bars, source_name)，
    并把失败原因累积到 self.last_errors，出问题时一眼看出死在哪一环。
    """

    def __init__(self, priority: Optional[list[str]] = None,
                 cache: Optional[CsvCache] = None, use_cache: bool = True):
        self.priority = priority or DEFAULT_PRIORITY
        self.cache = cache if cache is not None else CsvCache()
        self.use_cache = use_cache
        self.last_errors: dict[str, str] = {}
        self.last_source: str = ""
        # 最近一次 fetch 的时点审计台账（as_of 未指定时为 None）
        self.audit: Optional[asof.Audit] = None
        self._instances: dict[str, object] = {}

    def _get(self, name: str):
        if name not in self._instances:
            cls = _REGISTRY.get(name)
            self._instances[name] = cls() if cls else None
        return self._instances[name]

    def _emit(self, bars: list[Bar], sym: str, source: str) -> list[Bar]:
        """
        数据层的统一出口 —— 无论来源是网络、缓存还是过期缓存，都在这里裁一次。

        「不信任上游」是刻意的：缓存路径、降级路径、将来的新数据源
        都会经过这里，加一处等于加所有。
        """
        if self.audit is None:
            return bars
        return asof.clip_bars(bars, self.audit.as_of,
                              label=f"{sym}/{source}", audit=self.audit)

    def fetch(self, symbol: str, start: str, end: str,
              adjust: str = "qfq", is_index: bool = False,
              force_refresh: bool = False,
              as_of: Optional[str] = None) -> tuple[list[Bar], str]:
        self.last_errors = {}
        # 台账的生命周期：as_of 变了就换一本。
        # 单次调用时每次都重置；批量抓取时由 fetch_many 预先建好，
        # 这里复用同一本 —— 否则每只标的都会把前一只的裁剪记录冲掉。
        if as_of is None:
            self.audit = None
        elif self.audit is None or self.audit.as_of != asof.norm(as_of):
            self.audit = asof.Audit(as_of=as_of)
        sym = normalize_symbol(symbol)

        # 0) 时点门控第一道：请求区间本身钉在 as_of 内。
        #    未来数据连请求都不发出去 —— 从源头断掉泄漏，也顺手省下数据源配额
        #    （对 tushare 这种带限流的源，这不是小事）。
        if as_of is not None:
            req_s, req_e = asof.clip_range(start, end, as_of)
            if (req_s, req_e) != (start, end):
                log.info("[%s] 时点门控：请求区间 %s~%s → %s~%s",
                         sym, start, end, req_s, req_e)
            if asof.norm(start) > self.audit.as_of:
                self.audit.note(f"请求区间起点 {asof.norm(start)} 晚于 as_of，"
                                f"整体压缩为空区间（取不到数据是正确结果）")
            start, end = req_s, req_e

        # 1) 缓存优先（必须同时满足：未过期 且 已覆盖请求区间）
        if self.use_cache and not force_refresh and self.cache.is_fresh(sym, adjust):
            cached = self.cache.read(sym, adjust)
            cov = _coverage(cached, start, end)
            if cov is not None:
                log.info("[%s] 命中本地缓存 %d 根", sym, len(cov))
                self.last_source = "cache"
                return self._emit(cov, sym, "cache"), "cache"

        # 2) 依次降级
        for name in self.priority:
            inst = self._get(name)
            if inst is None:
                continue
            if hasattr(inst, "available") and not inst.available():
                self.last_errors[name] = "依赖未安装或未配置 Token"
                continue
            try:
                # 只有声明了 supports_as_of 的源才需要这个参数
                # （目前只有 tushare：复权在本地算，基准日必须时点化）
                extra = {"as_of": as_of} if getattr(inst, "supports_as_of", False) else {}
                bars = inst.fetch(sym, start, end, adjust=adjust,
                                  is_index=is_index, **extra)
                if not bars:
                    raise RuntimeError("返回 0 根K线")
                if self.use_cache:
                    try:
                        self.cache.write(sym, adjust, bars)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("缓存写入失败(不影响主流程): %s", exc)
                log.info("[%s] 由 %s 提供 %d 根K线", sym, name, len(bars))
                self.last_source = name
                self.last_errors.pop(name, None)
                return self._emit(bars, sym, name), name
            except Exception as exc:  # noqa: BLE001
                self.last_errors[name] = f"{type(exc).__name__}: {exc}"
                log.warning("[%s] %s 失败: %s", sym, name, exc)
                continue

        # 3) 全部失败，退回任意缓存
        if self.use_cache:
            cached = self.cache.read(sym, adjust)
            if cached:
                # 过期缓存同样要切到请求区间 —— 这里原本是「原样返回整段」，
                # 越界的尾巴就是未来数据。降级路径最容易漏，也最容易出事。
                sliced = _coverage(cached, start, end) or cached
                log.warning("[%s] 全部数据源失败，使用过期缓存 %d 根%s",
                            sym, len(sliced),
                            "" if len(sliced) == len(cached) else
                            f"（已从 {len(cached)} 根切到请求区间）")
                self.last_source = "cache(stale)"
                return self._emit(sliced, sym, "cache(stale)"), "cache(stale)"

        raise NoDataSourceError(sym, self.last_errors)

    def fetch_many(self, symbols: Iterable[str], start: str, end: str,
                   adjust: str = "qfq", is_index: bool = False,
                   force_refresh: bool = False,
                   as_of: Optional[str] = None) -> tuple[dict[str, list[Bar]], dict[str, str]]:
        """
        批量抓取。单只失败不中断整体，失败清单写日志并跳过。

        注意 symbols 先物化成 list —— 否则生成器会在 len() 检查里被耗尽。
        """
        sym_list = list(symbols)
        result, sources = {}, {}
        # 批量抓取要累积审计：先建好台账，循环里的 fetch 会复用它。
        # 不这么做的话，「裁掉了多少」只会剩下最后一只标的的数字 ——
        # 审计漏报比不报更糟，因为它看起来是完整的。
        self.audit = asof.Audit(as_of=as_of) if as_of is not None else None
        for idx, s in enumerate(sym_list):
            try:
                bars, src = self.fetch(s, start, end, adjust=adjust,
                                       is_index=is_index, force_refresh=force_refresh,
                                       as_of=as_of)
                result[s] = bars
                sources[s] = src
            except Exception as exc:  # noqa: BLE001
                log.error("[%s] 获取失败: %s", s, exc)
            if idx < len(sym_list) - 1:
                # 加抖动，避免固定节奏触发风控
                time.sleep(0.4 + random.random() * 0.4)
        return result, sources


# 模块级便捷函数
_CHAIN = FetcherChain()


def get_bars(symbol: str, start: str, end: str, adjust: str = "qfq",
             is_index: bool = False, force_refresh: bool = False,
             chain: Optional[FetcherChain] = None,
             as_of: Optional[str] = None) -> tuple[list[Bar], str]:
    ch = chain or _CHAIN
    return ch.fetch(symbol, start, end, adjust=adjust, is_index=is_index,
                    force_refresh=force_refresh, as_of=as_of)


def get_benchmark(start: str, end: str, code: str = "sh000300",
                  chain: Optional[FetcherChain] = None,
                  as_of: Optional[str] = None) -> list[Bar]:
    """基准指数（默认沪深300），失败返回空列表而不是抛错。"""
    try:
        bars, _ = (chain or _CHAIN).fetch(code, start, end, adjust="",
                                          is_index=True, as_of=as_of)
        return bars
    except Exception as exc:  # noqa: BLE001
        log.warning("基准 %s 获取失败: %s", code, exc)
        return []


# ============================================================
# 工具
# ============================================================

def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def _compact(d: str) -> str:
    return str(d).replace("-", "").replace("/", "")[:8]


def _dash(d: str) -> str:
    s = _compact(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else str(d)


def _coverage(bars: list[Bar], start: str, end: str) -> Optional[list[Bar]]:
    """
    缓存是否可用：起点必须已覆盖，终点允许差几天（当日数据可能还没落盘）。
    不满足则返回 None，让调用方去打网络。
    """
    if not bars:
        return None
    s, e = _dash(start), _dash(end)
    start_ok = bars[0].date <= s
    end_ok = bars[-1].date >= e or _days_between(bars[-1].date, e) <= 5
    if not (start_ok and end_ok):
        return None
    sliced = [b for b in bars if s <= b.date <= e]
    return sliced or None


def _days_between(d1: str, d2: str) -> int:
    try:
        return abs((datetime.strptime(d2, "%Y-%m-%d") - datetime.strptime(d1, "%Y-%m-%d")).days)
    except Exception:  # noqa: BLE001
        return 9999
