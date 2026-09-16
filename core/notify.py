#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推送层 —— 多通道投递
============================================================
抄 daily_stock_analysis 的通道覆盖，但用标准库实现（不依赖 requests）。

支持：企业微信机器人 / 飞书机器人 / Telegram / Discord / Slack / 邮件 / 控制台 / 本地文件

设计原则：任何单个通道失败都不能影响其他通道，也不能让主流程挂掉。
推送是"最后一公里"，它坏了不应该导致分析白跑。
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

log = logging.getLogger("notify")

MAX_LEN = 3800  # 单条消息上限，超出自动分片


class Notifier:
    """按配置逐个投递，逐个记录成败。"""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.wecom = cfg.get("wecom_webhook", "")
        self.feishu = cfg.get("feishu_webhook", "")
        self.tg_token = cfg.get("telegram_bot_token", "")
        self.tg_chat = cfg.get("telegram_chat_id", "")
        self.discord = cfg.get("discord_webhook", "")
        self.slack = cfg.get("slack_webhook", "")
        self.mail = cfg.get("email", {}) or {}
        self.console = cfg.get("console", True)
        self.log_file = cfg.get("log_file", "")
        self.results: list[tuple[str, bool, str]] = []

    # ---------- 对外统一入口 ----------

    def send(self, title: str, content: str) -> list[tuple[str, bool, str]]:
        self.results = []
        chunks = _split(content)
        for i, chunk in enumerate(chunks, 1):
            suffix = f"（{i}/{len(chunks)}）" if len(chunks) > 1 else ""
            for name, fn in (
                ("console", self._console),
                ("wecom", self._wecom),
                ("feishu", self._feishu),
                ("telegram", self._telegram),
                ("discord", self._discord),
                ("slack", self._slack),
                ("email", self._email),
                ("file", self._file),
            ):
                if not self._enabled(name):
                    continue
                try:
                    fn(f"{title}{suffix}", chunk)
                    self.results.append((name, True, ""))
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s 推送失败: %s", name, exc)
                    self.results.append((name, False, str(exc)))
        return self.results

    def _enabled(self, name: str) -> bool:
        return bool({
            "console": self.console,
            "wecom": self.wecom,
            "feishu": self.feishu,
            "telegram": self.tg_token and self.tg_chat,
            "discord": self.discord,
            "slack": self.slack,
            "email": self.mail.get("smtp_host") and self.mail.get("to"),
            "file": self.log_file,
        }.get(name))

    # ---------- 各通道实现 ----------

    def _console(self, title: str, body: str) -> None:
        print("\n" + "=" * 62)
        print(title)
        print("=" * 62)
        print(body)
        print("=" * 62 + "\n")

    def _wecom(self, title: str, body: str) -> None:
        self._post_json(self.wecom, {
            "msgtype": "markdown",
            "markdown": {"content": f"**{title}**\n{_md_to_wecom(body)}"}})

    def _feishu(self, title: str, body: str) -> None:
        self._post_json(self.feishu, {
            "msg_type": "text",
            "content": {"text": f"{title}\n{body}"}})

    def _telegram(self, title: str, body: str) -> None:
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        self._post_json(url, {
            "chat_id": self.tg_chat, "text": f"{title}\n{body}",
            "disable_web_page_preview": True})

    def _discord(self, title: str, body: str) -> None:
        self._post_json(self.discord, {"content": f"**{title}**\n{_clip(body, 1900)}"})

    def _slack(self, title: str, body: str) -> None:
        self._post_json(self.slack, {"text": f"*{title}*\n{_clip(body, 3000)}"})

    def _email(self, title: str, body: str) -> None:
        m = self.mail
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(title, "utf-8")
        msg["From"] = m.get("sender") or m.get("username", "")
        to = m["to"]
        msg["To"] = to if isinstance(to, str) else ",".join(to)

        host, port = m["smtp_host"], int(m.get("smtp_port", 465))
        cls = smtplib.SMTP_SSL if m.get("use_ssl", True) else smtplib.SMTP
        with cls(host, port, timeout=30) as s:
            if not m.get("use_ssl", True):
                s.starttls()
            s.login(m.get("username", ""), m.get("password", ""))
            s.sendmail(msg["From"], to if isinstance(to, list) else [to], msg.as_string())

    def _file(self, title: str, body: str) -> None:
        p = Path(self.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n\n## {title}\n{body}\n")

    # ---------- HTTP ----------

    @staticmethod
    def _post_json(url: str, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        # 企微/飞书成功时 code=0 / StatusCode=0，失败也返回 200，需检查 body
        try:
            obj = json.loads(text)
            code = obj.get("errcode", obj.get("code", obj.get("StatusCode", 0)))
            if code not in (0, "0", None):
                raise RuntimeError(f"通道返回错误: {text[:200]}")
        except json.JSONDecodeError:
            pass

    def summary(self) -> str:
        if not self.results:
            return "未配置任何推送通道"
        ok = [n for n, s, _ in self.results if s]
        bad = [(n, e) for n, s, e in self.results if not s]
        s = f"推送成功 {len(ok)} 次" + (f"（{', '.join(sorted(set(ok)))}）" if ok else "")
        if bad:
            s += f"；失败 {len(bad)} 次: " + "; ".join(f"{n}({e[:60]})" for n, e in bad)
        return s


# ---------- 工具 ----------

def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 20] + "\n...(已截断)"


def _split(text: str, limit: int = MAX_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def _md_to_wecom(md: str) -> str:
    """企业微信 markdown 不支持表格，转成紧凑列表，避免刷屏乱码。"""
    if "|---" not in md:
        return md
    lines, out = md.split("\n"), []
    for ln in lines:
        s = ln.strip()
        if s.startswith("|---") or set(s) <= {"|", "-", ":", " "}:
            continue
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            out.append("· " + " | ".join(c for c in cells if c))
        else:
            out.append(ln)
    return "\n".join(out)
