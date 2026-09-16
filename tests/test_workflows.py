#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workflow 静态守卫 —— 把「CI 必须真的跑起来」变成可以本地证伪的断言
============================================================

**起因是一次真实事故**：为了让测试不往仓库里写"假记忆 / 假辩论记录"，把

    env:
      DSA_MEMORY_DIR: ${{ runner.temp }}/dsa-memory

写进了 **job 级** `env:`。语法完全合法（PyYAML 解析通过、缩进无懈可击），
但 `runner` 上下文在 `jobs.<id>.env` 里**不可用** —— GitHub 直接判定
「workflow 无效」，整条流水线**零 job**、立刻变红，而且 **run 日志里一个字都没有**。

于是出现了最难查的一类故障：**CI 红了，但没有任何日志能告诉你为什么**。
本地把 Python 代码、YAML 语法全查一遍都查不出来；最后是靠
`gh workflow run ci.yml` 的 422 响应才拿到那行真正的报错。

这个测试就是那次事故的止血带，五条守卫各自带对照组：

    [1] 上下文可用性   job 级 env 里出现 runner.* → 拦下（就是那次事故）
    [2] 表达式根名      ${{ }} 里的根必须是 GitHub 认可的那些，防拼错
    [3] 结构完整性      每个 job 有 runs-on / 有 step；uses 钉了版本；无 TAB
    [4] 承诺守卫在场    ci.yml 至少要保住那条"零依赖"流水线
    [5] 新测试必须接上 CI  tests/ 下每个测试文件都得真的被执行

**为什么这条守卫必须存在**：项目所有其他测试都在验证"代码对不对"，
而这条验证的是**"验证本身有没有真的在跑"**。一个静默失效的 CI 会让
后面每一条测试都变成摆设，比没有 CI 更危险 —— 因为它会给你绿灯。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WF_DIR = ROOT / ".github" / "workflows"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


# ============================================================
# 工具：一个极简的「按缩进认层级」的 workflow 扫描器
#   刻意不依赖 PyYAML —— 零依赖是项目对外承诺，守卫自己不能先破坏它
# ============================================================

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_blank(line: str) -> bool:
    s = line.strip()
    return s == "" or s.startswith("#")


def job_level_env_entries(text: str) -> list[tuple[int, str, str]]:
    """取出 `jobs.<id>.env` 下的 (行号, 键, 值)。

    只认**缩进 4** 的 `env:` —— 这就是 job 级。
    step 级 env 缩进是 8（`steps:` 6 → `- name:` 6 → `env:` 8），天然不会被误收。
    """
    lines = text.split("\n")
    out: list[tuple[int, str, str]] = []
    in_jobs = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        ind = _indent(raw)
        if ind == 0 and s == "jobs:":
            in_jobs = True
            i += 1
            continue
        if in_jobs and ind == 0 and not _is_blank(raw) and not s.startswith("-"):
            return out  # 离开 jobs 块
        if in_jobs and ind == 4 and s == "env:":
            j = i + 1
            while j < len(lines):
                l3 = lines[j]
                s3 = l3.strip()
                i3 = _indent(l3)
                if _is_blank(l3):
                    j += 1
                    continue
                if i3 <= 4:
                    break
                if i3 == 6 and ":" in s3:
                    key, _, val = s3.partition(":")
                    out.append((j + 1, key.strip(), val.strip()))
                j += 1
            i = j
            continue
        i += 1
    return out


def job_blocks(text: str) -> dict[str, tuple[int, list[str]]]:
    """{job_id: (起始行号, 该 job 的行列表)} —— 同样只靠缩进判断。"""
    lines = text.split("\n")
    blocks: dict[str, tuple[int, list[str]]] = {}
    start = None
    for i, raw in enumerate(lines):
        if _indent(raw) == 0 and raw.strip() == "jobs:":
            start = i + 1
            break
    if start is None:
        return blocks
    i = start
    cur: str | None = None
    cur_start = 0
    cur_lines: list[str] = []
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        ind = _indent(raw)
        if not _is_blank(raw) and ind == 0:
            break
        if ind == 2 and s.endswith(":") and not s.startswith("-"):
            if cur is not None:
                blocks[cur] = (cur_start, cur_lines)
            cur = s[:-1].strip()
            cur_start = i + 1
            cur_lines = []
        elif cur is not None:
            cur_lines.append(raw)
        i += 1
    if cur is not None:
        blocks[cur] = (cur_start, cur_lines)
    return blocks


_EXPR_RE = re.compile(r"\$\{\{(.+?)\}\}")
_STR_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")

# GitHub 认可的上下文根（含函数名，防把 format(...) 误判成上下文）
ALLOWED_ROOTS = {
    "github", "needs", "strategy", "matrix", "vars", "secrets", "inputs",
    "runner", "env", "job", "steps", "jobs",
}
ALLOWED_FUNCS = {"format", "toJSON", "fromJSON", "always", "success", "failure", "cancelled", "hashFiles"}
# runner 只在这些位置可用；job 级 env 不在其中（这就是那次事故）
RUNNER_UNSAFE_VALUE = "runner."


def context_roots(text: str) -> list[tuple[int, str]]:
    """返回 [(行号, 表达式根名)]，字符串字面量先剔掉再取根。"""
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        for m in _EXPR_RE.finditer(line):
            body = _STR_RE.sub(" ", m.group(1))
            idents = _IDENT_RE.findall(body)
            if not idents:
                continue
            root = idents[0]
            if root in ALLOWED_FUNCS:
                continue
            out.append((lineno, root))
    return out


def uses_entries(text: str) -> list[tuple[int, str]]:
    out = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        s = line.strip()
        if s.startswith("- uses:") or s.startswith("uses:"):
            out.append((lineno, s.split("uses:", 1)[1].strip()))
    return out


def missing_tests_in_ci(ci_text: str, tests_dir: Path) -> list[str]:
    """tests/ 下应该被执行的测试文件里，哪些没在 ci.yml 出现。"""
    if not tests_dir.is_dir():
        return []
    want = []
    for name in sorted(os.listdir(tests_dir)):
        if not name.endswith(".py"):
            continue
        if name.startswith("test_") or name == "selftest.py":
            want.append(name)
    return [n for n in want if n not in ci_text]


# ============================================================
# [1] 检查器自证：先证明它真的能抓到，也真的不会乱抓
# ============================================================

BROKEN_JOB_ENV = """\
name: t
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    env:
      DSA_MEMORY_DIR: ${{ runner.temp }}/dsa-memory
    steps:
      - run: echo hi
"""

STEP_ENV_OK = """\
name: t
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - name: x
        env:
          X: ${{ runner.temp }}/dsa-memory
        run: echo hi
"""

PLAIN_JOB_ENV = """\
name: t
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    env:
      FOO: bar
    steps:
      - run: echo hi
"""

UNKNOWN_ROOT = """\
name: t
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: echo ${{ runners.temp }}
"""

KNOWN_ROOTS = """\
name: t
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - env:
          A: ${{ secrets.X }}
          B: ${{ github.ref }}
          C: ${{ matrix.v }}
        run: echo ok
"""


def main() -> int:
    print("[1] 检查器自证 —— 先证明它会抓、且不乱抓（否则后面全是空转）")

    bad = [e for e in job_level_env_entries(BROKEN_JOB_ENV) if RUNNER_UNSAFE_VALUE in e[2]]
    check("job 级 env 里的 runner.* 被抓到", len(bad) == 1,
          f"命中 {len(bad)} 条 @L{bad[0][0] if bad else '-'}")

    step_entries = job_level_env_entries(STEP_ENV_OK)
    check("对照组：step 级 env 里的 runner.* 不被误报（证明是按层级判断）",
          step_entries == [], f"取到 {len(step_entries)} 条 job 级条目")

    plain = [e for e in job_level_env_entries(PLAIN_JOB_ENV) if RUNNER_UNSAFE_VALUE in e[2]]
    check("对照组：job 级 env 用普通字面量不报（不是见 env 就报）", plain == [],
          f"job 级条目 {len(job_level_env_entries(PLAIN_JOB_ENV))} 条，命中 {len(plain)}")

    roots_bad = [r for _, r in context_roots(UNKNOWN_ROOT) if r not in ALLOWED_ROOTS]
    check("拼错的上下文根 runners.temp 被抓到",
          roots_bad == ["runners"], f"非法根 {roots_bad}")

    roots_ok = [r for _, r in context_roots(KNOWN_ROOTS) if r not in ALLOWED_ROOTS]
    check("对照组：secrets / github / matrix 都是合法根，不报", roots_ok == [],
          f"非法根 {roots_ok}")

    # ============================================================
    print()
    print("[2] 真实 workflow：job 级 env 里不许出现 runner.*（那次事故的根因）")

    ci = (WF_DIR / "ci.yml").read_text(encoding="utf-8")
    daily = (WF_DIR / "daily.yml").read_text(encoding="utf-8")

    for fname, text in (("ci.yml", ci), ("daily.yml", daily)):
        hits = [e for e in job_level_env_entries(text) if RUNNER_UNSAFE_VALUE in e[2]]
        check(f"{fname} 的 job 级 env 未引用 runner.*", hits == [],
              f"命中 {len(hits)} 条" + (f" → L{hits[0][0]}" if hits else ""))

    # ============================================================
    print()
    print("[3] 真实 workflow：${{ }} 的上下文根必须合法")

    for fname, text in (("ci.yml", ci), ("daily.yml", daily)):
        roots = context_roots(text)
        bad_roots = sorted({r for _, r in roots if r not in ALLOWED_ROOTS})
        # 断言「确实扫到了表达式」—— 防止 0 个表达式也被判通过（空转陷阱）
        check(f"{fname} 的表达式根全部合法", bad_roots == [],
              f"扫到 {len(roots)} 处表达式，非法根 {bad_roots or '无'}")
        check(f"{fname} 确实含表达式（守卫没有空转）", len(roots) > 0,
              f"{len(roots)} 处")

    # ============================================================
    print()
    print("[4] 结构完整性：job 得真的能跑")

    for fname, text in (("ci.yml", ci), ("daily.yml", daily)):
        check(f"{fname} 无 TAB 字符", "\t" not in text, "")

        lines = text.split("\n")
        check(f"{fname} 首行是 name:（被解析成功的标志之一）",
              lines[0].startswith("name:"), f"首行={lines[0][:40]!r}")

        blocks = job_blocks(text)
        check(f"{fname} 至少 1 个 job", len(blocks) >= 1, f"jobs={list(blocks)}")

        no_runner = [j for j, (_, ls) in blocks.items()
                     if not any(l.strip().startswith("runs-on:") for l in ls)]
        check(f"{fname} 每个 job 都声明了 runs-on", no_runner == [],
              f"缺 runs-on：{no_runner}")

        no_steps = [j for j, (_, ls) in blocks.items()
                    if not any(l.strip() == "steps:" for l in ls)]
        check(f"{fname} 每个 job 都有 steps", no_steps == [], f"缺 steps：{no_steps}")

        unpinned = [(ln, u) for ln, u in uses_entries(text)
                    if "@" not in u and not u.startswith("./")]
        check(f"{fname} 所有 uses 都钉了版本或为本地路径", unpinned == [],
              f"未钉版本：{unpinned}")

    check("ci.yml 保住了「零依赖」这条流水线（项目对外承诺的守卫）",
          "stdlib-only" in job_blocks(ci) and "零依赖" in ci,
          f"jobs={list(job_blocks(ci))}")

    # ============================================================
    print()
    print("[5] 新测试必须真的接上 CI —— 加了测试却忘了挂，等于没加")

    missing = missing_tests_in_ci(ci, ROOT / "tests")
    check("tests/ 下每个测试文件都被 ci.yml 执行", missing == [],
          f"漏挂：{missing or '无'}")

    # 对照组：故意用一份缺文件的 ci.yml 去问，检查器必须报出来
    missing_ctrl = missing_tests_in_ci("name: CI\n# 什么都不跑\n", ROOT / "tests")
    check("对照组：缺失的测试确实会被报出来（守卫没有空转）",
          len(missing_ctrl) > 0, f"报出 {len(missing_ctrl)} 个")

    # ============================================================
    failed = [r for r in RESULTS if not r[1]]
    print()
    print("=" * 66)
    print(f"workflow 静态守卫：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过")
    if failed:
        print("失败项：")
        for name, _, detail in failed:
            print(f"  - {name}  {detail}")
    print("=" * 66)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
