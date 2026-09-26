"""
today.py —— 生成「今天该做什么」的清单
========================================================================

想法：计划都写在文件里（PLAN_12W.md 的周任务 + leetcode/SCHEDULE.md 的当天题号），
但没人愿意每天去翻两份文件。这个脚本把它们合成一张当天清单。

    cd E:\\llm-lab
    python common/today.py

输出三块：
    · 今日任务（项目 / 刷题 / 日报）
    · 本周任务原文（来自 PLAN_12W.md）
    · 进度提示（这是 12 周里的第几周）

它只是把已有的东西整理出来，不产生新信息 —— 所以无论以后把清单送到
哪个渠道（邮件 / 待办 App / 在线文档），内容都是同一份，改一处即可。

========================================================================
参数：
    --date 2026-10-05     指定日期（默认今天），用来预演某天会看到什么
    --plain               纯文本输出（去掉框线，方便塞进邮件正文）
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "PLAN_12W.md"
SCHEDULE = ROOT / "leetcode" / "SCHEDULE.md"

PLAN_START = dt.date(2026, 9, 28)      # 12 周版的 W1 周一
PLAN_WEEKS = 12
PLAN_END = PLAN_START + dt.timedelta(weeks=PLAN_WEEKS) - dt.timedelta(days=1)

WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# --------------------------------------------------------------------------
# 解析计划文件
# --------------------------------------------------------------------------
def parse_plan_weeks(text: str) -> dict[int, dict[str, str]]:
    """从 PLAN_12W.md 里抽出每一周那一行，返回 {周号: {任务, 验收, 小时}}。"""
    weeks: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        m = re.match(r"\|\s*\*{0,2}W(\d+)\*{0,2}\s*\|", s)
        if not m:
            continue
        n = int(m.group(1))
        cells = [c.strip() for c in s.strip("|").split("|")]
        # 单元格：[周, 日期, 任务, 验收, 小时]（P1' 有 5 列，其余 4 列）
        if len(cells) >= 5:
            weeks[n] = {"date": cells[1], "task": cells[2], "accept": cells[3], "hours": cells[4]}
        elif len(cells) == 4:
            weeks[n] = {"date": cells[1], "task": cells[2], "accept": "", "hours": cells[3]}
    return weeks


def parse_schedule(text: str) -> dict[tuple[int, int], dict[str, str]]:
    """从 SCHEDULE.md 抽题号表，返回 {(月, 日): {day, num, title, level}}。"""
    out: dict[tuple[int, int], dict[str, str]] = {}
    pat = re.compile(
        r"\*\*D(\d+)\*\*\s*·\s*(\d{2})\.(\d{2})\s*·\s*\*\*(\d+)\*\*\s*(.+?)\s*·\s*(简单|中等|困难)"
    )
    for m in pat.finditer(text):
        mm, dd = int(m.group(2)), int(m.group(3))
        out[(mm, dd)] = {
            "day": m.group(1), "num": m.group(4),
            "title": m.group(5).strip(), "level": m.group(6),
        }
    return out


# --------------------------------------------------------------------------
def build(today: dt.date, plain: bool = False) -> str:
    lines: list[str] = []
    bar = "" if plain else "─" * 54

    head = f"今日清单 · {today:%Y-%m-%d}（{WEEKDAY_CN[today.weekday()]}）"
    lines.append(head)
    if bar:
        lines.append(bar)

    delta = (today - PLAN_START).days
    if delta < 0:
        lines.append(f"计划尚未开始 —— 距 12 周版启动还有 {-delta} 天（{PLAN_START:%m-%d} 周一）。")
        lines.append("")
        lines.append("启动前的准备（PLAN_12W.md 第 8 节）：")
        lines.append("  [ ] 补完 tensor_api_drill.py 的 ex19 / ex20（先跑一遍 week02/nn_basics.py）")
        lines.append("  [ ] 注册 AutoDL 并完成实名认证（1–2 天，W2 要租卡）")
        lines.append("  [ ] 把 leetcode/0011_max_v.py 提交掉（09-15 就写了）")

    week_no = delta // 7 + 1
    if delta >= 0:
        if week_no > PLAN_WEEKS:
            lines.append(f"12 周计划已于 {PLAN_END:%Y-%m-%d} 结束（现在是结束后第 {delta - PLAN_WEEKS * 7 + 1} 天）。")
        else:
            lines.append(f"12 周计划的第 {week_no} 周 / 共 {PLAN_WEEKS} 周"
                         f"　（本周 {PLAN_START + dt.timedelta(weeks=week_no - 1):%m.%d}"
                         f" – {PLAN_START + dt.timedelta(weeks=week_no) - dt.timedelta(days=1):%m.%d}）")

        weeks = parse_plan_weeks(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else {}
        w = weeks.get(week_no)
        if w:
            lines.append("")
            lines.append("【本周任务】" + (f"（预算 {w['hours']} 小时）" if w["hours"] else ""))
            lines.append("  " + w["task"])
            if w["accept"]:
                lines.append("  验收：" + w["accept"])

        # 并行线（2026-09-26 v2：目标改为「大模型方向的寒假实习」）
        extras: list[str] = []
        if today <= dt.date(2026, 10, 25):
            extras.append("SQL · 3 小时/周（底线：JOIN / 子查询 / 窗口函数能写）")
        if today >= dt.date(2026, 10, 18):
            extras.append("投递 · 大模型实习，简历 v1 可用（滚动招聘，招满即止）")
        if today >= dt.date(2026, 11, 23):
            extras.append("投递 · 简历 v2 定稿，大规模投递")
        if extras:
            lines.append("")
            lines.append("【并行】")
            for e in extras:
                lines.append("  [ ] " + e)

        # 今日刷题
        if SCHEDULE.exists():
            sched = parse_schedule(SCHEDULE.read_text(encoding="utf-8"))
            item = sched.get((today.month, today.day))
            lines.append("")
            lines.append("【今日刷题】")
            if item:
                lines.append(f"  [ ] D{item['day']} · {item['num']} {item['title']} · {item['level']}"
                             f"　（限时 15/25/40 分钟，超时看题解后必须关掉重写）")
            else:
                lines.append("  （今天的题号表里没有条目 —— 可能是二刷期或已刷完）")

        # 固定动作
        lines.append("")
        lines.append("【固定动作】")
        lines.append("  [ ] 健身 1.5 小时 —— 固定锚点，不在可砍清单里")
        lines.append(f"  [ ] 日报 journal/{today:%Y-%m-%d}.md（≤10 行，重点是「卡点/报错」那一栏）")
        lines.append("  [ ] 提交：git add -A && git commit && git push")
        if today.weekday() == 6:
            lines.append("  [ ] 周日：更新 README 进度表 + 检查这周 commit 有没有断档")

    if bar:
        lines.append(bar)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成今天的任务清单")
    ap.add_argument("--date", help="指定日期 YYYY-MM-DD（默认今天）")
    ap.add_argument("--plain", action="store_true", help="纯文本输出（去掉框线）")
    args = ap.parse_args()

    if args.date:
        today = dt.date.fromisoformat(args.date)
    else:
        today = dt.date.today()

    if not PLAN.exists():
        sys.exit(f"找不到 {PLAN}")

    print(build(today, plain=args.plain))


if __name__ == "__main__":
    main()
