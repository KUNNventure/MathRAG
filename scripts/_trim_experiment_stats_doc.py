#!/usr/bin/env python
"""Trim redundant sections from RAG优化实验统计.md; keep §一 table intact."""

from pathlib import Path

STATS = Path(r"C:\Users\xsk\Desktop\RAG项目优化\RAG优化实验统计.md")

PART3_MINIMAL = """
---

# 第三部分 · 附录（精简）

> 逐题明细、分 Phase 长表已删（与 §一 重复或无效）。查数：`results/<run_id>/report.json`、桌面 `实验汇总_MASTER.json`。

## 口径与公式

| 指标 | Phase 1/2 子集 | 全量 / 上线 |
|------|----------------|-------------|
| CR / CP | ✅（汇报用 **@ref**） | ✅ |
| Faith | — | ✅（边界用**调整规则**） |
| AR | — | **不跑** |

```
Score   = 0.4×CR + 0.3×Faith + 0.3×CP
Score_p = 0.6×CR + 0.4×CP
```

**禁止混用**：E004b CR/CP + E005 Faith；@ref + 非源答案 Faith；用 E005 旧 CR/CP 证「检索变差」；21 题均值当 48 题。

**裁判** qwen-turbo · **生成** qwen-max · 新实验：`run_ref_eval_pipeline.py`（跳过传统 CR/CP）。

## 最终配置（生产）

| 配置项 | 值 |
|--------|-----|
| collection | `math_textbooks` c1000 |
| fusion_top_k | 10；dense/sparse 20；RRF 60；**dense:bm25 1:1** |
| 检索 | hybrid + **rerank v2 中文** |
| 生成 | **v3.4** `answer_generation_v3.4.txt` |
| **汇报 Score** | **E004b 0.8083**（@ref + 调 Faith） |

**已废弃**：`results/baseline/` 仅 11/48 题有效 → 以 **baseline_v2 / E001b** 为准。

## 历史说明

- **Phase 0** `baseline/`：API 欠费导致 37/48 无 RAGAS，**勿引用**表中 0.9485 等数字。
- **p1.1_full_48**：与 `p1.1-full48`（E002）重复，以 E002 为准。
- **1.1-a**：无独立目录，聚合 0.6901/0.5007 仅日志留存。

"""


def main() -> None:
    text = STATS.read_text(encoding="utf-8")

    # --- header: drop experiment count block ---
    import re

    text = re.sub(
        r"\n### 实验次数统计.*?(?=\n---\n\n# 第一部分)",
        "\n",
        text,
        flags=re.DOTALL,
    )

    # --- after §一 table: remove 读表防混 + 3.4-r2 节 + trim tail to part 2 ---
    # Find end of §一 table (last row 重复勿用)
    marker_table_end = "| —   | 重复勿用               | p1.1_full_48"
    idx = text.find(marker_table_end)
    if idx == -1:
        raise SystemExit("table end marker not found")
    idx = text.find("\n\n", idx + len(marker_table_end))
    part2_start = text.find("# 第二部分")
    if part2_start == -1:
        raise SystemExit("part 2 not found")

    after_table = """

†Faith 与 CR/CP 须同次、同答案；E001b/E004b 合法合成 Score。‡**旧口径**（CP 多绑生成答案）；公平 CR/CP 见 **@ref** 行。

**@ref + Faith 合法 Score**（`SCORES_AT_REF.json`）：

| ID | CR@ref | CP@ref | Faith | Score_p | **Score** |
|----|--------|--------|-------|---------|-----------|
| **E001b** | 0.8049 | 0.6712 | 0.8853 | 0.7514 | **0.7889** |
| **E004b** | 0.8066 | 0.6748 | **0.944**（调） | 0.7539 | **0.8083** |
| **E002b** | 0.7726 | 0.6739 | 0.8348† | 0.7331 | **0.7617** |
| **P1-1.5-full48-ref** | 0.7910 | 0.6801 | **0.9509**（调） | 0.7466 | **0.8057** |
| Δ E004b−E001b | +0.002 | +0.004 | — | +0.0025 | **+0.019** |
| Δ P1-1.5-ref−E004b | −0.016 | +0.005 | +0.007 | −0.007 | **−0.0026** |

**Phase 2**：定稿 c1000 k=10 + v3.4，**不采纳 c1500**（见 §一 合并行 vs 3.4-r2 摘21）。

"""

    text = text[:idx] + after_table + "\n---\n\n" + text[part2_start:]

    # --- Part 2: merge 2.7 into 2.3, remove 2.7 section, move 2.6 to end ---
    text = re.sub(
        r"\| \*\*1\*\* \| \*\*不采纳\*\* 无 rerank 上线 \|.*?\n",
        "| **1** | **不采纳** 无 rerank 全量 | 全量 @ref **0.8057 < 0.8083**；子集 CR 平、CP +0.037，全量 CR −0.016 | 见 §一 `1.5-full48-v34-ref` |\n",
        text,
    )
    text = re.sub(
        r"\n## 2\.7 2026-05-18 补充：无 rerank.*?(?=\n---\n\n# 第三部分)",
        "\n",
        text,
        flags=re.DOTALL,
    )

    # Move 2.6 to end of part 2 (before part 3)
    m_26 = re.search(
        r"\n## 2\.6 个人决策留白.*?(?=\n---\n\n# 第三部分)",
        text,
        flags=re.DOTALL,
    )
    if m_26:
        block_26 = m_26.group(0)
        text = text[: m_26.start()] + text[m_26.end() :]
        insert_at = text.find("# 第三部分")
        text = text[:insert_at] + block_26 + "\n" + text[insert_at:]

    # Reorder 2.2 before 2.1? User asked 2.2 first under methodology - optional; keep 2.1 first for narrative

    # --- Replace entire Part 3 with minimal appendix ---
    text = re.sub(
        r"\n# 第三部分 · 附录细节.*",
        PART3_MINIMAL,
        text,
        flags=re.DOTALL,
    )

    # Clean duplicate --- 
    text = re.sub(r"\n---\n\n---\n", "\n---\n", text)
    text = re.sub(
        r"> \*\*结构\*\*：.*?\n",
        "> **人读唯一文档**。机读：`实验汇总_MASTER.json`、`SCORES_AT_REF.json`。  \n",
        text,
        count=1,
    )

    STATS.write_text(text, encoding="utf-8")
    lines = len(text.splitlines())
    print(f"Wrote {STATS} ({lines} lines)")


if __name__ == "__main__":
    main()
