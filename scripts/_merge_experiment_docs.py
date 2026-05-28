#!/usr/bin/env python
"""One-off: merge human-readable experiment docs into RAG优化实验统计.md"""
from pathlib import Path

DESKTOP = Path(r"C:\Users\xsk\Desktop\RAG项目优化")
STATS = DESKTOP / "RAG优化实验统计.md"
EXEC = DESKTOP / "RAG优化实验执行文档.md"
TRACK = DESKTOP / "RAG优化实验追踪.md"
FULL = DESKTOP / "实验数据全量统计.md"
INDEX = DESKTOP / "文档索引与整理说明.md"

STUB = """# 已合并

> **2026-05-18**：正文已并入 **[RAG优化实验统计.md](./RAG优化实验统计.md)**。  
> 请只维护该文件；机读表仍为 `实验汇总_MASTER.json`、`SCORES_AT_REF.json`。

| 原章节 | 现位置 |
|--------|--------|
{rows}

"""

PART2 = """
---

# 第二部分 · 个人思考与决策

> 原《RAG优化实验执行文档》复盘与定稿章节。数据表见上文 §一；逐题见 `results/*/report.json`。

## 2.1 一句话结论（2026-05-18 收工）

**上线**：c1000、k=10、hybrid、**rerank v2 中文**、生成 **v3.4** → **E004b Score 0.8083**（@ref CR/CP + 调 Faith 0.944）。  
**不采纳**：c1500；RRF 0.7:0.3（1b.3 全量输 E002）；**无 rerank 全量**（1.5-full48-v34-ref **0.8057**，略低于 E004b **−0.0026**）。

## 2.2 合法 Score 与禁止拼接

| 组合 | 合法 | 说明 |
|------|:----:|------|
| **E001b** | ✅ | @ref CR/CP + baseline **同答案** Faith → **0.7889** |
| **E004b** | ✅ | @ref + Faith **调 0.944** → **0.8083** |
| **P1-1.5-full48-ref** | ✅ | 对照实验；无 rerank + v3.4 @ref |
| E004b CR/CP + E005 Faith | ❌ | 不同次全量 |
| 任意 @ref + 非源答案 Faith | ❌ | 答案不一致 |

## 2.3 各 Phase 决策与原因

| Phase | 决策 | 原因（摘要） |
|-------|------|-------------|
| **0** | 用 **E001b** 作 baseline 对照 | 旧 CR 高估 ~0.05；E005 旧分不能否定检索 |
| **1** | **v2 rerank + RRF 1:1 + k10** | 旧子集排序 1.1-b 最优；E002 全量 0.8114；@ref E004b−E001b Score_p +0.0025 |
| **1** | **不采纳** 1b.3 融合 0.7:0.3 | 子集 @ref 赢、全量 E002b **0.762 < 0.811** |
| **1** | **不采纳** 无 rerank 上线 | 全量 @ref Score **0.8057 < 0.8083**；CR@ref −0.016 |
| **2** | **否定 c1500** | k5 合并 Faith/Score 仍输 3.4-r2 子集对照 |
| **3** | **v3.4-r2 上线** | 边界拒答、韦达合规、控幅；产品优先于 raw Faith |

## 2.4 评测「尺子」定稿（D1–D4）

| 决策 | 内容 |
|------|------|
| **D1** | CR/CP 汇报以 **@reference** 为准（`rescore_ragas_cr_cp.py`） |
| **D2** | 综合 Score 仅 **同次、同答案** 合成（E001b、E004b 等） |
| **D3** | Faith 产品向用 **调整规则**（边界合规拒答、韦达「无法确定」计 1） |
| **D4** | 旧口径 CR/CP 表内保留 ‡，**不作主胜负依据** |

**机制**：旧→@ref 系统性 **CR↑、CP↓**（p1.1 子集约 +0.14 / −0.11）→ 旧表「v2 CR 大涨」含口径差，**Δ 幅度勿写进汇报**。

## 2.5 工程合理性与面试口径

| 维度 | 常见「有用」 | 本项目 |
|------|-------------|--------|
| 检索 @ref | +0.02～+0.05 | Score_p **+0.0025**（偏小有方向） |
| 综合 Score | +0.02～+0.05 | E001b→E004b **+0.019** |
| 产品向 | 误答率降即值 | v3.4 拒答/不编造（人工抽检优先） |

> 48 题 golden：Phase1 rerank @ref 约 **+0.25%**；v3.4 边界/幻觉产品向；合成 Score **0.808** 与 v1 全量 **0.811** 同量级。**勿用 E005 旧 0.742 否定检索**。

## 2.6 个人决策留白（请你执笔）

- [ ] 为何接受 v3.4 尽管 raw Faith 低于 p1.1-full48
- [ ] 上线后优先监控：边界误答 / 韦达 / 跨章节
- [ ] 无 rerank 全量 0.8057：是否仅作对照、不做 A/B
- [ ] 是否投入 E002b（p1.1-full48 @ref）或线上 A/B

## 2.7 2026-05-18 补充：无 rerank（1.5）

| 对比 | 子集 21（1.5′ vs 1.1-b′） | 全量 48（vs E004b） |
|------|---------------------------|---------------------|
| CR@ref | **持平** 0.673 | **−0.016** |
| CP@ref | **+0.037** | **+0.005** |
| Score | — | **0.8057 vs 0.8083（−0.0026）** |

→ 子集不能代表全量；**维持 v2 rerank**。

"""

PART3_HEADER = """
---

# 第三部分 · 附录细节

> 原《实验数据全量统计》《RAG优化实验追踪》细节。Phase 分表见下；全量逐题见 `results/<run_id>/report.json`。

## 附录 A · 评测口径与流程

（原《RAG优化实验追踪》）

"""

APPENDIX_A = """
### 指标与公式

| 指标 | Phase 1/2 | Phase 3 / 全量 | 说明 |
|------|-----------|----------------|------|
| CR / CP | ✅ | ✅（@ref 汇报） | 天花板 / 平衡 |
| Faith | — | ✅ | 最贵；边界用调整规则 |
| AR | — | — | **全程不跑** |

```
Score   = 0.4×CR + 0.3×Faith + 0.3×CP
Score_p = 0.6×CR + 0.4×CP          # Phase 1/2 子集排序
```

### 固定 21 题子集

- 文件：`golden_subset_21.json`
- 中间实验复用；全量裁决见 Phase 切换条件

### Phase 顺序（实际执行）

**Baseline → Phase 1 检索 → Phase 3 Prompt → Phase 2 切片（已停）**

### 禁止混用

- E004b CR/CP + E005 Faith
- @ref CR/CP + 非源答案 Faith
- 用 E005 旧 CR/CP 证明「检索变差」
- 21 题子集分数直接等同 48 题全量

### 裁判与生成

- 裁判：**qwen-turbo**（全实验统一）
- 生成：**qwen-max**
- 新实验推荐：`run_ref_eval_pipeline.py`（跳过传统 CR/CP → @ref → Faith 调整）

---

## 附录 B · 全量 48 题聚合对照

（原《实验数据全量统计》§零；@ref 行为主）

| 实验 ID | 目录 | Rerank | CR | CP | Faith | Score | Score_p |
|---------|------|--------|-----|-----|-------|-------|---------|
| E001 | `baseline_v2/` | v0 | 0.8516‡ | 0.6393‡ | 0.8853 | 0.7980 | 0.7667 |
| **E001b** | `baseline_v2-ref-cr-cp/` | v0 | **0.8049** | **0.6712** | 0.8853 | **0.7889** | **0.7514** |
| **E002** | `p1.1-full48/` | v2 | 0.8339‡ | 0.6743‡ | 0.9183 | **0.8114** | 0.7701 |
| E003/E004 | `p3.4/` `p3.4-r2/` | v2 | — | — | faith-only | — | — |
| **E004b** | `p3.4-r2-ref-cr-cp/` | v2 | **0.8066** | **0.6748** | **0.944**调 | **0.8083** | **0.7539** |
| E005 | `final-v3.4/` | v2+v3.4 | 0.7681‡ | 0.6335‡ | 0.8143 | 0.7416 | 0.7143 |
| **P1-1.5-full48-ref** | `p1.5-full48-v34-ref-cr-cp/` | **无** | **0.7910** | **0.6801** | **0.9509**调 | **0.8057** | **0.7466** |
| E002b | `p1b.3-full48-ref-cr-cp/` | v2 0.7:0.3 | 0.7726 | 0.6739 | 0.8348† | 0.7617† | 0.7331 |

‡旧口径；†Faith 与 @ref 非同次合成时注意 meta。

**关键 Δ**：E004b−E001b Score **+0.019**；P1-1.5-full48-ref−E004b **−0.0026**。

---

## 附录 C · Phase 分阶段明细

"""


def main() -> None:
    text = STATS.read_text(encoding="utf-8")

    # Update header
    text = text.replace(
        "# RAG优化实验统计\n\n> 数据来源：",
        "# RAG优化实验统计（唯一人读文档）\n\n"
        "> **结构**：上文 **第一部分 实验汇总** → **第二部分 个人思考** → **第三部分 附录细节**。  \n"
        "> 机读：`实验汇总_MASTER.json`、`SCORES_AT_REF.json`（`results/_MASTER_SUMMARY.json` 同源）。  \n"
        "> 数据来源：",
    )
    text = text.replace(
        "收口全文：《RAG优化实验执行文档》→ **「优化复盘与收口定稿」**  \n"
        "机器表：`SCORES_AT_REF.json`（仓库 `results/_SCORES_AT_REF.json`）\n",
        "",
    )
    text = text.replace(
        "详见《实验数据全量统计》零-B。",
        "详见下文 **附录 B/C**。",
    )
    text = text.replace(
        "详见《执行文档》「Phase 1 全量：相对 Baseline 的优化与不足」。",
        "见 **§2.3** 与下文 Phase 1 附录。",
    )

    # Part 1 marker
    text = text.replace(
        "## 配置分层（必读，勿混）",
        "---\n\n# 第一部分 · 实验汇总\n\n## 配置分层（必读，勿混）",
        1,
    )

    # Insert Part 2 before Phase 0
    marker = "## 二、Phase 0: Baseline"
    if marker in text and PART2.strip() not in text:
        text = text.replace(marker, PART2 + "\n## 二、Phase 0: Baseline", 1)

    # Part 3 header before Phase 0
    text = text.replace(
        "## 二、Phase 0: Baseline",
        PART3_HEADER + APPENDIX_A + "\n## 二、Phase 0: Baseline",
        1,
    )

    # Remove old section 7 doc consolidation (merged into header)
    import re
    text = re.sub(
        r"\n---\n\n## 七、收工：实验文档怎么留.*?(?=\n---\n\n## 八、)",
        "\n\n---\n\n> **文档合并说明**：原《执行文档》《追踪》《全量统计》已并入本文件；旧文件仅留跳转页。\n",
        text,
        flags=re.DOTALL,
    )

    STATS.write_text(text, encoding="utf-8")
    print(f"Wrote merged {STATS} ({len(text.splitlines())} lines)")

    stubs = [
        (EXEC, "执行规则、Phase 设计、复盘定稿"),
        (TRACK, "评测口径、Phase 顺序、禁止项"),
        (FULL, "全量聚合、零-B 子集机制、逐题明细索引"),
    ]
    for path, desc in stubs:
        rows = f"| {desc} | 见 **RAG优化实验统计.md** 第二/三部分 |"
        path.write_text(STUB.format(rows=rows), encoding="utf-8")
        print(f"Stub: {path.name}")

    INDEX.write_text(
        """# RAG项目优化 — 文档索引（2026-05-18 合并版）

## 人读（只维护 1 个）

| 文件 | 用途 |
|------|------|
| **[RAG优化实验统计.md](./RAG优化实验统计.md)** | 汇总表 + 个人思考 + 附录细节 |

## 机读（脚本自动刷）

| 文件 | 用途 |
|------|------|
| `实验汇总_MASTER.json` | = `results/_MASTER_SUMMARY.json` |
| `SCORES_AT_REF.json` | 合法 @ref 合成 Score |

## 已合并为跳转页（勿再改正文）

- `RAG优化实验执行文档.md`
- `RAG优化实验追踪.md`
- `实验数据全量统计.md`

## 其他

- `golden_test_set.json` / `golden_subset_21.json` — 评测集
- `RAG踩坑记录_汇总.md` — 文章素材
- `RAG项目交付总文档_20260517.md` — 交付总览
""",
        encoding="utf-8",
    )
    print("Updated 文档索引")


if __name__ == "__main__":
    main()
