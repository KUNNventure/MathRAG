# RAG项目优化踩坑记录

## PDF解析器中文乱码

- **现象**：原项目MarkItDown解析中文PDF，输出乱码
- **原因**：MarkItDown对中文PDF字体映射支持差，数学符号和中文混排导致大量乱码
- **解决**：保留MarkItDown（pymupdf4llm输出纯文本无结构，更不可用），在pdf_loader.py新增两类字符映射表：
  - `_MOJIBAKE_LATIN_MAP`：修复字体映射错误（"狓"→"x"、"犃"→"A"等52个映射）
  - `_MATH_SYMBOL_MAP`：修复数学符号（"槡"→"√"，后续按需扩展）
  - 配合NFKC归一化处理全角字符
- **遗留**：数学符号仍有部分丢失（x²→x2、根号缺失），后续通过LLM ChunkRefiner修复

## HierarchicalSplitter未被调用

- **现象**：settings.yaml已改成hierarchical，但ingestion后chunk无chapter_num等metadata
- **原因**：Dashboard的Ingestion Manager页面splitter下拉框硬编码了["recursive","semantic","fixed_length"]，没有hierarchical选项。每次通过Dashboard导入时，UI把运行参数覆盖成默认的recursive
- **解决**：修改ingestion_manager.py，从SplitterFactory.list_providers()动态读取可用splitter
- **教训**：配置改了不代表生效，要加print确认代码是否真正被调用

## 图片占位符堆积成超大chunk

- **现象**：最后一个chunk 52486字符，全是[IMAGE: xxx]占位符
- **原因**：文档末尾的图片占位符没有归属到任何章节，被合并成一个巨型chunk
- **影响**：超大chunk导致embedding失败（超token限制(8016)），整个upsert阶段报错，Vectors=0
- **解决**：在document_chunker.py新增图片占位符比例过滤——文本占比<20%的chunk直接丢弃；去掉了对大chunk用RecursiveSplitter二次切分的逻辑（章节级切分粒度已够）
- **备注**：这个过滤逻辑后来发现是精修文本丢失的一个原因（见后续 为解决PDF解析失败而过度设计缓存机制 ），已优化

## metadata未写入chunk — TOC跳过逻辑失效

- **现象**：chunk的metadata里没有chapter_num/section_num/grade/volume
- **根因**：`_find_body_start`无法区分目录行和正文章节标题。测试PDF格式良好，真实教材pymupdf4llm把目录输出成"章名+页码粘连"的单行
- **修复**：
  - hierarchical_splitter.py：新增`_has_toc_page_number()`检测行尾页码数字；SECTION_RE加re.MULTILINE
  - pipeline.py：Stage4 LLM Transform临时注释禁用
  - scripts/ingest.py：新增--grade/-g和--volume/-V CLI参数
- **验证**：此修复仅在测试PDF上验证通过，真实教材上仍然失败（见踩坑7）

## 图片描述生成覆盖率低

- **现象**：开启图片描述生成后，只有20-30%的图片成功生成描述
- **原因**：切分器把很多完整图片切成条状，解析进一些非图片元素，Vision LLM对这部分内容难以识别
- **影响**：不影响核心功能（文本知识点检索为主）
- **决策**：关闭ImageCaptioner。数学几何图与知识点检索场景不重合
- **连带优化**：关闭图片提取（extract_images=False）

## HierarchilSplitter在真实教材上反复失败，最终放弃

### 背景与初始动机

选型HierarchicalSplitter的逻辑：教材有"章/节/知识点"层级结构→通用RecursiveSplitter按字数盲切会破坏结构→层级感知切分可提升Context Recall。逻辑本身没错，但低估了"让正则稳定匹配真实教材格式"的工程难度。

### 踩坑过程

**第1轮**：HierarchicalSplitter代码生成→测试PDF跑通（6 chunk，metadata正确）。切到真实教材（七下，148206字符）：Chunks=24，总字符~4万（丢失10万+），无chapter_num/section_num。

**第2轮**：定位到`_find_body_start`无法区分目录行和正文标题。修复：新增`_has_toc_page_number()`。测试PDF验证通过，切到真实教材还是失败。

根因：
1. `_clean_body_lines`把页眉行整行丢弃——但页眉和正文在**同一行**（"第七章 相交线与平行线因为∠1与∠2互补..."），整行丢弃=丢失正文
2. `SECTION_RE`要求`^(\d+\.\d+)\s+`——但真实教材是`7.1相交线`（无空格）、`7.1.1`（三节编号）
3. TOC crammed格式每条拥挤5-6个标题，单行匹配不符合任何已有规则
4. 封面/版权/印刷信息段被误判为正文，真实正文反而被跳过

### 真实教材格式 vs 假设格式

|      | 假设（Markdown输出）  | 实际（MarkItDown输出）                        |
| ---- | --------------- | --------------------------------------- |
| 章标题  | `# 第七章 相交线与平行线` | `第七章 相交线与平行线7.1相交线在上一章中...`（标题+节号+正文粘连） |
| 节标题  | `## 7.1 相交线`    | 行内嵌入，无独立行，`7.1相交线`（无空格）                 |
| 目录   | 每章一行独立条目        | 3行超长行，5-6个章节条目挤在一起，页码粘连                 |
| 页眉   | 无               | 每页重复`第七章 相交线与平行线`，与正文同在一行               |
| 数学符号 | 正常              | `槡`→√需修复，`²`→2需修复，全角数字需归一化              |

### 最终决策：切换RecursiveSplitter

| 维度                       | HierarchicalSplitter | RecursiveSplitter         |
| ------------------------ | -------------------- | ------------------------- |
| 状态                       | 修了3轮还有bug            | 稳定可用                      |
| chunk数（七下）               | 24（错乱）               | **150**                   |
| Vectors                  | 0（embedding失败）       | **150**（1024-dim）         |
| chapter/section metadata | 目标有，实际无              | 无（LLM MetadataEnricher可补） |
| 总耗时                      | 3轮修复~10h              | 10分钟切换                    |

## Vectors=0 — 空chunk导致整批embedding全部失败

- **现象**：ingestion完成，Chunks=150，但Vectors=0
- **根因**：**双重bug叠加**
  1. `DenseEncoder.encode()` line 106-110——如果任一个chunk的text为空，直接raise ValueError，整批embedding全部不执行
  2. `BatchProcessor.process()` line 164-171——catch了异常但**没有log**，静默吞掉，dense_vectors保持空列表
- **连带根因**：ChunkRefiner rule-based清理后可能把某个chunk清空（只有白空格/符号），触发bug 1
- **修复**：
  - `DenseEncoder.encode()`：跳过空chunk不报错，返回zero vector占位，最终映射回原chunk索引
  - `BatchProcessor.process()`：加`logger.error()`打印异常详情
  - `pipeline.py` Stage 4后：加空chunk过滤（text.strip() < 10 → 丢弃），防止空chunk进入编码阶段
- **教训**：静默吞异常是埋雷，锁死了30分钟的调试时间

## PDF源文件路径混乱 — 多Collection引用了不同来源的教材（Claude Code锅）

- **现象**：`math_textbooks` (c1000) 的chunk source_path指向 `C:\Users\xsk\Downloads\`，`math_textbooks_c500` (c500) 的source_path指向 `C:\Users\xsk\WorkBuddy\...\rj_g7_top.pdf`，而用户实际要用的教材在 `C:\Users\xsk\Desktop\人教版初中数学课本\`。三套不同路径的PDF。
- **原因**：多次导入过程中没有统一PDF来源。第一次（c1000）从Downloads目录导入了2024/2025修订版，第二次（c500）从WorkBuddy项目的临时目录导入了另一套文件。用户自己也没意识到有三个不同位置存放了教材PDF。
- **教训**：
  1. **导入前先确认数据源**：不要在多个位置存放同名文件，数据源路径应写死在优化文档里
  2. **Collection的source_path metadata是诊断关键**：出问题时先查metadata，确认chunk从哪个文件来的
  3. **三个存放位置盘点**：
    - `Downloads/` — c1000用的2024/2025修订版（可能是正确的，但非用户指定）
    - `WorkBuddy/.../textbook/rj_g7_top.pdf` — c500七上用的（可能是另一个项目残留）
    - `Desktop/人教版初中数学课本/` — 用户指定的正确文件

## 为解决PDF解析失败而过度设计缓存机制

- **现象**：七下PDF markitdown解析做 `extract-only` 时失败，`rj_g7_bot.txt` 输出0字节。CURRENT_STATE.md 设计了复杂的迂回方案：从ChromaDB `math_textbooks` 中提取已导chunk文本拼接为缓存文件→用 `--from-cache` 重新切片导入。
- **根因**：七下PDF解析失败本质上是 **用错了PDF文件**，不是markitdown真的有问题。WorkBuddy项目中的那本七下PDF可能损坏或格式不同，但Desktop的正确文件可以正常解析。
- **解决**：删掉 `pipeline.py` 和 `ingest.py` 中的 `--from-cache`/`--extract-only`/`RAW_TEXT_CACHE_DIR`/`_save_raw_text()`/`_load_from_cache()` 全部缓存相关代码，改回直接从PDF导入。
- **教训**：
  1. **先排除数据源问题，再假设代码有bug**：markitdown解析失败→先确认PDF文件本身是否正确，而不是立刻写workaround
  2. **workaround代码要能干净回退**：这次添加的缓存逻辑耦合在pipeline核心路径中（Stage 2分支、构造参数传递），回退时需要改两个文件多处。如果当初设计成独立脚本/CLI可选功能会更干净
  3. **遇到未知失败先诊断根因，不要急于设计替代方案**——这个问题从"七下解析失败"到"设计缓存→失败→怀疑用错文件→验证→删除代码"，绕了一大圈

## RAGAS 口径混用导致「假退步 / 假提升」

- **现象**：
  - E005 `final-v3.4` 表观 CR/CP 远低于 baseline，像「检索崩盘」；
  - E002−E001 显示 Score +0.013、CP +0.035，像「rerank 大提升」；
  - 把 @ref 重算的 CR/CP 与另一次跑的 Faith 拼成 Score，又得出 E005b=0.769 等中间结论，前后矛盾。
- **根因**：
  1. 旧跑分用 `ContextPrecisionWithoutReference`，CP 绑**生成答案**；CR 有时未稳定用 golden `reference_answer` → baseline CR **高估 ~0.05**。
  2. **机制调查（同 chunk、同答案，仅换 @ref）**：相对参考答案，旧口径**几乎唯一系统性变化**是 **CR↑、CP↓**（p1.1 子集均值 **+0.14 / −0.11**；48 题 **+0.05 / −0.03**）。旧表「v2 CR 暴涨」多为口径差，不是 rerank 单因。
  3. Phase 3 边界**合规拒答** → RAGAS Faith=0，产品反而对；须用 **faith_adjusted（0.944）** 谈 E004，不能只用 raw 0.819。
  4. `p1.1` 21 题 **cr_cp_only** 无 Faith；`p1.1-full48` 是**整链重跑**才有 Faith 0.918 —— 不是「只测 CR/CP 却改了 Faith 表」。
  5. **1.1-a** 跑分后 **被 1.1-b 覆盖** `results/p1.1/`，仅保留聚合三数字，**无逐题 / 无 @ref**。
- **合法做法**（已定稿）：
  - **E001b**：同 baseline 答案 + CR/CP@ref → Faith 0.8853 → Score **0.7889**；
  - **E004b**：同 p3.4-r2 答案 + CR/CP@ref + **调整 Faith 0.944** → Score **0.8083**；
  - 检索对比看 **Score_p**（E004b−E001b **+0.0025**）。
- **文档**：执行文档「优化复盘与收口定稿」；`SCORES_AT_REF.json`。
- **教训**：**分维度报告**；综合 Score 仅在同次、同答案、同口径下比较；面试勿单报 E005 旧 0.742。
