# 训练数据问题记录

记录各训练池中发现的数据问题、处理状态和复查方法。新发现的问题追加到对应池子下；处理后更新状态，不删除记录。

| 状态 | 含义 |
|---|---|
| 已处理 | 已修正并生成新池（旧池保留，不覆盖） |
| 搁置 | 已确认存在、量小，暂不处理；有明确的重新评估条件 |
| 正常 | 看似异常，核实后属于数据本身的合理内容 |

## 代码池

### 已处理

**D1 stage1 子集与代码能力不对口**（2026-10-22）
- 现象：`data/sft/opencoder`（OpenCoder opc-sft-stage1 `filtered_infinity_instruct`）的回答中 33% 不含代码、44% 含 Python 函数；代码专家 HumanEval 比 Base 低约 6 点（按官方 harness 计分）。
- 处理：改用 stage2 `educational_instruct` + `evol_instruct`，生成 `data/sft/opencoder_stage2`。64 步短训验证：HumanEval 83.5 / MBPP 72.0，对比 Base 84.8 / 72.0，基本不掉点。
- 证据：`configs/experiments/probe-code2.json`、`runs/20261022-probe-code2-expert-c/`。

**D2 回答开头的固定模板与客套话**（2026-10-22）
- 现象：`educational_instruct` 100% 的回答以 "Here is the code to solve this problem:" 开头；`evol_instruct` 34% 以 "Certainly! / Sure! / Here's…" 开头。evol 的来源链条：CodeAlpaca_20K → evol-codealpaca-v1（GPT-4 作答）→ Magicoder-Evol-Instruct-110K → OpenCoder 原样收入。经 MOPD 放大后扩散到指令题：三领域 MOPD 在 IFEval 上以客套话开头的回答从 1.5% 升到 29.6%，IFEval 严格分判错、宽松分判对的题从 18 增至 43。直接对这份数据做 SFT 的模型没有出现这种扩散（6.7%–7.6%）。
- 处理：`scripts/prepare_code_pool.py --set stage2clean` 中的 `wash_opener` 只删回答开头那一句独立的套话，且仅在后面还有实质内容时才删，正文和代码不改；生成 `data/sft/opencoder_stage2_clean`。
- 证据：`data/sft/opencoder_stage2_clean/preparation_report.json`；正式重跑规格 `configs/experiments/scale256-code2clean.json`。

### 搁置

重新评估条件：scale256-code2clean 的结果中，若三领域 MOPD 在 IFEval 上以客套话开头的比例没有回到约 5%，或者回答中出现 "As an AI" 之类的拒答话术，就把下列几项合并成下一版数据一起处理。

| 编号 | 问题 | 数量（占 `opencoder_stage2_clean` 训练集 198,821 条） | 拟处理方式 |
|---|---|---|---|
| D3 | 纯拒答或免责声明，例如 "As an AI, I don't have the ability to access and run files…"、"I'm sorry, you need to provide the error message…"，全部来自 evol（GPT-4） | 330（0.17%）；另有 639 条带同类字样但后面给了代码，可以接受 | 删除不含代码的拒答 |
| D4 | 完全相同的回答：题目说法不同、解答一字不差，最多同一份解答重复 9 次，几乎都来自 educational | 6,058 行（3.0%），对应 2,551 份不同的回答 | 相同回答只保留一条 |
| D5 | DS-1000 风格的脚手架残留，HTML 被转义后留在回答末尾，如 `&lt;/code&gt;`、`END SOLUTION` | 560（0.3%） | 删除这些行，或去掉残留的标记 |

### 正常

- 10% 的回答没有代码块，非 Python 语言约占 15%：evol 本来就包含解释类和多语言题目。

## 数学池（`data/sft/numina_v3`，10,239 条）

### 搁置

| 编号 | 问题 | 数量 | 拟处理方式 |
|---|---|---|---|
| D6 | 回答第一行是孤零零的 ": "（上游格式残留），解答内容本身正常 | 295（2.9%） | 去掉开头的 ": " |

### 正常

- 700 条没写 `\boxed{}`：其中 598 条题目要求证明（prove / show that），以 ∎ 收尾。
- 100 条含中文：中文竞赛题本身的内容，如"华杯赛"。

## 指令池（`data/sft/smol_constraints`，29,135 条）

### 正常

- 2,363 条以 "Is there anything else I can help with?" 结尾：其中 2,261 条的题目要求用这句话结尾（这本来就是约束遵循数据）。
- "SECTION 1"、`<<poem of joy>>` 等开头：题目约束要求的格式。
- 225 行回答完全相同、170 条短于 40 字符、14 条带拒答字样：量小，未发现系统性问题。

## 复查方法

在服务器 `/ssd/lzx/mopd` 下，对各训练池的 `train.jsonl` 统计：

- 最常见的首行和末行（检查有没有一字不差重复出现的模板）；
- 长度分布、极短回答、完全相同的回答；
- 拒答用语：回答前 300 字符匹配 `I'm sorry / I cannot / as an AI / as a language model`；
- 中文字符；
- 各领域专属指标：数学看 `\boxed{}` 个数；代码看代码块、编程语言、数据来源。

客套话开头的判定：回答开头匹配 `certainly|sure|of course|absolutely|great question|I'd be happy|here's/here is/here are`。模型输出的统计口径：各 run 在统一协议 `unified-imc-plainqa-greedy4096-v1` 下的 `cases.jsonl`，instruction 领域的 541 条回答。
