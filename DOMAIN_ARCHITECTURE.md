# 领域扩展与数据抽样
## 当前代码
- src/mopd/backends/hf.py：加载冻结教师、学生 LoRA；核对词表一致。
- scripts/train_batched_opd.py：现有数学训练入口，仍在运行；直接读取 Numina，打乱后顺序取样。
- src/mopd/training/dual_template.py：原数学提示、教师前缀与 sampled reverse-KL 策略梯度。
- src/mopd/training/batching.py：变长序列补齐、答案位置掩码、分块计算 log probability。
- scripts/eval_math_opd.py：MATH 生成和答案判定。
- scripts/watch_fixed_kl.py：固定答案位置上的完整词表 KL；不是准确率。
- monitor/：数学运行的看板；尚未接入指令遵循运行。

## 本次新增
- src/mopd/data/domains.py：按领域构造学生、教师提示。instruction 原样使用指令，不追加数学解题要求。
- src/mopd/data/pool.py：JSONL 数据池验证、固定 seed 无放回抽样、训练/留出集规范化文本查重。
- scripts/prepare_instruction_pool.py：通用 JSONL 导入，保留原始字段到 metadata，记录来源与文件哈希。
- scripts/build_instruction_pilot.py：公开 Alpaca 指令入口，丢弃参考答案；独立 dev 256，IFEval 全部保留评测。
- scripts/plan_domain_batches.py：无需 GPU 即可生成逐步 batch 计划。
- scripts/train_domain_opd.py：新领域训练入口，复用加载器、log probability 与 loss。旧入口保留用于已有数学实验。
- configs/instruction_opd.json：Qwen3-4B-Instruct-2507 教师，Qwen3-4B-Base 学生，64 步，每步 4 条，microbatch 2。
- scripts/test_domain_pool.py：模板隔离、抽样复现、无重复、耗尽报错、拒绝非训练数据。

## 一条训练数据
JSONL 字段：id、domain、split、source、prompt；metadata 可选。
domain 是 math 或 instruction。模型只看到 prompt 经对应模板处理后的文本，来源标签不进入提示。
教师接收学生原始 response token IDs，不能 decode 后再重新分词。

## 数据池怎样选 batch
先检查数据与留出集，再按 seed 打乱整个单领域池，按每步 4 条切分。
不放回；池不足就报错，不偷偷循环。
batch_plan.json 保存完整计划；rollouts.jsonl 保存样本 ID、来源、领域和学生版本。
同一步的两个 microbatch 使用同一学生版本；完成 4 条梯度累积才更新一次。
双领域入口已新增：scripts/train_multidomain_opd.py；配额抽样见 src/mopd/data/multidomain.py，配置和多卡实现见 DUAL_DOMAIN.md。

## 指令遵循数据与验收边界
Alpaca 仅作为通用指令的入口验证，不代表严格约束训练数据，也不直接保证 IFEval 提升。
下载来源和字节哈希保存在 data/instruction/raw；使用上游数据应核对其许可。
规范化完全相同检查不能消除语义近重复或模型预训练污染。
IFEval 不参与训练和选 batch。正式报告应包含严格 prompt-level 和 instruction-level 通过率。
IFEval 官方判定器和通用评测入口 scripts/eval_domains.py 已接入，541 条规则规格均可执行；真实模型前后评测仍待训练后运行。
新教师权重已按固定 HF revision 哈希校验，词表一致、模板已核验。GPU 1 保留；最终双领域分配见 DUAL_DOMAIN.md。
双领域两步 smoke 已通过，64 步训练已启动；原数学任务保持运行。

## 使用
在 /ssd/lzx/mopd 中 source scripts/env.sh 后：
1. 数据下载完成后：python scripts/build_instruction_pilot.py
2. 验证计划：python scripts/plan_domain_batches.py --config configs/instruction_opd.json --output data/instruction/pilot_batch_plan.jsonl
3. 教师准备与评测基线完成后：python scripts/train_domain_opd.py --config configs/instruction_opd.json

## 新增第三个领域
添加数据池、domains.py 中的提示规则、对应 teacher 配置和独立 evaluator。
无需重写 token 评分、loss 或批处理。暂不把数学与指令的效果混成单一指标。

## 已生成产物
- data/instruction/train.jsonl：51,746 条 Alpaca 训练指令，仅使用 instruction/input，不使用参考答案。
- data/instruction/dev.jsonl：256 条独立开发样本。
- data/instruction/ifeval.jsonl：541 条评测题，保留原始约束元数据。
- data/instruction/pilot_batch_plan.jsonl：64 步，每步 4 条，共 256 条无放回训练样本。
- 数据池测试 4 项通过；原批处理梯度一致性和双模板测试通过。
- 双领域新教师 GPU 端到端两步验证已通过，两位教师不变、学生适配器变化 L2=0.197147。
