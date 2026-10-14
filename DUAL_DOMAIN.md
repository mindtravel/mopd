# 双领域 MOPD 工程验证
## 目标与边界
共享 Qwen3-4B-Base + LoRA 学生，用 Qwen3-4B 教数学、Qwen3-4B-Instruct-2507 教通用指令。
这是两领域多教师 on-policy 蒸馏的工程验证；教师不是从同一个 SFT 检查点分别训练出的领域专家，不能称为严格复现原论文教师构建方式。
指令训练池是 Alpaca instruction/input，不使用参考答案，也不是专门的严格约束训练集。

## 配置
- GPU 0：可训练学生。
- GPU 4：数学冻结教师；与原固定 KL 验证任务共存。
- GPU 5：指令冻结教师；与原固定 KL 验证任务共存。
- GPU 6、7：各一个独立进程、一个学生生成副本。
- GPU 1：保留空闲，训练入口显式拒绝该设备。
- 每个生成进程 batch 16，每步共 32 条（数学 16 + 指令 16）。
- 反向 microbatch 2，每条先按回答 token 求平均，再按领域求平均，两个领域权重各 0.5。
- 正式训练 max_new_tokens=8192，64 步，共 2048 条（各领域 1024 条），不放回。
- 两步 smoke 仅生成上限 256 token，验证管线，不评价能力。正式运行重新从 Base 初始化，不接着 smoke 权重训练。

## 同步步骤
1. 固定数据池和随机种子，保存全部 batch_plan.json。
2. 每步将主学生 LoRA 写入项目内 rollout_runtime/adapter.pt。
3. 两个生成进程加载并逐 tensor 检查 LoRA 完全一致，记录学生版本。
4. 两进程并发生成，收齐同版本的 32 个回答。
5. 按领域选择教师，保留原始回答 token IDs，评分与反向按 microbatch 2 执行。
6. 检查每领域梯度贡献非零，合并后裁剪并更新一次。
7. 下一步重新同步；当前没有异步跨版本队列。

## 实测加速
相同 32 条提示，每条强制生成 256 token，贪心输出。数值用于吞吐对照，不是质量评测。
| 进程/卡数 | 每卡 batch | 总吞吐 token/s |
|---|---:|---:|
| 1 | 4 | 151 |
| 2 | 4 | 301 |
| 2 | 8 | 598 |
| 2 | 16 | 1130 |
线程版两卡 batch 4 只有约 183 token/s，因此采用独立进程。
LoRA 同步及发布约 0.05–0.08 秒，生成耗时另计。
短等长输出不代表实际长短混合回答仍能达到此吞吐。

## 显存验证
实际 64 步候选计划中最长学生提示 529 token。
以 529+8192=8721 个位置预置零值 KV cache，再运行一个真实解码 token：
- batch 8：峰值 18.20 GiB；
- batch 16：峰值 28.85 GiB，输出有限，无 OOM。
该测试验证 KV 解码容量；实际长回答生成、内存碎片和长提示 prefill 仍需由正式训练检查。
证据：logs/rollout-long-cache-probe.json、logs/process-rollout-benchmark.json。

## 验收与评测
- 回归测试：双领域抽样、域均值 loss 权重、LoRA 同步仅修改适配器、样本分发。
- smoke：真实双教师评分与学生梯度，结束检查教师指纹不变、无梯度、学生适配器确有变化。
- 训练：metrics.jsonl 中分领域 loss、sampled_log_ratio、截断数和梯度贡献；生成吞吐及每卡显存。
- 策略梯度 loss 不是准确率，也不是非负 KL，不能直接按越低越好解释。
- 固定小规模评测：数学 32 条 + IFEval 32 条，完全不参与训练。Google 官方 IFEval 严格判定器存于 vendor/instruction_following_eval。
- 自动流程在训练后依次评测 Base、最终学生和指令教师；Base 是原始不变权重，虽在训练后执行也不受学生训练影响。
- 数值只代表 pilot 子集，不是完整 541 题 IFEval 或完整 MATH 成绩。完整结论需扩大评测。

## 路径与运行
- 入口：scripts/train_multidomain_opd.py
- 配置：configs/dual_domain.json、configs/dual_domain_smoke.json
- 调度：src/mopd/data/multidomain.py
- 模型：src/mopd/backends/multi_teacher.py
- 多卡生成：src/mopd/rollout/processes.py、worker.py
- 自动流程：scripts/run_dual_domain.py（权重校验→smoke→64步→前后评测）
- 日志：logs/dual-domain-supervisor-status.json、dual-domain-smoke.log、dual-domain-train.log
- 正式产物：runs/20261013-dual-domain-batch32-mgpu-opd/
- 查看进度：在项目目录 source scripts/env.sh 后运行 python scripts/check_dual_domain.py。

## 已验证状态
教师 revision：cdbee75f17c01a7cc42f958dc650907174af0554，全部权重/配置文件哈希通过。
两步 smoke 完成，两位教师指纹不变且无梯度；学生 LoRA 变化 L2=0.197147。
已启动 64 步正式训练；IFEval 全部 541 条输入规则可执行，能力评测尚未产生结果。

## 正式首步实测（8192 输出上限）
完成 32 条，数学 10,435 token、指令 6,508 token，均自然停止，没有截断。
生成 119.81 秒，教师评分 2.17 秒，学生反向 7.15 秒，总计 129.47 秒。
有效生成吞吐 141.42 token/s；两 worker 分别耗时 63.78 和 119.72 秒，长尾等待明显。
实际生成 allocated 峰值分别 11.72 / 13.75 GiB；nvidia-smi 显示的进程显存还包含缓存保留量，不能与 allocated 直接混用。
已验证 8192 上限下能完成真实更新，但首步没有回答达到上限，不能据此宣称实际完整 8192 长生成已全部验证。
下一步优先比较 vLLM 等高吞吐生成后端及长短任务调度；不能将短等长基准 1130 token/s 当作真实训练吞吐。
