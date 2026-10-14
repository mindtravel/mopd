# MATH 双模板 OPD 初步实验

## 目的与运行位置
比较原始 Qwen3-4B-Base 和经过单教师 OPD 的学生在同一批独立数学题上的准确率。
这不是完整 MOPD 复现，也不是量化实验。所有文件位于 /ssd/lzx/mopd。

## 数据
来源：EleutherAI/hendrycks_math（Hugging Face 镜像下载）。
固定 revision、文件 SHA256、抽样策略见 data/math/manifest.json。
下载全部官方 train=7500、test=5000；训练候选从 Level 3–5 中固定抽取1024题，不含参考解答。
本轮64次更新，每次累积4条不同问题，实际使用256条训练问题。
评测固定抽取官方 test 中128题，各难度均有覆盖；排除题面包含 Asymptote 图形代码的题。
训练候选排除了与全部 test 规范化空白后完全相同的题。未做语义近重复检测。
不是完整 MATH benchmark 成绩，亦无法排除基础模型预训练污染。

## 训练与模板
学生：Qwen3-4B-Base；教师：Qwen3-4B。
学生普通 Question/Answer 前缀；教师原生 chat 非思考前缀。
两者问题正文相同；学生生成的 response token ID 直接拼到教师前缀，不解码再编码、不去除或改写正文。
教师冻结，只对同一批 response token 的条件概率评分；学生 LoRA 采用 sampled-token reverse KL 策略梯度，优势裁剪到[-5,5]。
512个新token上限；temperature=1、top_p=1、top_k=0；学习率5e-5。
每16步保存一次 adapter 及 optimizer/RNG。训练日志记录实际token数与触及长度上限的样本数。
两步旧冒烟实验不计入本轮，学生从原始 Base 重新初始化。
GPU4学生、GPU5教师；GPU6/7同时执行原始学生/教师基线评测。

## 评测
scripts/eval_math_opd.py；math-verify==0.8.0，校验等价表达式。
固定128题，greedy，最多1024新token，要求最终答案放入 boxed。
仅提取最后一个完整 boxed 答案；缺少答案算错，解析异常单独记录；因此准确率同时受格式遵循影响。
学生训练前后使用完全相同模板和解码条件。教师使用自己的非思考模板，作为参考上限，不能归因于纯权重差异。
不使用本测试集选择checkpoint；训练完成后自动评测最终checkpoint-0064并汇总差值。
128题仅作初筛，差异小需扩大评测及重复种子，不能直接宣称稳定能力提升。

## 文件
- configs/math_opd.json：固定训练参数
- src/mopd/training/dual_template.py：模板、token对齐、采样策略损失
- scripts/prepare_math.py：下载与数据划分
- scripts/train_math_opd.py：训练、checkpoint、状态日志
- scripts/eval_math_opd.py：能力评测与逐题回答
- scripts/run_math_experiment.py：并行启动及自动后评测
- runs/20261013-math-level345-opd：新训练产物
- runs/20261013-math-evaluation：三组评测、最终comparison.json
- logs/math-experiment-supervisor.log：任务PID及退出码

## 验证与待办
- [x] 回答token保真、因果位置对齐、教师梯度隔离测试
- [x] 等价分数/错误答案/无最终答案的判分检查
- [x] 真实MATH题训练一轮，学生参数变化、教师全参数哈希不变
- [x] 实际生成adapter_config.json和adapter_model.safetensors
- [x] 下载冻结训练/测试数据，启动并行实验
- [ ] 新训练64步完成，核对checkpoint与日志
- [ ] 原始学生/教师/训练后学生128题评测完成，检查逐题答案与准确率差值
