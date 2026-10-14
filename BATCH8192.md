# 8192 token 批量 OPD
配置：configs/numina_8192_batch2.json。原始Base重新初始化LoRA，固定同一数据顺序；64次更新，每次4条（micro-batch=2，累积2次）。学生GPU2、教师GPU3，两个8192基线评测在GPU6/7。最终checkpoint-0064自动评测。

## 已实施
学生左padding批量生成；只保留首个EOS及之前的实际回答。评分/训练右padding并屏蔽padding、前缀，显式position_ids。教师与学生接收同一回答token。每条先平均，再按4条平均；两次micro-batch之间不更新参数，保持同步on-policy。
输出头分块及重计算减少长序列logits显存。CPU小模型验证batch与逐条评分/梯度一致。
两条各8192响应（总长8704）的完整前后向压力测试通过：学生峰值14.55 GiB、教师8.84 GiB；不含生成KV cache，不能据此断言所有真实题目显存都安全。
生成、教师、学生前后向使用CUDA同步计时，写入metrics.jsonl。

## 查看
http://localhost:8765 已切换至本轮。SSH隧道沿用。
logs/numina-8192-batch2-train.log；logs/batch8192-supervisor.log。
runs/20261013-numina-8192-batch2-opd：训练。
runs/20261013-math-evaluation-8192：同长度评测，避免与旧1024结果混比。

## 后续infra优先级
1. 按实测generation_seconds、teacher_seconds、student_seconds定位瓶颈，比较tokens/s而不是仅比较每步时间。生成长度、采样随机序列改变，不能承诺batch2恰好2倍加速。
2. HF普通batch会等最长回答结束，短回答结束后有闲置；优先用vLLM连续批处理生成，提高不同长度样本并行效率。每次更新后同步LoRA权重，确保下一轮用新学生。
3. 教师批量prefill按长度分桶并限制总token数；现有真batch已完成第一步，不需要为4B立刻引入Megatron。
4. verl可用于管理rollout、教师评分与训练worker，但需确认Qwen3、LoRA热更新、自定义双模板OPD损失兼容性；不等于替换vLLM，通常由verl调度vLLM。
5. 最后再考虑跨批次异步流水。记录学生版本、限制滞后，避免为了吞吐破坏on-policy假设。
当前不部署vLLM/verl，不改变本轮优化目标。
