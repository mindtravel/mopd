# 单教师 LoRA OPD

## 当前实验
- 教师：Qwen/Qwen3-4B，冻结后训练权重。
- 学生：Qwen/Qwen3-4B-Base，LoRA rank 16 / alpha 32，仅训练 q/k/v/o 投影适配器。
- 官方教师模型卡声明底座为 Qwen3-4B-Base；这是底座→后训练模型的继承关系，不是论文共同 SFT 起点分叉的领域教师复现。
- 两者使用教师聊天模板，关闭 thinking；先验证词表完全相同。
- 每步由当前学生生成新轨迹（temperature=1、无 top-k/top-p 截断），冻结教师对同一轨迹评分。
- sampled reverse-KL 策略梯度：优势为 stop_gradient(log p_teacher - log p_student)，双边裁剪到 ±5。
- 仅回答 token 进入损失，正确将位置 t-1 logits 对齐 token t。
- 16 步，每步一个数学提示，最多生成 64 token；真实 4B 管线验证，不是质量评测。
- 两个固定留出轨迹记录训练前后精确分布 reverse KL；采样 log-ratio 可以为负，不当作精确 KL。

## 运行与审核
```bash
cd /ssd/lzx/mopd
source scripts/env.sh
python scripts/test_losses.py
python scripts/train_opd.py --config configs/single_teacher.json
```
输出目录已存在时拒绝覆盖；新实验需要修改配置中的 output。
`logs/opd-status.json` 是等待/运行/结束状态，真实运行证据以进程和训练日志为准。
`logs/opd-train.log` 保存训练输出；`runs/20261013-qwen3-4b-lora-opd/` 保存：
- config.json：实际配置
- metrics.jsonl：每步 loss、梯度范数、采样比率、token 数与耗时
- rollouts.jsonl：每步 prompt、回答、token 和掩码
- adapter/：可重新加载的 LoRA 权重
- summary.json：只有完成训练、教师 hash 未变、LoRA 已更新、重载输出一致后才写 PASS

模型 revision 见 scripts/download_models.py 和 models/revisions.json。
这套代码是本项目最小同步 OPD 实现，不声称是官方 MOPD/verl 实现。
