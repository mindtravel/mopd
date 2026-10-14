# 环境验收（2026-10-14）

远端：`para5090`；项目：`/ssd/lzx/mopd`；独立环境：`mopd_train`。

- Python 3.10；torch 2.7.1+cu128；`mopd_train` 使用 Transformers 5.2.0、tokenizers 0.22.2、huggingface-hub 1.33.0。
- `mopd_infer` 保持 Transformers 4.55.4 + vLLM 0.10.2；两个环境分离，避免 QAT 模型依赖升级影响 vLLM。
- CUDA wheel 12.8 原生包含 sm_120；驱动 580.105.08。
- `mopd_train/bin/pip check` 通过，实际版本见 `requirements.lock.txt`。
- 8 卡 BF16 SDPA 前向与反向通过。
- 8 卡随机微型 Qwen3 前向、有限 loss、有限梯度通过。
- 8 卡 NCCL all-reduce 通过：每卡结果 36。
- 检查结束后八卡显存占用归零。
- 环境约 6.7GiB，缓存约 3.8GiB，中断下载临时文件约 736MiB，总计约 11.3GiB。
- 数据盘复查可用约 1.1TiB（与此前 733GiB 快照不同；本任务未删除已有数据）。

Fairy2i IF checkpoint `/ssd/lzx/checkpoints/4b/qat/source/if_checkpoint-4065` 在 GPU6 完整加载、生成 32 token，并通过 `quant_scale` 反向传播：252 个 scale 张量、887,040 个参数、有限 loss、有限且非零梯度。
MOPD 自有 `fairy2i_qat_adapter.py` 包含与原始公式一致的 PyTorch scale-backward fallback；不修改共享 checkpoint 内的 Triton 源码。
尚未完成：双领域 MOPD 实际训练、统一 benchmark backend dispatcher、Fairy2i 全量 benchmark；QAT 不能由 vLLM 0.10.2 推理。

历史日志：`logs/install-torch.log`、`logs/finish-environment.log`、`logs/gpu-smoke.log`。
成功标志：`logs/environment-status.txt`。
重建：`bash scripts/setup.sh`；激活：`source scripts/env.sh`。
首次多卡初始化有较长等待；没有测量吞吐，不能据此推断训练速度。
