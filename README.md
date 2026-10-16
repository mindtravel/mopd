# MOPD 基础环境

固定工作目录：`/ssd/lzx/mopd`。已有工程和权重只读，不混用其依赖。

## 当前范围
先完成 8 × RTX 5090 基础环境验证，再接入 4B 单教师同步 MOPD。
`check_environment.py` 使用随机初始化的微型 Qwen3，仅验证 GPU/依赖/梯度与通信，不代表已跑通 4B 模型或 MOPD。
尚未下载真实教师、数据、4B 权重。verl、vLLM、Megatron 尚未安装，后续分别验证适配。

## 目录
- `scripts/`：环境与检查入口。
- `configs/`：实验配置；未选定教师前不写入虚假模型 ID。
- `src/mopd/data/`：领域数据和抽样。
- `src/mopd/backends/`：模型加载、前向和量化适配。
- `src/mopd/rollout/`：学生生成。
- `src/mopd/teacher/`：教师评分与路由。
- `src/mopd/training/`：损失、更新及权重同步。
- `runs/`：按实验保存配置、指标、checkpoint；建议仅保留最近两份。
- `logs/`、`cache/`、`tmp/`：全部位于本项目中，不使用系统临时目录保存产物。

## 开发与同步
代码在本地 worktree 修改、提交并推送到 GitHub，服务器 `/ssd/lzx/mopd` 只是运行镜像。用 `scripts/dev/remote_sync.sh diff|sync|status` 把源码同步到服务器，规则见 `AGENTS.md`。

## 使用
```bash
cd /ssd/lzx/mopd
source scripts/env.sh
python -m pip check
torchrun --standalone --nproc_per_node=8 scripts/check_environment.py
```
基础依赖：Python 3.10、PyTorch 2.7.1 CUDA 12.8、Transformers 4.51.3。
系统 CUDA toolkit 13.0 与 wheel 内置 CUDA 12.8 不同；基础 wheel 运算不需要编译，但后续自定义扩展必须另行核验编译工具链，不能直接假定兼容。

## OPD 实跑结果

真实 4B 单教师 + LoRA 学生已完成 16 步，结果见 `runs/20261013-qwen3-4b-lora-opd/REPORT.md`。
原先‘尚未下载权重’与‘没有算法实现’是环境搭建阶段记录，现已完成该部分；效果质量尚未验证。
