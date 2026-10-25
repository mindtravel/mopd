# Study：一份 yaml 描述一组对比实验

适用范围：SFT 专家（全参或 LoRA）、MOPD 学生的训练和评测（`src/mopd/study/`，入口 `scripts/study.py`）。新实验一律写成 study。

旧配置的迁移对照：

| 旧配置 | study | 说明 |
|---|---|---|
| full256-v2 + 四个 lr 扫描点 + 队列 lrscan-full-sft | `full-sft-lr.yaml` | 15 个运行全部按身份复用 |
| sweep code-full-v1 | `code-full.yaml` | lr 1e-5 的两个运行用的是旧版本数据，已归档 |
| sweep code-think-v1 | `code-think.yaml` | 补上了低 lr 变体 |
| sweep code-data-v1（LoRA） | `code-data.yaml` | 11 个运行全部复用 |
| full256-mopd | `mopd-mic.yaml` | 教师引用 full-sft-lr 的专家 |

还没迁移的：full256 / scale256 里的混合 SFT 和线性合并（study 暂不支持 merge）。

## 为什么改

旧做法里，同一件事要写很多遍：

- GPU 卡号、输出路径前缀、数据路径和 sha、评测协议，分别写在 experiment spec、sweep spec、生成的配置、plan 和 queue 里。
- 做一组变体只能复制整份 spec。例如 lr 扫描复制出 5 个 spec，再加 1 个 queue。
- 「这次训练和以前某次一样，可以直接复用」要靠人工判断。

现在每件事只写一次：

| 内容 | 写在哪 |
|---|---|
| GPU 用途 | `src/mopd/paths.py` 的 `GPU_ROLES` |
| 模型、数据集、评测套件（名字 → 路径 + sha） | `configs/registry.yaml` |
| 默认训练设置 | `src/mopd/study/resolve.py` 的 `DEFAULTS` / `MOPD_DEFAULTS`；lr 用 `controls.LR_DEFAULTS` |
| 一组对比 | `configs/studies/<name>.yaml`，只写和默认值不同的部分，以及要扫描的轴 |

## 写法

```yaml
study: full-sft-lr
base: {method: sft}
axes:                                   # 网格：每种组合是一个运行
  domain: [math, instruction, code]
  lr: [5.0e-7, 1.0e-6, 2.0e-6, 5.0e-6, 1.0e-5]
name: "{domain}-lr{lr:g}"               # 运行名模板，必须包含所有轴
data: {math: math-numina-v3, instruction: instruction-smol, code: code-rstar-instr11}
evals: [unified-v3, {suite: lcbdev-v3, if: "domain == 'code'"}]
checkpoints: [64, 128, 256]
```

- **`if:`** 只能写字段名、常量、比较、`in`、`and/or/not`，其它表达式一律拒绝执行。
- **lr**：没写就用默认表里的值；写了和默认值不同的值，要同时写 `lr_reason`。lr 本身作为扫描轴时不需要写原因。
- **MOPD**：
  - `base` 里写 `method: mopd`，`teachers` 按名字引用别的 study 里的某个运行和 checkpoint（见 `configs/studies/mopd-mic.yaml`）。
  - 设备布局由 `GPU_ROLES` 推导：学生用前两张训练卡（开 offload），教师轮流放在学生卡上，其余训练卡跑 rollout。

## 复用

- **身份哈希**：每个运行算一个身份哈希，只包含改变训练结果的字段：
  - 基座权重和训练数据（都按 sha）；
  - 方法、prompt 格式、seed、步数、lr 和调度、序列长度、每个领域的样本数和权重；
  - MOPD 另外包括教师和生成参数。
- LoRA 的 rank 和 target 也计入哈希。
- **不计入哈希的字段**：路径、checkpoint 间隔、micro-batch、FSDP 设置、holdout（holdout 只做校验）。
- **自动复用**：`runs/` 下任何已完成的运行（包括旧 experiment 的），只要身份相同就直接复用，只补跑缺失的评测。
- **验证结果**：full-sft-lr 的 15 个运行与旧配置的身份全部一致；lr 1e-5 的数学和指令专家自动对应到 full256 的专家。

## 命令（para5090）

```bash
cd /ssd/lzx/mopd && source scripts/env.sh
mopd_train/bin/python scripts/study.py show configs/studies/full-sft-lr.yaml   # 每个运行：身份、复用还是训练、评测
setsid nohup mopd_train/bin/python scripts/study.py run configs/studies/full-sft-lr.yaml >> logs/study-full-sft-lr.log 2>&1 &
python scripts/study.py pin                                                    # 本地：给注册表里缺 sha 的文件补 pin
```

- 生成的配置和日志放在 `results/studies/<study>/`，不提交。每个运行目录里保存它的完整配置和来源记录。
- 新训练的输出放在 `runs/studies/<study>/<run>/`。
- 测试：`tests/test_study.py`。

## 旧数据版本的运行

recipe 重建后，用旧版本数据训出来的运行身份对不上，所以既不会被复用，也不会进报告。`scripts/dev/stale_runs.py --archive` 会把它们移到 `runs/archive/`，并在 `index.jsonl` 里记录原因。
