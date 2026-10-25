# 学习率：按（全参/LoRA，训练方法）给默认值

## 结论

学习率不是独立的超参，它的合理范围同时取决于：

- **更新哪些参数**：LoRA 只更新低秩增量（r=16，scale = alpha/r = 2）。同样的 lr，LoRA 对模型输出的改变比全参小得多，所以 LoRA 通常需要大 1–2 个数量级的 lr。
- **loss 是什么**：MOPD 的 loss 是逐 token 的反向 KL，再乘领域权重，梯度量级和 SFT 的交叉熵不同。全参 SFT 的 lr 不能直接搬到全参 MOPD。

所以默认 lr 按（student backend，training method）给出，写在 `src/mopd/training/controls.py` 的 `LR_DEFAULTS`：

| backend | 方法 | 默认 lr | 依据 | 适用范围 |
|---|---|---|---|---|
| hf_full | SFT | 2e-6 | code-full-v1，见下 | 已验证：代码数据，每步 32 条，256 步，constant 调度 |
| hf_lora | SFT | 5e-5 | scale256、code-data-v1 沿用 | 没有专门扫过 |
| hf_lora | MOPD | 5e-5 | scale256 的 MOPD 臂沿用 | 没有专门扫过 |
| hf_full | MOPD | 无默认 | 还没扫 | 必须显式给 lr，并写 `lr_reason` |

每步样本数和 lr 调度也会影响合适的 lr，但目前所有实验都固定在每步 32 条、constant 调度，所以先不进入默认表的键。哪天要改它们，需要先重新确认 lr。

## 规则（生成配置时强制检查）

- experiment（`scripts/make_experiment.py`）和 sweep（`scripts/run_sweep.py`）的配置里没写 lr 时，按表自动填入。
- 写了和默认值不同的 lr，或者这一组合没有默认值，必须在 controls 里写 `lr_reason` 说明原因，否则生成配置时直接报错。
- 同一个 experiment 里，lr 只要求在同一种训练方法的臂之间一致；SFT 臂和 MOPD 臂可以不同。

这些规则要防止的问题（2026-10-25）：「降 lr 减少遗忘」的结论当时只以单个变体覆盖的形式写在 code-full-v1 里，sweep 的公共默认值仍是 1e-5。之后的 code-think-v1、full256、full256-mopd 都沿用了 1e-5，这个结论没有被带过去。

## 依据：code-full-v1（全参 SFT，rstar-code-instr11）

评测集 lcb-dev-v3，713 题，greedy，输出上限 8192；base 为 28.1（functional 42.9 / stdin 16.0）。

| lr | ck64 | ck128 | ck256 |
|---|---|---|---|
| 1e-5 | – | 25.8 | 25.9 |
| 5e-6 | – | 29.2 | – |
| 2e-6 | 29.5 | 30.3（functional 39.2 / stdin 23.1） | 29.0 |
| 1e-6 | 29.0 | 30.3 | 29.9 |
| 5e-7 | 30.6（functional 41.4 / stdin 21.8） | 29.9 | 29.6 |

- lr 1e-5 比 base 低 2.2 分，2e-6 以下都比 base 高。
- 和 base 逐题配对比较（McNemar）：只有 lr 5e-7 ck64（p=0.036）和 WiSE-FT a0.5（p=0.012）显著好于 base。2e-6 ck128 的 p=0.099，不显著。
- 默认值选 2e-6 而不是 5e-7：两者分数接近，2e-6 在 128–256 步之间更稳；5e-7 在第 64 步之后就开始下降。

复现命令（para5090）：

```bash
cd /ssd/lzx/mopd && source scripts/env.sh
mopd_train/bin/python scripts/report.py sweep configs/sweeps/code-full-v1.json --protocol lcbdev-v3-greedy8192
```

## 尚未覆盖的情况

- 全参 SFT 的 2e-6 只在代码数据上验证过。数学、指令专家（full256-v2 的 expert-m / expert-i）训练后，要对照 full256（lr 1e-5）的结果确认一次。
- 全参 MOPD 没有默认值。full256-mopd 现在用 1e-5，并写了 `lr_reason` 说明原因；正式训练前应先扫几个值。
