# IFBench / AutomationBench 评测服务

适用范围：para5090（219.146.211.36:2228）上的 checkpoint 评测服务，供本机和其它服务器提交模型。
代码：`src/mopd/serving/`（服务端；`scripts/bench_service.py` 是启动入口）、`scripts/bench_client.py`（客户端）、`docker/benchmarks/`（评测镜像）。

## 资源划分

| GPU | 用途 |
|---|---|
| 0–5 | 训练、sweep、队列、常规评测 |
| 6、7 | 评测服务（常驻租约），两张卡对等，从同一个队列取任务 |

划分写在 `src/mopd/paths.py` 的 `SERVICE_GPUS`。`runtime.scheduler.usable_gpus` 对所有编排器（sweep、experiment、queue、eval suite）去掉 6、7；`Scheduler.add` 拒绝显式包含 6、7 的固定 GPU 组。full256-mopd 相应改为 6 卡：学生 0–3、教师全在 4、rollout 在 5。

## 架构

- 两个评测都是 OpenAI 兼容 API 的客户端，只用 CPU，打包在镜像 `mopd-bench:latest` 里，源码和依赖锁与 `/srv/ifbench`、`/srv/automationbench` 的原生部署一致：
  - IFBench：allenai/IFBench @ `1c40f0c`；
  - AutomationBench：zapier/AutomationBench @ `4a8e106`。
- 本机 Docker 没有 GPU runtime，vLLM 跑在宿主机（`mopd_infer` 环境的 `vllm serve`），容器用 `--network host` 访问。本机 Docker 的 bridge 网络是坏的，所以只能用 host 网络。
- 服务只监听 `127.0.0.1:18090`。其它机器通过 ssh 上传 checkpoint、通过 ssh 调用 API，不在共享服务器上开放新端口。
- GPU 6、7 是两个对等的 worker，从同一个队列取任务。一个任务占一张卡，它的 IFBench 和 AutomationBench 同时打到这张卡的 vLLM 上，请求在同一个 batch 里合并；两张卡可以同时评测两个 checkpoint。
- 每张卡上常驻一个 vLLM OpenAI 服务（打开工具调用：hermes parser，Qwen3 chat template 支持；对外模型名固定为 `policy`）。换 checkpoint 时按下面的规则处理：
  - **全参 checkpoint，结构、tokenizer 和 chat template 都和当前相同**：通过 worker 扩展（`src/mopd/serving/weight_swap.py`，dev 模式的 `/collective_rpc`）原地换权重，再清前缀缓存。如果 checkpoint 没覆盖全部参数，直接报错，不会悄悄沿用上一个模型的权重。
  - **LoRA adapter，基座和当前相同**：运行时加载（`/v1/load_lora_adapter`）。
  - **其它情况**：重启 vLLM。按卡复用编译缓存（`runtime/compile_cache`）。
- 换权重只在 tokenizer 和 chat template 一致时才做，因为服务端始终用启动时加载的 tokenizer；不一致时如果硬换，prompt 会悄悄用错模板。
- 任务失败后会停掉引擎，下一个任务从干净状态启动。
- 任务记录在 `runs/bench-service/jobs/<id>.json`，服务重启后未完成的部分自动重新排队。

## 其它服务器怎么用

前提：能 ssh 到 para5090，有 python3（客户端只用标准库）。

```bash
# 1. ~/.ssh/config
Host para5090
    HostName 219.146.211.36
    Port 2228
    User root

# 2. 取客户端
scp para5090:/ssd/lzx/mopd/scripts/bench_client.py .

# 3. 提交 checkpoint（自动 rsync 到 para5090 的 inbox，跳过优化器状态和 *.pt），--wait 等结果
python3 bench_client.py submit /path/to/checkpoint-0256 --name my-ckpt --wait

# 小规模试跑：IFBench 12 题 + AutomationBench simple 8 题
python3 bench_client.py submit /path/to/ckpt --name my-ckpt-smoke --ifbench-limit 12 --auto-limit 8 --wait

# AutomationBench 正式领域：all = 6 个领域共 600 题；也可以只跑一个领域
python3 bench_client.py submit /path/to/ckpt --auto-domains all
python3 bench_client.py submit /path/to/ckpt --auto-domains sales

# 只跑一个评测
python3 bench_client.py submit /path/to/ckpt --benchmarks ifbench

# checkpoint 已经在 para5090 上，不上传
python3 bench_client.py submit /ssd/lzx/mopd/runs/xxx/checkpoint-0128 --remote-path

# 查询
python3 bench_client.py status          # 所有任务
python3 bench_client.py status <任务ID>  # 单个任务和分数
```

- `--name` 是 inbox 目录名，也是 LoRA 在 vLLM 里的名字，不同 checkpoint 请用不同名字。
- 客户端只能选评测、子集大小（`--ifbench-limit`、`--auto-limit`，取前 N 题）和 AutomationBench 领域（`--auto-domains`）。评测设置本身由服务端固定，见下节。

## 评测设置（固定，`src/mopd/serving/protocol.py`）

| 评测 | 设置 | 输出指标 |
|---|---|---|
| IFBench | 300 题；temperature 0，seed 42，最多 4096 输出 token，64 并发请求（上游封装的上限） | strict / loose 下的 prompt 准确率和 instruction 准确率 |
| AutomationBench | 默认 simple 200 题；每题最多 50 轮，8 个任务并发；工具集 api | pass rate（全部断言通过）和平均部分得分，按领域拆分 |

- 所有 checkpoint 都用同一套设置，分数可以直接比较。每个任务的记录里会带上实际用的设置（`protocol` 字段）。
- vLLM 上下文长度固定为 32768。
- 以下几项按 checkpoint 自带的 chat template 决定（`protocol.settings`）：
  - **工具调用格式**：模板里有 `<function=`（Qwen3.5 的 XML 格式）就用 `qwen3_coder` parser，否则用 `hermes`（Qwen3 的 JSON 格式）。
  - **思考模型**：模板里有 `enable_thinking` 就算思考模型。vLLM 加 `--reasoning-parser qwen3`，把思考内容和最终回答分开，评测只看最终回答；IFBench 的输出上限提到 16384，否则很多回答会在思考阶段被截断。
  - **思考模型的分数不能和非思考模型直接比较**：两者输出上限不同，报告时要标明。
- 服务用独立的推理环境 `mopd_serve`（vLLM 0.31），支持 Qwen3.5 等新架构；训练用的 `mopd_infer`（vLLM 0.10.2）不变。多模态 checkpoint 只按纯文本服务，不给图像和视频预留显存。
- AutomationBench 要求模型会调用工具。没学过工具调用的 base 模型或 plain_qa 学生预计接近 0 分，这反映的是模型能力，不是服务故障。
- AutomationBench 的 simple 任务和 6 个正式领域要分开报告；公开任务的成绩和官方排行榜（私有留出集）不等价。

## 服务端运维（para5090）

```bash
cd /ssd/lzx/mopd && source scripts/env.sh
# 启动（常驻）
setsid nohup mopd_train/bin/python scripts/bench_service.py --port 18090 >> runs/bench-service/service.log 2>&1 &
# 健康检查：GPU 归属、是否在等租约、各队列长度
curl -s localhost:18090/health
# 重建镜像（自检：IFBench 官方样例答案 strict 应为 0.2800）
bash docker/benchmarks/build.sh
```

结果目录：`runs/bench-service/runs/<任务ID>/`，包含：

- `ifbench/run/`：上游原始输出和 `metrics.json`；
- `automationbench/results.json`；
- `ifbench.log`、`automationbench.log`。

vLLM 日志按卡写在 `runs/bench-service/vllm-gpu<N>.log`。

## 资源利用率记录

每个任务完成后，记录里有一个 `utilization` 字段。它统计这张卡的引擎在本任务期间做了多少工作，两个评测合并计算：

| 字段 | 含义 |
|---|---|
| `generated_tok_s` | 每秒生成的 token 数 |
| `tokens_per_step` | 引擎每一步平均处理的 token 数，相当于平均 batch 大小 |
| `mfu` | 算力利用率，用实测的 bf16 峰值做分母 |
| `mbu_weights` | 显存带宽利用率，只算读权重，是下限 |
| `prefix_hit` | 前缀缓存命中率 |
| `preemptions` | 抢占次数 |

- 数据来源：任务开始和结束时各读一次 vLLM 的 `/metrics`，相减得到本任务的部分。不做采样，不影响评测时间。
- 计算公式在 `src/mopd/profiling/`，MOPD 训练的 rollout 也用同一套。
- 峰值缓存在 `cache/profile/`。重新测量的命令（要在空闲的卡上跑）：`python -m mopd.profiling.peaks --measure --device <N>`。

## 已验证

- 原地换权重正确：同一请求在换权重前后，输出文本逐字节相同，log-prob 最大差 0.0。验证方法是 `/collective_rpc` 前后各发 3 次单条请求。
- 模型加载耗时：冷启动 75–80 s，原地换权重 3.8–4.6 s。
- vLLM 服务在 temperature 0 下也不能逐字复现：换权重之前，同一请求连发 3 次就得到 2 种输出，原因是前缀缓存命中和 batch 组成改变了数值。同一 checkpoint 重复评测，分数会有小幅波动。

- 镜像自检：IFBench 官方样例答案 300 题 strict prompt accuracy 0.2800，与原生部署一致。验证命令：`docker run --rm --network host mopd-bench:latest selftest`。
- 用 Qwen3-4B-Instruct-2507 跑通端到端：IFBench 12 题 + AutomationBench 8 题，两个评测都正常出分。
- 合并组 batch（IFBench 64 题 + AutomationBench 64 题，同一张卡）：两个评测在一张卡上 56 s 跑完；分开跑时各占一张卡，分别用 53 s 和 67 s。分数和分开跑一致，差异在服务本身的波动范围内。
