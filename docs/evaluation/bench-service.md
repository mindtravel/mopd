# IFBench / AutomationBench 评测服务

适用范围：para5090（219.146.211.36:2228）上的 checkpoint 评测服务，供本机和其它服务器提交模型。
代码：`scripts/bench_service.py`（服务端）、`scripts/bench_client.py`（客户端）、`docker/benchmarks/`（评测镜像）。

## 资源划分

| GPU | 用途 |
|---|---|
| 0–5 | 训练、sweep、队列、常规评测 |
| 6 | IFBench 服务（常驻租约） |
| 7 | AutomationBench 服务（常驻租约） |

划分写在 `src/mopd/paths.py` 的 `SERVICE_GPUS`。`runtime.scheduler.usable_gpus` 对所有编排器（sweep、experiment、queue、eval suite）去掉 6、7；`Scheduler.add` 拒绝显式包含 6、7 的固定 GPU 组。full256-mopd 相应改为 6 卡：学生 0–3、教师全在 4、rollout 在 5。

## 架构

- 两个评测都是 OpenAI 兼容 API 的客户端，只用 CPU，打包在镜像 `mopd-bench:latest` 里，源码和依赖锁与 `/srv/ifbench`、`/srv/automationbench` 的原生部署一致：
  - IFBench：allenai/IFBench @ `1c40f0c`；
  - AutomationBench：zapier/AutomationBench @ `4a8e106`。
- 本机 Docker 没有 GPU runtime，vLLM 跑在宿主机（`mopd_infer` 环境的 `vllm serve`），容器用 `--network host` 访问。本机 Docker 的 bridge 网络是坏的，所以只能用 host 网络。
- 服务只监听 `127.0.0.1:18090`。其它机器通过 ssh 上传 checkpoint、通过 ssh 调用 API，不在共享服务器上开放新端口。
- 每个评测一个 worker，绑定自己的 GPU，各自按队列串行处理任务。同一任务的两个评测在两张卡上同时跑。每个任务的流程：
  1. 在该卡上启动 vLLM，打开工具调用（hermes parser，Qwen3 chat template 支持）；LoRA 目录挂在 `adapter_config.json` 写的基座模型上。
  2. 在容器里跑评测。
  3. 记录分数，停掉 vLLM。
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
python3 bench_client.py submit /path/to/ckpt --name my-ckpt-smoke --ifbench-limit 12 --auto-examples 8 --wait

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

`--name` 同时是 vLLM 里的模型名和 inbox 目录名，不同 checkpoint 请用不同名字。

## 默认评测设置

| 评测 | 默认设置 | 输出指标 |
|---|---|---|
| IFBench | 300 题；temperature 0，seed 42，最多 4096 输出 token，64 并发请求 | strict / loose 下的 prompt 准确率和 instruction 准确率 |
| AutomationBench | simple 200 题；每题最多 50 轮，32 个任务并发；工具集 api | pass rate（全部断言通过）和平均部分得分，按领域拆分 |

- 同一个 checkpoint 只要设置相同就可比；改了设置要换 `--name`。
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
- `vllm-<bench>.log`、`<bench>.log`。

## 已验证

- 镜像自检：IFBench 官方样例答案 300 题 strict prompt accuracy 0.2800，与原生部署一致。验证命令：`docker run --rm --network host mopd-bench:latest selftest`。
- 用 Qwen3-4B-Instruct-2507 的端到端冒烟测试：待 GPU 6、7 释放后补上。
