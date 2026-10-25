# GPU 资源池

适用范围：para5090 上我们自己的所有 GPU 任务，包括训练、评测、打分、合并和基准测试。代码在 `src/mopd/pool/`，入口是 `scripts/pool.py`。

## 为什么要有资源池

以前每个 study、queue、基准脚本都各自起一个调度器，彼此只靠 GPU 租约互斥：

- 没有谁能看到全局的排队情况。例如 2026-10-25，GPU 1、3 空了半小时，因为线程数基准在等 code-think 的进程退出，而不是在等卡。
- 任务按固定卡号申请，只空出零散几张卡时，什么任务都放不进去。
- 空出来的卡不会被别的任务借用，借出去的卡也收不回来。

## 怎么工作

- **常驻 daemon**：每台机器一个，独占 `paths.SCHEDULED_GPUS`（目前是 0–3）。
- **任务只提交，不自己跑**。每个任务声明：
  - 形状：固定张数 `{"count": 4}`，或弹性 `{"min": 1, "max": 4}`，或 `0`（只用 CPU）；
  - 优先级：训练 50、合并 30、评测 20、打分 0；基准测试 70；
  - 能否被抢占；
  - 依赖；
  - 完成检查：训练看运行目录的 status 是否 complete，评测看是否所有行都已生成，其它看指定文件是否存在。
- **daemon 每 5 秒做一次匹配**（`allocate.decide`）：
  - 按优先级处理；多卡任务的卡必须一次凑齐才启动；
  - 有任务在等整组卡时，空闲的卡只能给可抢占的任务用（回填），它们随时可以被收回；
  - 如果停掉低优先级的可抢占任务就能凑够卡，就停掉它们，让它们回到队列，之后再跑。评测从已经写好的行接着生成。
- **启动方式**：任务启动时，`CUDA_VISIBLE_DEVICES` 设为分到的卡，所以每个任务看到的都是 0..n-1；命令里的 `{gpus}` 会展开成这些编号。MOPD 的布局（学生、教师、rollout）也用逻辑编号。
- **和旧工具共存**：daemon 会替任务拿 GPU 租约，旧调度器和评测服务照样会避开这些卡。别的用户正在用的卡（有显存占用、但不是我们的租约）不会被分配。
- **状态全部落盘**，在 `results/pool/` 下：`jobs/` 是每个任务的记录，`events.jsonl` 是事件日志，`state.json` 是当前状态快照。daemon 重启后会接回仍在运行的任务。

## 用法（para5090）

```bash
cd /ssd/lzx/mopd && source scripts/env.sh
setsid nohup mopd_train/bin/python scripts/pool.py serve >> logs/pool.log 2>&1 &   # 启动 daemon（每台机器一个）
mopd_train/bin/python scripts/study.py submit configs/studies/mix-v2.yaml           # 把一个 study 的所有任务交给资源池
mopd_train/bin/python scripts/pool.py status [--all]                                # 查看排队和运行情况
mopd_train/bin/python scripts/pool.py cancel <任务 id>
mopd_train/bin/python scripts/pool.py submit jobs.json                              # 提交自定义任务（JSON 列表）
```

- 重复提交同一个 id：如果任务正在排队、正在运行或已完成，什么也不发生；如果它失败了、被阻塞或被取消，就重新入队。
- 失败的任务不会自动重试。先查日志（任务记录里的 `log` 字段），再重新提交。

## 已验证

- `tests/test_pool.py`：匹配逻辑，包括整组分配、回填、抢占、优先级、弹性任务、只用 CPU 的任务。
- `tests/test_pool_daemon.py`：用真实子进程、在假的 GPU 上端到端验证：
  - 依赖和完成检查；
  - 失败的任务会阻塞依赖它的任务；
  - 被抢占的任务会重新跑；
  - 同一秒内的多次提交都能收到。
- 2026-10-25 上线：线程数基准、merge-v2 和 mix-v2 都交给资源池调度。

## 评测服务的卡（6–7）出借

- 评测服务的某个 worker 空闲超过 10 分钟，就停掉引擎、释放租约，并写出借标记 `results/pool/lend/gpu<N>`。
- 资源池把出借的卡算进空闲卡，但只分给可抢占的任务（`allocate.decide(lent=...)`）。
- 评测服务来了任务，就删掉出借标记、写召回标记 `results/pool/recall/gpu<N>`。资源池立刻抢占那张卡上的任务（评测之后会续跑）。评测服务等租约和显存都空出来，再拿回这张卡。

## 训练的抢占和续训（全参 SFT）

- **协作式停止**：资源池给每个任务传一个停止文件路径 `$MOPD_POOL_STOP`。要抢占可协作停止的任务时，就创建这个文件。
- **训练这边**：`train_sft` 每一步结束后，各个 rank 一起确认要不要停（`TrainRun.stop_requested`，所有 rank 结论一致）。要停的话：
  - 保存完整训练状态到运行目录下的 `resume/`，用 `TrainRun.save_resume`；
  - 模型和优化器通过 `torch.distributed.checkpoint` 按分片保存，另外存 lr 调度器、随机数状态和步数；
  - 然后以退出码 75 退出，资源池把它放回队列。
- **续训**：再次启动时，`TrainRun.start` 看到 `resume/` 就从保存的那一步接着训，日志截断到那一步。分片在加载时会按新的卡数重新切分，所以可以换卡数续训，比如从 4 卡换成 2 卡 + offload。
- **超时兜底**：如果超过 `stop_grace`（训练是 900 s）还没退出，资源池才发 SIGTERM。
- study 提交的 SFT 训练都标记为可抢占（`preemptible`、`graceful`）。MOPD 训练还不支持续训，所以不可抢占。
- GPU 上的验证脚本是 `tests/gpu/test_sft_resume.py`：
  - 不间断跑 6 步作为对照；
  - 另跑一次，第 3 步后要求停止，再换成 2 卡 + offload 续训；
  - 比较续训后各步的 loss。

## 还没做

- MOPD 训练的续训（异步 rollout 的状态、发布出去的权重、教师），做完之后 MOPD 也能被抢占。
- 按瓶颈动态增减 MOPD 的 rollout 引擎。
- 训练按可用卡数选择形状，比如卡紧张时自动用 2 卡 + offload。现在续训已经支持换卡数，只差选择策略。
