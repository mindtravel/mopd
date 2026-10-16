# MOPD 工作规范

- 代码只在本地 `/home/mindtravel/mopd` 修改和提交，Git 远端为 `mindtravel/mopd`，分支 `main`。服务器 `para5090:/ssd/lzx/mopd` 只是运行镜像，不在上面改代码或提交。
- 同步用 `scripts/dev/remote_sync.sh`：`diff` 预览，`sync` 同步，`status` 查看两端版本。只复制 Git 可见的文件（已跟踪 + 未忽略的新文件），不删除服务器上的任何文件，不碰 `runs/`、`data/`、`models/`、`logs/`、虚拟环境。
- 每次同步把源码版本写入服务器 `.deploy/source-version`（`<commit>-clean` 或 `<commit>-dirty<哈希>`），并记录文件校验。如果服务器上的同步文件被改过，下次同步会拒绝执行；先把改动搬回本地，确实要丢弃才用 `FORCE=1`。
- 训练和评测进程启动时读入代码；同步只影响之后启动的进程（包括评测队列新起的子进程），不改变已在运行的训练。
- 服务器上的 `.git` 停在迁移时的提交 599749e，之后不再更新，以 `.deploy/source-version` 为准。
- SSH 使用 `para5090`，读取 `~/.ssh/config`。GPU1 保留，不使用。
- 文档保持中文。
