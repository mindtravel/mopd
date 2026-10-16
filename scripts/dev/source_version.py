"""Print the source version of a worktree: <HEAD>-clean, or <HEAD>-<hash of uncommitted changes>.

The hash covers tracked diffs against HEAD and the contents of untracked, non-ignored
files, i.e. exactly what scripts/dev/remote_sync.sh copies to the server.
"""
import hashlib
import subprocess
import sys
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def source_version(root):
    root = Path(root)
    head = git(root, "rev-parse", "--short=12", "HEAD").decode().strip()
    diff = git(root, "diff", "--binary", "HEAD")
    untracked = sorted(p for p in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p)
    if not diff and not untracked:
        return f"{head}-clean"
    digest = hashlib.sha256(diff)
    for raw in untracked:
        digest.update(raw + b"\0")
        digest.update((root / raw.decode("utf-8", "surrogateescape")).read_bytes())
        digest.update(b"\0")
    return f"{head}-dirty{digest.hexdigest()[:12]}"


if __name__ == "__main__":
    print(source_version(sys.argv[1] if len(sys.argv) > 1 else "."))
