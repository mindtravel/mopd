#!/usr/bin/env bash
# Deploy source from this local worktree to the para5090 runtime mirror.
#
# The local worktree is the only place code is edited and committed (and pushed to
# GitHub). The server copy only runs it: it owns runs/, data/, models/, logs/ and the
# virtualenvs, which this script never touches. Exactly the files Git sees here
# (tracked + untracked-not-ignored) are copied; nothing on the server is deleted.
#
# Each sync records .deploy/source-version and a checksum manifest on the server. The
# next sync refuses to run if a previously synced file was edited on the server, so
# server-side hotfixes are never silently overwritten (FORCE=1 overrides).
#
# usage: scripts/dev/remote_sync.sh [sync|status|diff]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${MOPD_REMOTE_HOST:-para5090}"
REMOTE_ROOT="${MOPD_REMOTE_ROOT:-/ssd/lzx/mopd}"
SSH_CONFIG="${MOPD_SSH_CONFIG:-${HOME}/.ssh/config}"
ACTION="${1:-sync}"

[[ -r "$SSH_CONFIG" ]] || { echo "[ERROR] SSH config is missing: $SSH_CONFIG" >&2; exit 2; }
ssh_remote() { ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" "$@"; }

# NUL-separated list of files to deploy; tracked files deleted locally are skipped.
file_list() {
  git -C "$ROOT" ls-files -co --exclude-standard -z |
    while IFS= read -r -d '' path; do [[ -f "$ROOT/$path" ]] && printf '%s\0' "$path"; done
}

local_version() { python3 "$ROOT/scripts/dev/source_version.py" "$ROOT"; }

# Prints files that were changed on the server since the last sync; empty if none.
remote_drift() {
  ssh_remote "cd '$REMOTE_ROOT' && [ -f .deploy/manifest.sha256 ] && sha256sum --quiet -c .deploy/manifest.sha256 2>/dev/null || true" |
    sed -n 's/: FAILED.*$//p'
}

check_drift() {
  local drift
  drift="$(remote_drift)"
  [[ -z "$drift" ]] && return 0
  echo "[DRIFT] these files were changed on $REMOTE_HOST after the last sync:" >&2
  sed 's/^/  /' <<<"$drift" >&2
  if [[ "${FORCE:-0}" == 1 ]]; then
    echo "[DRIFT] FORCE=1: overwriting them" >&2
  else
    echo "[DRIFT] move the changes into the local worktree first, or rerun with FORCE=1 to discard them" >&2
    exit 3
  fi
}

do_sync() {
  [[ -d "$ROOT/.git" ]] || { echo "[ERROR] not a git worktree: $ROOT" >&2; exit 2; }
  check_drift
  local version manifest
  # Runs refuse to start on uncommitted source (mopd.runtime.provenance), so a dirty sync makes every
  # job a queue launches next fail (2026-10-24: code-think-v1 training). Commit first.
  if [[ "$(local_version)" == *dirty* && "${MOPD_SYNC_DIRTY:-0}" != 1 ]]; then
    echo "[ERROR] uncommitted changes; commit before syncing (MOPD_SYNC_DIRTY=1 to override for scratch roots)" >&2; exit 2
  fi
  version="$(local_version)-$(date -u +%Y%m%dT%H%M%SZ)"
  file_list | rsync -rlp --checksum --from0 --files-from=- -e "ssh -F $SSH_CONFIG" "$ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/"
  manifest="$(cd "$ROOT" && file_list | xargs -0 sha256sum)"
  # Files from the previous manifest that no longer exist locally stay on the server.
  local stale
  stale="$(ssh_remote "cd '$REMOTE_ROOT' && [ -f .deploy/manifest.sha256 ] && cut -c67- .deploy/manifest.sha256 || true" |
    grep -vxF -f <(printf '%s\n' "$manifest" | cut -c67-) || true)"
  ssh_remote "mkdir -p '$REMOTE_ROOT/.deploy' && cat > '$REMOTE_ROOT/.deploy/manifest.sha256'" <<<"$manifest"
  ssh_remote "printf '%s\n' '$version' > '$REMOTE_ROOT/.deploy/source-version'"
  echo "[SYNC] $REMOTE_HOST:$REMOTE_ROOT <- $version ($(printf '%s\n' "$manifest" | wc -l) files)"
  if [[ -n "$stale" ]]; then
    echo "[SYNC] removed locally but still on the server (not deleted):"
    sed 's/^/  /' <<<"$stale"
  fi
}

case "$ACTION" in
  sync) do_sync ;;
  status)
    echo "local:  $(local_version)"
    echo "remote: $(ssh_remote "cat '$REMOTE_ROOT/.deploy/source-version' 2>/dev/null || echo '<never synced>'")"
    drift="$(remote_drift)"
    [[ -z "$drift" ]] && echo "drift:  none" || { echo "drift:"; sed 's/^/  /' <<<"$drift"; }
    ;;
  diff)
    drift="$(remote_drift)"
    [[ -z "$drift" ]] || { echo "[DRIFT] changed on the server since the last sync:"; sed 's/^/  /' <<<"$drift"; }
    echo "[DIFF] files a sync would update:"
    file_list | rsync -rlp --checksum --from0 --files-from=- --dry-run --itemize-changes -e "ssh -F $SSH_CONFIG" \
      "$ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/" | grep -v '^\.' || echo "  (none)"
    ;;
  *) echo "usage: $0 [sync|status|diff]" >&2; exit 2 ;;
esac
