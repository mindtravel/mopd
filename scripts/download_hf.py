"""Download files of a Hugging Face dataset at a pinned revision, recording what was fetched.

  python scripts/download_hf.py KodCode/KodCode-V1-SFT-4o data/raw/kodcode-4o --include 'data/train-*'

Resolves the branch to a commit, downloads every matching file from <endpoint>/datasets/<repo>/
resolve/<commit>/<path> (HF_ENDPOINT, default hf-mirror), and writes download-manifest.json with the
commit and each file's size and sha256. Existing files whose size matches are kept, so re-running
resumes an interrupted download.
"""
import argparse, fnmatch, hashlib, json, os, subprocess, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def api(endpoint, path):
    # hf-mirror rejects urllib's default User-Agent (403).
    req = urllib.request.Request(f"{endpoint}/api/datasets/{path}", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("dest")
    ap.add_argument("--include", nargs="+", default=["*"])
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    commit = api(endpoint, f"{a.repo}/revision/{a.revision}")["sha"]
    tree = api(endpoint, f"{a.repo}/tree/{commit}?recursive=true")
    files = [f for f in tree if f["type"] == "file" and any(fnmatch.fnmatch(f["path"], p) for p in a.include)]
    dest = ROOT / a.dest
    manifest = {"repo": a.repo, "revision": a.revision, "commit": commit, "endpoint": endpoint, "files": {}}
    for f in files:
        out = dest / f["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and out.stat().st_size == f["size"]):
            subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", str(out),
                            f"{endpoint}/datasets/{a.repo}/resolve/{commit}/{f['path']}"], check=True)
        if out.stat().st_size != f["size"]:
            raise SystemExit(f"size mismatch: {out}")
        h = hashlib.sha256()
        with out.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        manifest["files"][f["path"]] = {"size": f["size"], "sha256": h.hexdigest()}
        print(f["path"], f["size"], flush=True)
    (dest / "download-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
