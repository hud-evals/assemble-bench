"""Push an already-converted local LeRobot dataset to the Hub (skip .cache/).

    conda run -n vla python scripts/experts/util/push_lerobot.py \
        --root data/lerobot/assembly_bench_2 \
        --repo_id hud-evals/AssembleBench
"""

import argparse
import os
import subprocess
import time

# Accelerated LFS uploaders silently truncate large parquet.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "assembly_bench_2"))
parser.add_argument("--repo_id", default="hud-evals/AssembleBench")
args = parser.parse_args()

out_root = os.path.abspath(args.root)
api = HfApi()
print(f"pushing {out_root} -> https://huggingface.co/datasets/{args.repo_id}", flush=True)
api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)


def served(rel):
    """Download served bytes; return (size, sha256, parquet_tail)."""
    import hashlib
    url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/{rel}"
    subprocess.run(["curl", "-sL", url, "-o", "/tmp/served.bin"], check=False)
    try:
        data = open("/tmp/served.bin", "rb").read()
        return len(data), hashlib.sha256(data).hexdigest(), data[-4:]
    except OSError:
        return -1, "", b""


def local_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Skip HF download cache left by --append (hub rejects uploads under .cache/).
paths = sorted(
    os.path.relpath(os.path.join(dp, fn), out_root)
    for dp, _, fns in os.walk(out_root) for fn in fns
    if ".cache" not in os.path.relpath(os.path.join(dp, fn), out_root).split(os.sep)
)
print(f"{len(paths)} files to verify/upload", flush=True)

for rel in paths:
    local = os.path.join(out_root, rel)
    size = os.path.getsize(local)
    lsha = local_sha(local)
    is_parquet = rel.endswith(".parquet")
    for _ in range(8):
        hsz, hsha, tail = served(rel)
        # Size alone is not enough (info.json stayed 3719B while content changed).
        if hsz == size and hsha == lsha and (not is_parquet or tail == b"PAR1"):
            break
        api.upload_file(path_or_fileobj=local, path_in_repo=rel, repo_id=args.repo_id,
                        repo_type="dataset", commit_message=f"upload {rel}")
        time.sleep(2)
    hsz, hsha, tail = served(rel)
    assert hsz == size and hsha == lsha and (not is_parquet or tail == b"PAR1"), \
        f"{rel} mismatch (served {hsz}/{size}, sha {hsha[:8]}!={lsha[:8]}, tail={tail})"
    print(f"verified {rel} ({size} bytes)", flush=True)

main = [b.target_commit for b in api.list_repo_refs(args.repo_id, repo_type="dataset").branches
        if b.name == "main"][0]
try:
    api.delete_tag(args.repo_id, tag="v3.0", repo_type="dataset")
except Exception:
    pass
api.create_tag(args.repo_id, tag="v3.0", revision=main, repo_type="dataset")
print(f"done: https://huggingface.co/datasets/{args.repo_id} @ {main}", flush=True)
