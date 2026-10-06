"""Freeze committed baseline and hash-identified v3 donors in this task only."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
COMMIT = "bb2849a7b292099581907937860d96ecfdf42588"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    lock = ROOT / "SOURCE_LOCK.json"
    if lock.exists():
        print("Already frozen; no source overwritten.")
        return
    base = ROOT / "sources/stiff_base"
    fused = ROOT / "sources/stiff_fused"
    if base.exists() or fused.exists():
        raise RuntimeError("Partial source directory exists; inspect before resuming")
    bundle = ROOT / "bundles/base_source.tar"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(WORKSPACE / "externals/Stiff-GIPC-official"),
                    "archive", "--format=tar", "-o", str(bundle), COMMIT], check=True)
    base.mkdir(parents=True)
    with tarfile.open(bundle) as archive:
        archive.extractall(base, filter="data")
    shutil.copytree(base, fused)
    donor_records = json.loads((ROOT / "research/graph_source_provenance.json").read_text())
    donors = ROOT / "references/graph_v3"
    for row in donor_records:
        src = Path(row["local_candidate"])
        if digest(src) != row["sha256"]:
            raise RuntimeError(f"Donor changed since planning: {src}")
        dst = donors / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(dst) != row["sha256"]:
            raise RuntimeError(f"Donor copy changed: {dst}")
    shutil.copy2(WORKSPACE / "stiffGIPC/LICENSE", donors / "LICENSE")
    tracked = [{"path": str(p.relative_to(base)).replace("\\", "/"),
                "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(base.rglob("*")) if p.is_file()]
    lock.write_text(json.dumps({"baseline_commit": COMMIT,
        "baseline_repository": "https://github.com/KemengHuang/Stiff-GIPC",
        "baseline_archive_sha256": digest(bundle), "baseline_files": tracked,
        "graph_donors": donor_records}, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_files": len(tracked), "donor_files": len(donor_records),
                      "baseline_bytes": sum(r["bytes"] for r in tracked)}))

if __name__ == "__main__":
    main()
