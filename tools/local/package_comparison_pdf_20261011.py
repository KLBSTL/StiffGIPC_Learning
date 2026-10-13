"""Finalize the visually reviewed PDF package without changing any PDF bytes."""
import hashlib
import json
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"output/pdf/stiffgipc_comparison_20261011"
QA=ROOT/"tmp/pdfs/comparison_20261011"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    verification=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
    assert verification["passed_machine_checks"]
    assert len(verification["rendered_pages"])==46
    old=json.loads((QA/"reviewed_render_hashes.json").read_text(encoding="utf-8"))
    changed={name.replace("\\","/") for name,value in old.items() if digest(QA/name)!=value}
    assert changed=={
        "StiffGIPC_Comparison_Main/StiffGIPC_Comparison_Main-1.png",
        "StiffGIPC_Comparison_Main/StiffGIPC_Comparison_Main-2.png",
        "StiffGIPC_Comparison_Main/StiffGIPC_Comparison_Main-3.png",
        "StiffGIPC_Comparison_Appendix/StiffGIPC_Comparison_Appendix-32.png"},changed
    for path,value in manifest["inputs"].items():
        assert digest(ROOT/path)==value["sha256"],path
    for path,value in manifest["outputs"].items():
        assert digest(OUT/path)==value["sha256"],path
        reader=PdfReader(OUT/path)
        assert len(reader.outline)==len(reader.pages),path
    verification.update({"passed":True,"visual_review":"passed_all_46_pages",
        "visual_review_method":"All 40 report/appendix pages reviewed in rendered contact sheets; all six standalone tables and all four revised pages inspected at full page size. The other 42 final renders are pixel-identical to the reviewed version.",
        "final_render_hashes":{p:digest(ROOT/p) for p in verification["rendered_pages"]},
        "bookmark_pages_checked":46,
        "independent_verifier":{"path":"tools/local/verify_comparison_pdf_20261011.py",
            "sha256":digest(ROOT/"tools/local/verify_comparison_pdf_20261011.py")}})
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    files=sorted(p for p in OUT.rglob("*") if p.is_file())
    assert len(files)==21,len(files)
    manifest["package_files"]={p.relative_to(OUT).as_posix():{"bytes":p.stat().st_size,"sha256":digest(p)}
                               for p in files if p.name!="manifest.json"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    archive=OUT.parent/"StiffGIPC_Comparison_20261011.zip"
    # Exclusive creation preserves any pre-existing archive on a later rerun.
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            z.write(path,arcname=f"{OUT.name}/{path.relative_to(OUT).as_posix()}")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        for path in files:
            entry=f"{OUT.name}/{path.relative_to(OUT).as_posix()}"
            assert hashlib.sha256(z.read(entry)).hexdigest()==digest(path),entry
    print(json.dumps({"passed":True,"pdf_files":8,"pages":46,"archive_files":len(files),
        "archive":str(archive),"archive_MiB":round(archive.stat().st_size/1048576,2),
        "archive_sha256":digest(archive)},ensure_ascii=False))


if __name__=="__main__":
    main()
