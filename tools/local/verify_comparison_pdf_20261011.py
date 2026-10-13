"""Independently audit numerical summaries and render every delivered PDF page."""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/stiffgipc_comparison_20261011"
NEW = ROOT / "reports/local_recording105_dt0p04_20261011"
QA = ROOT / "tmp/pdfs/comparison_20261011"
POPPLER = Path("C:/Users/86189/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a,b):
    assert abs(float(a)-float(b)) <= max(1e-8,abs(float(a))*1e-10), (a,b)


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def contact_sheet(files,path):
    tile_w,tile_h=1000,750
    sheet=Image.new("RGB",(tile_w*2,tile_h*2),"#d9d9d9")
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf",23)
    for i,image_path in enumerate(files):
        im=Image.open(image_path).convert("RGB")
        im.thumbnail((tile_w-20,tile_h-48),Image.Resampling.LANCZOS)
        x=(i%2)*tile_w+(tile_w-im.width)//2
        y=(i//2)*tile_h+40
        sheet.paste(im,(x,y))
        draw.text(((i%2)*tile_w+14,(i//2)*tile_h+6),image_path.stem,font=font,fill="#151515")
    sheet.save(path)


def main():
    QA.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    for rel,info in manifest["inputs"].items():
        assert digest(ROOT/rel)==info["sha256"],rel
    for rel,info in manifest["outputs"].items():
        assert digest(OUT/rel)==info["sha256"],rel

    # Recompute 18 phase rows from 7,560 frame records, rather than using their
    # pre-aggregated values or reproducing the PDF formatter's calculations.
    frames=rows(NEW/"per_frame.csv")
    assert len(frames)==7560
    by_run=defaultdict(list)
    for f in frames:
        if int(f["physical_frame"])<=100 and f["method"]!="strict":
            by_run[f["run"]].append(f)
    assert len(by_run)==54
    by_rule=defaultdict(list)
    for run,data in by_run.items():
        assert len(data)==100 and {int(x["physical_frame"]) for x in data}==set(range(1,101)),run
        sums={k:sum(float(x[k]) for x in data) for k in
              ["solver_ms","assembly_ms","pcg_ms","ccd_ms","line_search_ms","state_update_ms",
               "terminal_assembly_ms","N","P"]}
        sums["P_bar"]=sums["P"]/sums["N"]
        sums["N_bar"]=sums["N"]/100
        sums["pcg_stage_us_per_iteration"]=sums["pcg_ms"]*1000/sums["P"]
        sums["Misc"]=sums["solver_ms"]-sum(sums[k] for k in ["assembly_ms","pcg_ms","ccd_ms","line_search_ms"])
        by_rule[data[0]["scene"],data[0]["method"]].append(sums)
    for r in rows(NEW/"phase_table_first100.csv"):
        runs=by_rule[r["scene"],r["method"]]
        assert len(runs)==3
        for k,field in [("TimeTot","solver_ms"),("assembly","assembly_ms"),("pcg","pcg_ms"),
                        ("ccd","ccd_ms"),("line_search","line_search_ms"),("state_update","state_update_ms"),
                        ("terminal_assembly","terminal_assembly_ms"),("Misc","Misc")]:
            close(r[k],median(x[field] for x in runs)/1000)
        for k in ["N","P","P_bar","N_bar","pcg_stage_us_per_iteration"]:
            close(r[k],median(x[k] for x in runs))
        baseline=median(x["solver_ms"] for x in by_rule[r["scene"],"original"])
        close(r["S_movement_over_rule"],baseline/median(x["solver_ms"] for x in runs))

    cps=rows(NEW/"checkpoint_per_frame.csv")
    candidate=[r for r in cps if r["method"] in ["original","pure","ours"]]
    assert len(candidate)==54 and {r["physical_frame"] for r in candidate}=={"100"}
    exits=Counter(r["exit_reason"] for r in candidate)
    assert exits=={"movement":51,"cumulative_toi":3},exits
    for r in rows(NEW/"checkpoint_frame100_table.csv"):
        selected=[x for x in candidate if x["scene"]==r["scene"] and x["method"]==r["method"]]
        assert len(selected)==3
        close(r["N"],median(float(x["N"]) for x in selected))
        close(r["P"],median(float(x["P"]) for x in selected))
        close(r["t"],median(float(x["solver_ms"]) for x in selected)/1000)

    borders={}
    rendered=[]
    contacts=[]
    for rel in manifest["contents"]:
        path=OUT/rel
        name=path.stem
        dest=QA/name
        dest.mkdir(exist_ok=True)
        violations=[]
        with pdfplumber.open(path) as pdf:
            for i,page in enumerate(pdf.pages,1):
                for char in page.chars:
                    if char["text"].strip() and (char["x0"] < 9 or char["x1"] > page.width-9 or
                        char["top"] < 5 or char["bottom"] > page.height-5):
                        violations.append({"page":i,"char":char["text"],"x0":char["x0"],
                                           "x1":char["x1"],"top":char["top"],"bottom":char["bottom"]})
        assert not violations,(rel,violations[:5])
        prefix=dest/name
        subprocess.run([str(POPPLER),"-r","125","-png",str(path),str(prefix)],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        images=sorted(dest.glob(f"{name}-*.png"),key=lambda p:int(p.stem.rsplit("-",1)[1]))
        assert len(images)==len(PdfReader(path).pages)
        for i,image_path in enumerate(images,1):
            with Image.open(image_path).convert("RGB") as im:
                assert im.size[0]>1400 and im.size[1]>1000
                extrema=im.getextrema()
                assert any(a < 100 and b>245 for a,b in extrema),(rel,i,extrema)
            rendered.append(image_path.relative_to(ROOT).as_posix())
        for start in range(0,len(images),4):
            contact=QA/f"{name}_contact_{start//4+1:02d}.png"
            contact_sheet(images[start:start+4],contact)
            contacts.append(contact.relative_to(ROOT).as_posix())
        borders[rel]={"pages":len(images),"out_of_page_characters":0}
    verification=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
    verification.update({"independent_phase_rows_recomputed":18,"source_frame_records_checked":7560,
        "common_checkpoint_candidates_checked":54,"checkpoint_exit_counts":dict(exits),
        "output_hashes_match":True,"text_bounds":borders,"rendered_pages":rendered,
        "contact_sheets":contacts,"renderer":"Poppler pdftoppm 125 dpi",
        "visual_review":"pending"})
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"independent_numerical_audit":"passed","pages_rendered":len(rendered),
        "contact_sheets":len(contacts),"text_bounds":"passed","visual_review":"pending"}))


if __name__=="__main__":
    main()
