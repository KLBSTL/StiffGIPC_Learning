"""Build the approved comparison PDFs from verified, existing experiment artifacts.

No simulation or source mutation is performed. Input hashes, display rows, and
PDF text/page validation are recorded for an independent delivery audit.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Flowable, Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "reports/local_combination_six_20261011_tables"
OLD_VIS = ROOT / "reports/local_combination_six_20261011_figures_depth"
NEW = ROOT / "reports/local_recording105_dt0p04_20261011"
PILOT = ROOT / "reports/local_hang_dt_pilot_20261011"
OUT = ROOT / "output/pdf/stiffgipc_comparison_20261011"
SCENES = ["Sphere", "Bunny", "Table", "Hang", "Mixed", "Boxes"]
METHODS = ["original", "pure", "ours"]
LABELS = {"original": "MOV", "pure": "TOI", "ours": "GATE"}
FULL_LABELS = {
    "original": "Movement only",
    "pure": "Cumulative TOI .01",
    "ours": "Compensated + residual gate",
}
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 32
WIDTH = PAGE_W - 2 * MARGIN
INPUTS: dict[str, dict] = {}
DISPLAY: dict[str, list] = {}
PAGE_CONTENTS: dict[str, list[str]] = {}


def register_fonts():
    base = Path("C:/Windows/Fonts")
    for name, path in [("Song", "simsun.ttc"), ("Hei", "simhei.ttf"),
                       ("Times", "times.ttf"), ("TimesBold", "timesbd.ttf"),
                       ("TimesItalic", "timesi.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(base / path)))
    pdfmetrics.registerFontFamily("Times", normal="Times", bold="TimesBold",
                                  italic="TimesItalic", boldItalic="TimesBold")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source(path: Path) -> Path:
    assert path.is_file(), path
    key = path.relative_to(ROOT).as_posix()
    if key not in INPUTS:
        INPUTS[key] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return path


def rows(path: Path) -> list[dict]:
    with source(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value, precision=3):
    v = float(value)
    assert math.isfinite(v)
    return f"{v:.{precision}f}"


def integer(value):
    v = float(value)
    assert v.is_integer(), value
    return str(int(v))


def para(text, size=10.3, font="Song", align=TA_LEFT, leading=None):
    return Paragraph(text, ParagraphStyle("p", fontName=font, fontSize=size,
        leading=leading or size * 1.45, alignment=align, wordWrap="CJK",
        textColor=colors.HexColor("#202020")))


def text(c, text, top, size=10.3, width=WIDTH, x=MARGIN, font="Song"):
    p = para(text, size=size, font=font)
    _, height = p.wrap(width, PAGE_H)
    assert top - height > 32, (text, top, height)
    p.drawOn(c, x, top - height)
    return top - height


def heading(c, title, subtitle, section, number, total):
    bookmark=f"{section}_{number}"
    c.bookmarkPage(bookmark)
    c.addOutlineEntry(title,bookmark,level=0,closed=False)
    c.setFillColor(colors.HexColor("#23384c"))
    c.setFont("Hei", 17)
    c.drawString(MARGIN, PAGE_H - 40, title)
    c.setFillColor(colors.black)
    text(c, subtitle, PAGE_H - 52, 9.2)
    c.setStrokeColor(colors.HexColor("#23384c"))
    c.setLineWidth(.7)
    c.line(MARGIN, PAGE_H - 80, PAGE_W - MARGIN, PAGE_H - 80)
    c.setFont("Song", 8)
    c.setFillColor(colors.HexColor("#606060"))
    c.drawString(MARGIN, 18, f"StiffGIPC | 本地实测 | 2026-10-11 | {section}")
    c.setFont("Times", 9)
    c.drawRightString(PAGE_W - MARGIN, 18, f"{number} / {total}")
    c.setFillColor(colors.black)


class BarLetter(Flowable):
    def __init__(self, letter):
        super().__init__()
        self.letter = letter
        self.width, self.height = 22, 15

    def draw(self):
        self.canv.setFont("TimesItalic", 12)
        self.canv.drawCentredString(11, 1, self.letter)
        self.canv.setLineWidth(.5)
        self.canv.line(6.7, 13.5, 15.3, 13.5)


def table(c, headers, values, top, weights, size=10.5, row_height=19,
          groups=3, row_header=25, minimum_bottom=63, key=None):
    if key:
        DISPLAY[key] = values
    col_widths = [WIDTH * w / sum(weights) for w in weights]
    body = []
    body.append([h if isinstance(h, Flowable) else para(h, size+0.5, "Times", TA_CENTER)
                 for h in headers])
    for r in values:
        body.append([para(html.escape(str(cell)), size, "Times",
                         TA_LEFT if j < 2 else TA_CENTER, leading=size*1.1)
                     for j, cell in enumerate(r)])
    t = Table(body, colWidths=col_widths, rowHeights=[row_header] + [row_height]*len(values))
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
    ]
    if groups:
        for k in range(groups, len(values), groups):
            style.append(("LINEBELOW", (0, k), (-1, k), .5, colors.black))
    t.setStyle(TableStyle(style))
    _, height = t.wrap(WIDTH, PAGE_H)
    assert top-height >= minimum_bottom, (key, top-height)
    t.drawOn(c, MARGIN, top-height)
    return top-height


def image_page(c, path: Path, title, subtitle, note, section, number, total):
    heading(c, title, subtitle, section, number, total)
    path = source(path)
    with Image.open(path) as im:
        w, h = im.size
    available_h = PAGE_H - 145
    scale = min(WIDTH/w, available_h/h)
    sw, sh = w*scale, h*scale
    c.drawImage(str(path), (PAGE_W-sw)/2, 69+(available_h-sh)/2,
                width=sw, height=sh, preserveAspectRatio=True, mask="auto")
    text(c, note, 60, 8.6)
    c.showPage()


def write_csv(path, headers, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def load_data():
    old_frozen = rows(OLD / "phase_table_frozen.csv")
    old_current = rows(OLD / "phase_table_current.csv")
    old_index = {(r["scene"],r["method"]):r for r in old_frozen+old_current}
    for scene in SCENES:
        a=next(r for r in old_frozen if r["scene"]==scene and r["method"]=="stiff")
        b=next(r for r in old_current if r["scene"]==scene and r["method"]=="stiff")
        assert a==b, scene
    new=rows(NEW / "phase_table_first100.csv")
    assert len(new)==18
    nindex={(r["scene"],r["method"]):r for r in new}
    cp=rows(NEW / "checkpoint_frame100_table.csv")
    trajectory=rows(NEW / "trajectory_frame100_table.csv")
    assert len(cp)==len(trajectory)==18
    strict=[]
    dispersion=rows(NEW / "reference_repeat_dispersion.csv")
    for scene in SCENES:
        selected=[r for r in dispersion if r["scene"]==scene and r["frame"]=="100"
                  and r["partition"]=="all"]
        assert len(selected)==3
        strict.append([scene, fmt(max(float(r["position_rms_mm"]) for r in selected),6),
                       fmt(max(float(r["position_max_mm"]) for r in selected),6)])
    verify=json.loads(source(NEW / "verification.json").read_text(encoding="utf-8"))
    assert verify["passed"] and verify["actual_frames"]==7560
    old_verify=json.loads(source(ROOT / "reports/local_combination_six_20261011_verification/verification.json").read_text(encoding="utf-8"))
    assert old_verify["passed"] and old_verify["formal_frames"]==9000
    for provenance in [NEW/"summary.json",NEW/"FINDINGS.md",PILOT/"summary.json",
                       ROOT/"reports/LOCAL_COMBINATION_SIX_20261011_REPORT.md"]:
        source(provenance)
    return old_index, nindex, cp, trajectory, strict


def phase_old(index):
    result=[]
    for s in SCENES:
        for m,label in [("stiff","Base"),("old","Fastest"),("ours","Current (legacy)")]:
            r=index[s,m]
            result.append([s if m=="stiff" else "",label] +
                [fmt(r[k]) for k in ["Hess","LSolver","CCD","LineS","Misc","TimeTot"]]+
                [fmt(r["P_bar"],2),fmt(r["N_bar"],2),integer(r["P_tot"]),fmt(r["S"])])
    return result


def phase_new(index):
    result=[]
    for s in SCENES:
        for m in METHODS:
            r=index[s,m]
            result.append([s if m=="original" else "",LABELS[m]]+
                [fmt(r[k]) for k in ["assembly","pcg","ccd","line_search","Misc","TimeTot"]]+
                [fmt(r["P_bar"],2),fmt(r["N_bar"],2),integer(r["P"]),fmt(r["S_movement_over_rule"])])
    return result


def checkpoint_rows(data, full_scene=True):
    result=[]
    for s in SCENES:
        for m in METHODS:
            r=next(x for x in data if x["scene"]==s and x["method"]==m)
            result.append([s if m=="original" else "", FULL_LABELS[m] if full_scene else LABELS[m],
                           integer(r["N"]),fmt(r["P"],1),fmt(r["t"],6),
                           fmt(r["e_RMS_mm"],6),fmt(r["e_max_mm"],6)])
    return result


def make_main(old, new, cp, trajectory, strict):
    filename="StiffGIPC_Comparison_Main.pdf"
    c=canvas.Canvas(str(OUT/filename),pagesize=(PAGE_W,PAGE_H),pageCompression=1)
    c.setTitle("StiffGIPC 实测对照 - 主报告")
    c.setAuthor("StiffGIPC local benchmark")
    titles=[]

    title="实验配置与时间步选择";titles.append(title)
    heading(c,title,"六场景：Sphere / Bunny / Table / Hang / Mixed / Boxes；Stack按用户决定暂缓。","主报告",1,8)
    y=PAGE_H-94
    y=text(c,"本报告包含两组独立对照：dt=0.01 的冻结 Base / 最快版 / 当前执行组合，以及 dt=0.04 当前源码记录副本的三种停止规则。各表只在其组内计算加速比，记录副本不等同于冻结最快版程序。",y,11)-12
    y=text(c,"版本对照：100帧 × 3轮；三规则记录：105帧 × 3轮，性能汇总使用前100帧、状态表使用物理第100帧。候选 rho=1e-4，原材料、FullCCD、MAS legacy；本地 RTX 3070 Laptop / WDDM，电源为实测的野兽模式。",y,10.4)-17
    y=text(c,"Hang 时间步先导：冻结 Base 与冻结最快版，各100帧、各配置1轮，保持原记录方式。",y,11)-8
    pilot=rows(PILOT/"speedups.csv")
    vals=[[fmt(r["dt"],3),fmt(r["physical_seconds"],1),fmt(r["base_seconds"],6),
           fmt(r["fastest_seconds"],6),fmt(r["speedup"],3),
           f'{integer(r["base_N"])} / {integer(r["fastest_N"])}',
           f'{integer(r["base_P"])} / {integer(r["fastest_P"])}'] for r in pilot]
    y=table(c,["dt (s)","Sim. time (s)","Base t (s)","Fastest t (s)","S","N: B / F","Ptot: B / F"],
            vals,y,[60,90,110,110,60,115,130],size=11,row_height=28,groups=0,key="pilot")-12
    y=text(c,"dt=0.04 在本场景的观测求解吞吐最好，加速比2.174×，因此用于后续记录。三个100帧对应1、4、0.4秒标称模拟时长；这个选择不构成积分精度最优或六场景统一2×的认证。",y,10.4)-13
    y=text(c,"MOV = Movement only；TOI = Cumulative TOI .01；GATE = Compensated + residual gate。三规则使用同一当前执行组合与 conditional Graph K1；门控阈值0.03、累计阈值0.01、最少6次有效更新。较严参考使用 MOV、rho=1e-6，外层阈值仍为0.01。",y,10.4)-12
    text(c,"N为实际求解方向数（包含最终 Movement 判断方向），P为PCG迭代总数；平均PCG迭代数=Ptot/Ntot，每帧平均方向数=Ntot/100。各数值从已有原生记录派生，本次只重新排版。",y,9.6)
    c.showPage()

    headers=["Scene","Method","Hess","LSolver","CCD","LineS","Misc","TimeTot",BarLetter("P"),BarLetter("N"),"Ptot","S"]
    weights=[65,115,56,60,53,53,53,68,53,53,66,48]
    title="执行优化对照：Base / 最快版 / 当前组合";titles.append(title)
    heading(c,title,"dt=0.01；rho=1e-4；100帧 × 3轮；阶段时间单位s。","主报告",2,8)
    y=table(c,headers,phase_old(old),PAGE_H-94,weights,size=10.1,row_height=18.5,key="old_phases")-10
    y=text(c,"Fastest为此前2.084×结果对应的冻结执行组合；Current为当前同配置执行组合，均使用legacy累计规则，此处未启用残差门控。冻结Base本身也有legacy累计退出，不能把整组收益归为新增TOI。",y,9.4)-7
    y=text(c,"时间、平均迭代数、平均方向数、Ptot各取运行级三轮中位数；S取三轮配对时间比的中位数，因此不一定等于表内两个中位时间的商。六场景逐轮时间和的配对倍率：Fastest 1.347435×，Current 1.336262×。",y,9.4)-7
    text(c,"Hess为原记录的组装阶段，LSolver含PCG准备、同步和Graph捕获；原记录未完整归入Hess的终止组装可能落在Misc中。各列独立取中位数，不要求列和精确等于TimeTot。",y,9.2)
    c.showPage()

    title="执行优化对照：第100帧实际网格";titles.append(title)
    image_page(c,OLD_VIS/"all_scenes_f0100_r1.png",title,
        "dt=0.01；第1轮；三行依次为Base、冻结最快版、当前同配置组合。",
        "实际网格，无几何平滑，变形比例1。图中Frozen combination对应Fastest，Current combination对应当前同配置组合；同场景各方法共用相机。视觉接近程度结合数值差异解释。","主报告",3,8)

    title="停止规则对照：前100帧阶段性能";titles.append(title)
    heading(c,title,"dt=0.04；rho=1e-4；每条轨迹105帧 × 3轮，本表只统计前100帧；阶段时间单位s。","主报告",4,8)
    y=table(c,headers,phase_new(new),PAGE_H-94,weights,size=10.1,row_height=18.5,key="new_phases")-9
    y=text(c,"MOV / TOI / GATE 的完整名称见第1页。S=同场景MOV时间中位数 / 当前规则时间中位数；这里的基线是当前程序MOV，不能替代第2页的冻结Base。",y,9.4)-7
    y=text(c,"六场景时间中位数之和：MOV 315.408955s，TOI 131.940016s，GATE 314.475985s；对应倍率2.390548×与1.002967×。纯TOI的时间优势尚未通过统一长期精度约束。",y,9.4)-7
    text(c,"Hess包含终止组装，terminal_assembly是其子集；Misc包含状态更新及求解区间内未分类时间。残差观察已包含在相关区间内，不能再次加到总时间。分项各取中位数，列和不要求精确相等。",y,9.2)
    c.showPage()

    title="停止规则对照：独立轨迹第100帧实际网格";titles.append(title)
    image_page(c,NEW/"visuals/all_scenes_frame100_r1.png",title,
        "dt=0.04；第1轮；行：Movement only / Cumulative TOI .01 / Compensated + residual gate。",
        "各规则从同一初态独立运行，前99步的轨迹已不同；本页对应独立轨迹表，不能当作共同检查点的误差图。统一相机，实际网格、变形比例1。","主报告",5,8)

    title="共同检查点：六场景第100步";titles.append(title)
    heading(c,title,"同一输入：MOV第1轮第99步后的状态；三规则各冷恢复第100步，3次候选与3次较严参考。","主报告",6,8)
    cph=["State (step 100)","Outer rule","N","P","t (s)","RMS diff. (mm)","Max diff. (mm)"]
    y=table(c,cph,checkpoint_rows(cp),PAGE_H-94,[105,270,40,60,95,115,115],size=10.3,row_height=18.5,key="checkpoint")-10
    y=text(c,"N、P、t取候选3次中位数；位置差异取候选3次与参考3次之间9个配对的最差值。参考为MOV、rho=1e-6，非精确解。RMS与最大点差异可能来自不同配对。",y,9.5)-7
    y=text(c,"计时包含冷恢复后的PCG/预条件器/Graph准备和首次执行，属于共同状态单步计时，不是纯稳态Graph重放。54次候选中51次按Movement退出，仅Boxes的3次纯TOI按累计规则退出。",y,9.5)-7
    text(c,"三个规则在各场景的N中位数均相同：1、2、2、1、2、7。晚期第100步并未显著减少PCG工作；完整轨迹中更早的接触阶段见附录逐帧曲线。",y,9.5)
    c.showPage()

    title="独立轨迹第100帧：差异与参考重复性";titles.append(title)
    heading(c,title,"本页前99步各自演化；与第6页共同输入检查点分开解释。t为完整轨迹中该帧的实际求解时间。","主报告",7,8)
    y=table(c,["Scene","Rule","N","P","t (s)","RMS diff. (mm)","Max diff. (mm)"],
        checkpoint_rows(trajectory,False),PAGE_H-92,[85,75,40,65,100,160,160],size=9.5,row_height=14, row_header=22,key="trajectory")-8
    y=text(c,"候选3轮与MOV/rho=1e-6参考3轮的最差9配对位置差异；不是相对精确解的绝对误差。下表给出参考自身3轮两两配对在第100帧的最差差异。",y,9.1)-5
    y=table(c,["Scene","Reference RMS diff. (mm)","Reference max diff. (mm)"],strict,y,
        [90,210,210],size=9.5,row_height=13,row_header=21,groups=0,key="reference_noise")-7
    text(c,"Table、Hang参考重复性较好；Sphere、Bunny、Mixed、Boxes参考自身存在明显长期分歧。尤其Boxes的较大差异应结合重复性解释，不能全部归因TOI，也不能据此认证物理正确性。",y,9.1)
    c.showPage()

    title="PCG工作量与阶段成本：变慢发生在哪里";titles.append(title)
    heading(c,title,"dt=0.04；前100帧；3轮运行级中位数。LSolver/Ptot含准备、归约、同步与Graph执行。","主报告",8,8)
    diagnostics=[]
    for s in SCENES:
        for m in METHODS:
            r=new[s,m]
            diagnostics.append([s if m=="original" else "",LABELS[m],integer(r["N"]),integer(r["P"]),
                fmt(r["P_bar"],2),fmt(r["pcg"],3),fmt(r["pcg_stage_us_per_iteration"],2),
                fmt(r["TimeTot"],3),fmt(r["S_movement_over_rule"],3)])
    y=table(c,["Scene","Rule","Ntot","Ptot",BarLetter("P"),"LSolver (s)","LS/P (us)","TimeTot (s)","S"],
        diagnostics,PAGE_H-94,[70,55,70,85,65,105,105,110,60],size=10.3,row_height=18,key="pcg_diagnostics")-10
    y=text(c,"Boxes：MOV→GATE，Ntot 3806→4611（+21.15%），Ptot 161054→185602（+15.24%），总时间185.180→216.189s（+16.75%）。PCG阶段均摊成本反而202.25→181.73us/迭代；本组证据主要指向更多工作量及相关组装、CCD和线搜索。",y,9.4)-7
    y=text(c,"Hang：GATE没有降低方向数，时间2.835→3.032s。源码在正常组装后先观察残差，再选择Movement退出，因此Movement退出仍可能承担观察成本；这一时间已经包含在求解区间内。",y,9.4)-7
    text(c,"记录开关54配对中32个选定工作计数一致；这些配对时间比中位数1.002007（观测+0.20%），范围0.903377-1.122664。当前重复数和桌面波动不足以精确分离纯记录开销。详表与逐帧曲线见附录。",y,9.2)
    c.showPage()
    c.save()
    PAGE_CONTENTS[filename]=titles


def make_appendix():
    filename="StiffGIPC_Comparison_Appendix.pdf"
    # Two data pages + six plots + 18 timelines + four repeated overview pages
    # + overhead + definitions = 32 pages.
    total=32
    c=canvas.Canvas(str(OUT/filename),pagesize=(PAGE_W,PAGE_H),pageCompression=1)
    c.setTitle("StiffGIPC 实测对照 - 详细附录")
    titles=[]
    page=1
    title="Graph消融：保留其余执行优化";titles.append(title)
    heading(c,title,"dt=0.01；100帧 × 3轮；Host/Graph在同一组合内只改变执行方式。这里Host不是冻结Stiff Base。","附录",page,total)
    y=PAGE_H-98
    for file,label in [("graph_table_old.csv","冻结最快版组合"),("graph_table_current.csv","当前同配置组合")]:
        y=text(c,label,y,11)-6
        data=rows(OLD/file)
        vals=[[r["scene"],fmt(r["Base_ms"],3),fmt(r["Graph_ms"],3),fmt(r["T_B_over_T_G"],3),
               fmt(r["Base_N"],2),fmt(r["Graph_N"],2),fmt(r["Base_P"],2),fmt(r["Graph_P"],2)] for r in data]
        y=table(c,["Scene","Host ms/step","Graph ms/step","T_H/T_G","N: Host","N: Graph","P: Host","P: Graph"],
                vals,y,[65,110,110,80,80,80,80,80],size=10.2,row_height=20,groups=0,key=file)-12
    text(c,"N、P列分别表示每帧平均方向数与每次求解平均PCG迭代数。倍率为三轮配对比值的中位数，不由表内两个中位时间直接相除。Graph收益依赖场景，不能把完整组合倍率作为纯Graph倍率。",y,9.2)
    c.showPage();page+=1

    title="时间步先导：Hang第100帧";titles.append(title)
    image_page(c,PILOT/"frame100.png",title,"冻结Base / 冻结最快版；dt=0.01、0.04、0.004；各100帧、1轮。",
        "同一dt内两版本对应同一标称模拟时刻；跨dt的第100帧分别对应1、4、0.4秒，不是相同物理时刻。视觉图不作为跨dt时间积分误差认证。","附录",page,total);page+=1

    for s in SCENES:
        title=f"{s}：逐帧性能与轨迹差异";titles.append(title)
        image_page(c,NEW/f"plots/{s.lower()}_performance.png",title,
            "dt=0.04；105帧 × 3轮；实线/区间为三轮中位数/范围，虚线标记物理第100帧。",
            "曲线包含求解、组装、PCG阶段时间、方向数、PCG总迭代数及最差参考位置差异。黑色点线为参考自身重复差异；需同时解释长期分歧。","附录",page,total);page+=1
    for repeat in [1,2,3]:
        for s in SCENES:
            title=f"{s}：关键帧时序，第{repeat}轮";titles.append(title)
            image_page(c,NEW/f"visuals/{s.lower()}_timeline_r{repeat}.png",title,
                "dt=0.04；三行停止规则，六列物理帧0 / 25 / 50 / 75 / 100 / 105。",
                "实际独立轨迹网格；同场景、同轮三规则/六关键帧共用相机。变形比例1，无平滑，网格坐标和拓扑已有无损导出验证。","附录",page,total);page+=1
    for repeat in [2,3]:
        title=f"执行版本对照：第100帧，第{repeat}轮";titles.append(title)
        image_page(c,OLD_VIS/f"all_scenes_f0100_r{repeat}.png",title,"dt=0.01；行：冻结Base / 冻结最快版 / 当前同配置组合。",
            "与主报告第3页对应的另一轮实际结果；同场景共用视角和范围。仅作为版本对照，不与dt=0.04的三规则结果混算。","附录",page,total);page+=1
        title=f"停止规则对照：第100帧，第{repeat}轮";titles.append(title)
        image_page(c,NEW/f"visuals/all_scenes_frame100_r{repeat}.png",title,"dt=0.04；行：Movement only / Cumulative TOI .01 / Compensated + residual gate。",
            "与主报告第5页对应的另一轮实际独立轨迹结果；共同检查点另见主报告第6页。视觉图与统计误差使用相同物理帧编号。","附录",page,total);page+=1

    title="记录开销：同一冷检查点的开关配对";titles.append(title)
    heading(c,title,"dt=0.04；每场景、每规则3个ON/OFF配对；完整开关明细保留在原始CSV中。","附录",page,total)
    overhead=rows(NEW/"recording_overhead_summary.csv")
    vals=[]
    for s in SCENES:
        for m in METHODS:
            r=next(x for x in overhead if x["scene"]==s and x["method"]==m)
            vals.append([s if m=="original" else "",LABELS[m],r["pairs"],r["same_selected_counter_pairs"],
                fmt(r["median_solver_ratio_all_pairs"],4),
                fmt(r["median_solver_ratio_selected"],4) if r["median_solver_ratio_selected"] else "--",
                fmt(r["minimum_solver_ratio_selected"],4) if r["minimum_solver_ratio_selected"] else "--",
                fmt(r["maximum_solver_ratio_selected"],4) if r["maximum_solver_ratio_selected"] else "--"])
    y=table(c,["Scene","Rule","Pairs","Matched","ON/OFF: all","ON/OFF: matched","Min: matched","Max: matched"],
        vals,PAGE_H-94,[70,55,55,70,120,145,110,110],size=10.3,row_height=19,key="overhead")-10
    y=text(c,"Matched要求选定计数一致：N/P、Newton记录、能量求值、回溯/相交、Graph命中及执行工作等。无匹配样本的项记为--，不能用全部配对的耗时变化估计纯记录成本。",y,9.5)-10
    y=text(c,"全部54配对中32个匹配；匹配时间比中位数1.002007，范围0.903377-1.122664。观测中心约+0.20%，波动不足以认证精确开销。单步冷恢复配对也不能推断105帧批量写盘成本。",y,9.5)-10
    text(c,"初始化、位置与实际速度导出、检查点hook、最终JSON写盘在run_costs.csv中分别记录实际host时间，不均摊到solver_ms。诊断CostScope为嵌套包含区间，父子不能求和，亦不混入正式性能统计。",y,9.4)
    c.showPage();page+=1

    title="数据定义、可复核性与交付索引";titles.append(title)
    heading(c,title,"本次只重排已有数据与实际网格图，未运行GPU仿真、未改动关键求解代码。","附录",page,total)
    y=PAGE_H-95
    notes=[
        "1. 时间：TimeTot/solver_ms为同步host求解窗口，包含Graph准备、捕获、执行和同步；排除程序初始化与轨迹导出。监控采样得到的process_wall_seconds是包络时间，不是精确进程耗时。",
        "2. 阶段：最新Hess包含终止组装；terminal_assembly是子集。残差观察host时间与求解/组装区间重叠，不能额外相加。LSolver/P是整个PCG阶段均摊成本，不能解释为单个迭代核的时间。",
        "3. 计数：N为实际PCG方向数，P为总PCG迭代数；100帧表平均PCG迭代数=Ptot/Ntot，每帧平均方向数=Ntot/100。逐Newton的rho_ratio是预处理内积比，不是真实L2相对残差或位置误差。",
        "4. 差异：共同检查点和独立轨迹分别记录。位置差异取3×3候选/较严参考配对最差值，参考重复性取参考3轮的3个两两配对最差值；较严参考仍使用Movement外层阈值0.01。",
        "5. 图像：采用已核验的OpenGL 3.3深度渲染、统一相机及实际网格，变形比例1；原始OBJ保留double坐标，光栅上传float32。未平滑或人为放大位移，参考差异不是穿透验证。",
        "6. 现有核验：dt=0.01的90次正式运行/9000帧、306个无损OBJ；dt=0.04的7560条实际帧记录、72条实际第100帧、126次检查点输入、324个无损OBJ。Stack不在本次完成场景中。",
        "7. 文件：主报告8页，附录32页，checkpoints/内6份单场景共同检查点PDF，data/内展示源CSV与参考重复性汇总。manifest.json记录输入/输出SHA256及显示行，verification.json记录本次PDF页数、文本、字体和边界检查。",
        "8. 原始记录目录：reports/local_combination_six_20261011_tables/；reports/local_hang_dt_pilot_20261011/；reports/local_recording105_dt0p04_20261011/。每帧/每Newton、GPU采样、轨迹差异与嵌套成本均保留在原目录。",
        "9. 复现排版：使用tools/local/build_comparison_pdf_20261011.py与工作区bundled Python。程序严格从CSV读取展示值；数据、图像及既有核验文件均写入输入哈希清单。"
    ]
    for note in notes:
        y=text(c,note,y,10.1)-12
    c.showPage();page+=1
    assert page-1==total, page
    c.save()
    PAGE_CONTENTS[filename]=titles


def make_checkpoints(cp):
    for s in SCENES:
        rel=f"checkpoints/Checkpoint100_{s}.pdf"
        c=canvas.Canvas(str(OUT/rel),pagesize=(PAGE_W,PAGE_H),pageCompression=1)
        c.setTitle(f"{s} - 第100步共同检查点")
        title=f"{s}：第100步共同检查点"
        heading(c,title,"dt=0.04；候选rho=1e-4；3次候选与3次较严参考；相同初始状态、冷恢复单步。","独立检查点表",1,1)
        # Keep all three rows despite the intentionally blank grouped scene cells.
        idx=SCENES.index(s)*3
        vals=checkpoint_rows(cp)[idx:idx+3]
        y=table(c,["State","Outer rule","N","P","t (s)","RMS diff. (mm)","Max diff. (mm)"],vals,
            PAGE_H-140,[80,290,40,55,100,115,115],size=12,row_height=38,row_header=32,
            groups=0,key=f"checkpoint_{s}")-28
        y=text(c,"输入：Movement only第1轮第99步后的同一状态，各规则只重新计算第100步。N、P、t取候选3次中位数；位置差异取候选3次与参考3次之间9配对的最差值。",y,11)-18
        y=text(c,"较严参考为Movement only、rho=1e-6，外层movement阈值仍为0.01，非精确解。计时包含冷恢复后的PCG/预条件器/Graph准备及首次执行，不是纯稳态Graph重放时间。",y,11)-18
        text(c,"本表与从第0步独立演化的轨迹第100帧表分别解释。全部六场景汇总、完整轨迹结果及逐帧曲线见主报告与附录。",y,11)
        c.showPage();c.save()
        PAGE_CONTENTS[rel]=[title]


def validate_pdfs():
    checks={}
    for rel,titles in PAGE_CONTENTS.items():
        reader=PdfReader(OUT/rel)
        assert len(reader.pages)==len(titles),rel
        texts=[]
        for page,title in zip(reader.pages,titles):
            assert abs(float(page.mediabox.width)-PAGE_W)<.02
            assert abs(float(page.mediabox.height)-PAGE_H)<.02
            extracted=page.extract_text()
            assert title in extracted, (rel,title,extracted[:100])
            assert "\ufffd" not in extracted
            assert len(extracted)>75
            fonts=page["/Resources"]["/Font"]
            for obj in fonts.values():
                font=obj.get_object()
                if font.get("/Subtype")=="/TrueType":
                    descriptor=font["/FontDescriptor"].get_object()
                    assert "/FontFile2" in descriptor, font
            texts.append({"title":title,"characters":len(extracted)})
        checks[rel]={"pages":len(reader.pages),"embedded_truetype_fonts":True,"text":texts}
    # Literal cells survive extraction, even when a PDF viewer is not available.
    text4="\n".join(p.extract_text() for p in PdfReader(OUT/"StiffGIPC_Comparison_Main.pdf").pages)
    for key in ["old_phases","new_phases","checkpoint","trajectory","pcg_diagnostics"]:
        for row in DISPLAY[key]:
            for cell in row[2:]:
                assert str(cell) in text4, (key,cell)
    return checks


def main():
    register_fonts()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"checkpoints").mkdir(exist_ok=True)
    old,new,cp,trajectory,strict=load_data()
    make_main(old,new,cp,trajectory,strict)
    make_appendix()
    make_checkpoints(cp)
    checks=validate_pdfs()
    for path in [OLD/"phase_table_frozen.csv",OLD/"phase_table_current.csv",OLD/"graph_table_old.csv",
                 OLD/"graph_table_current.csv",PILOT/"speedups.csv",NEW/"phase_table_first100.csv",
                 NEW/"checkpoint_frame100_table.csv",NEW/"trajectory_frame100_table.csv",NEW/"recording_overhead_summary.csv"]:
        source(path)
        dest=OUT/"data"/path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write_csv(OUT/"data/reference_frame100_dispersion.csv",["scene","RMS_diff_mm","max_diff_mm"],strict)
    for rel,info in INPUTS.items():
        assert sha(ROOT/rel)==info["sha256"],rel
    script_identity={"path":Path(__file__).relative_to(ROOT).as_posix(),"sha256":sha(Path(__file__))}
    outputs={p.relative_to(OUT).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)}
             for p in sorted(OUT.rglob("*.pdf"))}
    manifest={"schema":1,"simulation_rerun":False,"source_mutation":False,"scenes":SCENES,
        "excluded_scenes":["Stack"],"inputs":INPUTS,"outputs":outputs,
        "display_rows":DISPLAY,"contents":PAGE_CONTENTS,"generator":script_identity}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    verification={"passed_machine_checks":True,"page_total":sum(x["pages"] for x in checks.values()),
        "pdf_files":len(checks),"input_hashes_unchanged":True,"display_values_from_source":True,
        "page_size":"A4 landscape","checks":checks,"visual_review":"pending"}
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(OUT),"pdf_files":len(checks),"pages":verification["page_total"],
                      "machine_checks":"passed","visual_review":"pending"},ensure_ascii=False))


if __name__=="__main__":
    main()
