"""Render every cloud report page and check typography, bounds and source hashes."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/stiffgipc_autodl3_20261013'
QA=ROOT/'tmp/pdfs/autodl3_20261013'
POPPLER=Path('C:/Users/86189/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
spec=importlib.util.spec_from_file_location('pdf_qa',ROOT/'tools/local/verify_comparison_pdf_20261011.py')
helpers=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    QA.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/'PDF_manifest.json').read_text(encoding='utf-8'))
    for rel,data in manifest['inputs'].items():
        assert helpers.digest(ROOT/rel)==data['sha256'],rel
    contacts=[]
    pages=[]
    for rel,info in manifest['pdfs'].items():
        path=OUT/rel
        assert helpers.digest(path)==info['sha256']
        doc=PdfReader(path)
        assert len(doc.pages)==info['pages'] and len(doc.outline)==info['pages']
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for char in page.chars:
                    if char['text'].strip():
                        assert 9<=char['x0']<=char['x1']<=page.width-9
                        assert 5<=char['top']<=char['bottom']<=page.height-5
        dest=QA/path.stem
        dest.mkdir(exist_ok=True)
        prefix=dest/path.stem
        subprocess.run([str(POPPLER),'-r','125','-png',str(path),str(prefix)],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        images=sorted(dest.glob('*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
        assert len(images)==info['pages']
        pages.extend(p.relative_to(ROOT).as_posix() for p in images)
        for start in range(0,len(images),4):
            target=QA/f'{path.stem}_contact_{start//4+1:02d}.png'
            helpers.contact_sheet(images[start:start+4],target)
            contacts.append(target.relative_to(ROOT).as_posix())
    assert len(pages)==24
    result=dict(machine_checks_passed=True,pdf_files=5,pages=24,rendered_pages=pages,contacts=contacts,
                text_bounds='passed',bookmarks='passed',source_hashes='passed',visual_review='pending')
    (OUT/'PDF_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rendered_pages','contacts')}))


if __name__=='__main__':
    main()
