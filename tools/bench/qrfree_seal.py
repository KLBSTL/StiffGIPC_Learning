"""Seal current evidence, retaining failed attempts and the parent TODO mapping."""
import ast
import json
from pathlib import Path
import re
import shutil
from m3_numeric import ROOT,need
from local_identity import record,verify,tree
from linux_runner import write_new

REPORT=ROOT/'reports/innovation_p5_qrfree_20261016'
RAW=ROOT/'runs/innovation_p5_qrfree_20261016'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    audit=read(REPORT/'AUDIT.json');need(audit['passed'] and audit['solutions_rechecked']==6 and audit['curve_rows']==492,'Incomplete audit')
    verify([audit['source']]);todo=ROOT/'docs/research/INNOVATION_TODO_20261014.md'
    prior=REPORT/'prior_plan'/todo.name;parent_path=ROOT/'reports/innovation_p5_cgstats_20261015/DELIVERY_SEAL.json'
    parent=read(parent_path);mapping=[]
    for r in parent['files']:
        if Path(r['path']).resolve()==todo.resolve():
            saved=record(prior);need((saved['bytes'],saved['sha256'])==(r['bytes'],r['sha256']),'Parent TODO bytes differ')
            mapping.append(dict(original=r,archive=saved))
        else:verify([r])
    need(len(mapping)==1,'Parent TODO mapping missing')
    production=read(ROOT/'runs/innovation_m3_20261015/seal.json')['identity']
    verify(production['frozen_source']['original_sources']+[production['exe']]+production['dlls'])
    identity=read(ROOT/'build/qrfree_gpu_v2_20261016/identity.json')
    verify(identity['sources']+identity['evidence']+[identity['exe']])
    log=(ROOT/'build/qrfree_gpu_v2_20261016/build.log').read_text(encoding='utf-8-sig',errors='replace')
    need('compute_86' in log and '/utf-8' in log and '--use_fast_math' not in log,'Compiler flags differ')
    original=ROOT/'tools/bench/cgstats_gpu/cgstats_cost.cu';header=ROOT/'tools/bench/qrfree_gpu/frozen_runtime.cuh'
    prefix=original.read_text(encoding='utf-8').split('void run(const fs::path& folder,const fs::path& output){',1)[0]
    need(prefix==header.read_text(encoding='utf-8'),'Runtime textual prefix differs')
    origin=read(ROOT/'tools/bench/qrfree_gpu/ORIGIN.json');need(origin['sha256']==record(original)['sha256'],'Origin hash differs')
    write_new(REPORT/'RUNTIME_COPY_CHECK.json',dict(passed=True,original=record(original),copy=record(header),
        normalized_text_equal=True,raw_bytes_equal=False,operation='LF-to-CRLF line ending normalization only'))
    sources=[ROOT/'tools/bench'/f for f in ('qrfree_numeric.py','qrfree_numeric_test.py','run_qrfree_cpu.py',
        'prepare_qrfree_gpu.py','validate_qrfree_gpu.py','qrfree_audit_export.py','qrfree_seal.py')]
    sources += [ROOT/'tools/local/build_qrfree_probe.py',ROOT/'tools/local/run_qrfree_probe.py']
    sources += sorted((ROOT/'tools/bench/qrfree_gpu').glob('*'))
    archives=[]
    for p in sources:
        value=p.read_text(encoding='utf-8-sig');need(all(x.rstrip()==x for x in value.splitlines()),'Trailing whitespace: '+str(p))
        if p.suffix=='.py':ast.parse(value)
        saved=REPORT/'sources'/p.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True)
        need(not saved.exists(),'Snapshot exists');shutil.copy2(p,saved);archives.append(dict(original=record(p),archive=record(saved)))
    for p in REPORT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            need(link.startswith('https://') or (p.parent/link).exists(),'Missing report link: '+link)
    inputs=[]
    for name in ('inputs','fixture_positive_inputs','fixture_negative_inputs'):
        m=read(RAW/name/'manifest.json');verify(m['records']);inputs+=m['records']
    for name in ('fixture_positive','fixture_negative','mixed_cost','fixture_memcheck_v2'):
        launch=read(RAW/name/'launch.json');need(launch['passed'] and launch['cleanup_owned_job_empty'],'GPU Job failed')
        verify([launch['runner']]+launch['identity']['sources']+launch['identity']['evidence']+[launch['identity']['exe']])
    failed=read(RAW/'fixture_memcheck/launch.json');need(not failed['passed'] and failed['cleanup_owned_job_empty'],'Failed Job missing')
    records=tree(RAW)+tree(REPORT)+tree(ROOT/'build/qrfree_gpu_v2_20261016')+tree(ROOT/'build/qrfree_gpu_20261016')
    records+=inputs+[record(p) for p in sources]+[record(todo),record(parent_path),record(production['exe']['path'])]
    records+=production['frozen_source']['original_sources']
    records += [record(p) for p in sorted((ROOT/'build').glob('qrfree_*'))
                if p.is_file() and p.suffix=='.log' and 'seal' not in p.name]
    unique={r['path']:r for r in records};files=[unique[k] for k in sorted(unique)];verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='p5_qrfree_delivery.v1',files=files,source_archives=archives,
        parent=record(parent_path),parent_files_verified=len(parent['files']),parent_mutable_mapping=mapping,
        Python_tests=90,memcheck_errors=0,successful_GPU_jobs=4,failed_sandbox_GPU_jobs=1,
        native_PCG=False,production_modified=False,performance_certified=False))
    print(json.dumps(dict(sealed=True,files=len(files),sources=len(sources),Python_tests=90,memcheck_errors=0)),flush=True)


if __name__=='__main__':main()
