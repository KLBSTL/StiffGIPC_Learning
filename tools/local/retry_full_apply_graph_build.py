"""One explicit compile retry after adding the missing frozen ABD type header."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
import prepare_full_apply_graph as p
from local_identity import record,verify,tree,compiler_records
from linux_runner import require,write_new
from build_windows import CMAKE,checked

def main():
    require(not (p.BUILD/'identity.json').exists(),'Build already sealed')
    failed=(p.BUILD/'build.log').read_text(encoding='utf-8-sig',errors='replace')
    require('has no member "Matrix12x12"' in failed and not (p.BUILD/'build_retry1.log').exists(),'Only the declared retry is allowed')
    parent=p.read(p.REPORT/'PARENT.json');verify(parent['original_source_files']+[r['seal'] for r in parent['parents']])
    sources=tree(p.PROJECT)+tree(p.BUILD/'overlay')+parent['original_source_files']+[record(Path(__file__)),record(Path(p.__file__))]
    write_new(p.BUILD/'retry1_sources_before.json',sources)
    command=[str(CMAKE),'--build',str(p.BUILD),'--config','Release','--target','full_apply_graph','--','/m:2','/v:normal']
    checked(command,p.BUILD/'build_retry1.log');verify(sources)
    write_new(p.BUILD/'identity.json',dict(sources=sources,exe=record(p.BUILD/'Release/full_apply_graph.exe'),
        runtime=[record(v) for v in sorted((p.CUDA/'bin/x64').glob('cublas*.dll'))],compilers=compiler_records(p.BUILD),
        command=command,parents=parent['parents'],production_solver_modified=False,
        build_evidence=[record(p.BUILD/name) for name in ['CMakeCache.txt','full_apply_graph.vcxproj','configure.log','build.log',
            'build_retry1.log','retry1_sources_before.json']],
        initial_failure='Missing gipc/type_define.h in new standalone driver; original source unchanged; no GPU run before correction'))
    print(json.dumps(dict(compiled=True,retry=1,binary=record(p.BUILD/'Release/full_apply_graph.exe'))),flush=True)
if __name__=='__main__':main()
