"""CPU-only audit of two independent component quality trajectories."""
import argparse
import json
import subprocess
from config import ROOT, read, sha
from ipc_benchmark import write
from audit_autodl_factor import inspect_trace, discover_validator
from report_components import TAG


def main(scene):
    run=ROOT/f'runs/active/{TAG}_{scene}_both_quality'
    folder=ROOT/'reports/active'
    target=folder/f'{TAG}_{scene}_ccd.json'
    assert not target.exists(), 'Audit evidence is immutable'
    result=read(run/'result.json')
    assert result['status']=='completed' and result['recorded_frames']==100
    assert read(run/'requested.json')['expanded_config']['diagnostics']==['substeps']
    inventory=inspect_trace(run,100,seconds=60)
    inventory_path=folder/f'{TAG}_{scene}_accepted_inventory.json'
    write(inventory_path,inventory)
    assert inventory['coverage_passed'] and inventory['all_accepted_states_finite']
    validator=discover_validator()
    native=folder/f'{TAG}_{scene}_native_ccd.json'
    log=folder/f'{TAG}_{scene}_ccd.log'
    assert not native.exists() and not log.exists()
    command=[str(validator),str(run/'trace'),str(native),'substeps','--stable-nh1']
    record={'run':run.name,'scope':'Only this independent diagnostic trajectory; not timing runs',
            'validator_sha256':sha(validator),'command':command,'cpu_timeout_seconds':300,
            'inventory_path':inventory_path.relative_to(ROOT).as_posix(),'inventory_sha256':sha(inventory_path),
            'coverage_passed':inventory['coverage_passed'],'expected_validator_paths':inventory['expected_validator_paths'],
            'quality_certified':False,'performance_certified':False}
    try:
        with log.open('x',encoding='utf-8') as output:
            process=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=300)
        record['exit_code']=process.returncode
        if native.exists():
            record['ccd']=read(native)
            record['native_sha256']=sha(native)
            ccd=record['ccd']
            record['passed']=process.returncode==0 and ccd['paths_checked']==inventory['expected_validator_paths'] \
                and ccd['conservative_collision_flags']==0 and ccd.get('abd_tet_inversions',0)==0
        else:record.update(passed=False,failure='Missing independent CCD output')
    except subprocess.TimeoutExpired:
        record.update(passed=False,failure='CPU audit exceeded declared 300 seconds')
    write(target,record)
    print(json.dumps({k:record[k] for k in ('run','passed','expected_validator_paths','failure') if k in record}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',choices=['hang','fixed_bunny'],required=True)
    main(parser.parse_args().scene)
