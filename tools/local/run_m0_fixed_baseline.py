"""One same-state Bunny matrix capture for explicit, independent residual checks."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/local'))
import run_innovation_m0 as m


def main():
    parent=ROOT/'runs/innovation_m0_20261014'
    packet=m.read(parent/'seal.json');identity=packet['programs']['active']
    cp=parent/'prefix_bunny_movement/checkpoint'
    session=ROOT/'runs/innovation_m0_fixed_20261014'
    m.require(not session.exists(),'Fresh capture directory required');session.mkdir(parents=True)
    c=m.configuration('Bunny',1,'movement_only')
    c=m.expand(c|dict(diagnostics=['fixed'],fixed_frames='14',fixed_directions='1'))
    t=dict(name='fixed_bunny_f14',scene='Bunny',config=c,binary='active',target_frame=14)
    original_env=m.windows_runner.environment;original_query=m.windows_runner.gpu_query
    expected=packet['power_guid']
    def env(config,out):
        value=m.environment(config,out)
        value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'],
            GIPC_CHECKPOINT_LOAD=str(cp))
        return value
    def query(gpu,timeout=10):
        value=original_query(gpu,timeout=timeout)
        m.require(m.power_guid()==expected,'Power changed in fixed capture')
        return value
    m.write_new(session/'declared.json',dict(task=t,parent_seal=m.record(parent/'seal.json'),
        controller=m.record(Path(__file__)),runner=m.record(Path(m.windows_runner.__file__)),
        purpose='Diagnostic saved full A,b and MAS; not a timed performance arm'))
    try:
        with m.windows_runner.gpu_lock(ROOT):
            m.windows_runner.environment,m.windows_runner.gpu_query=env,query
            result=m.windows_runner.execute(session,t,identity,0)
    finally:m.windows_runner.environment,m.windows_runner.gpu_query=original_env,original_query
    m.require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Fixed capture failed')
    restored=m.validate_restore(session/t['name'],cp,identity,14)
    files=list((session/t['name']/'fixed').glob('*_study.json'))
    m.require(len(files)==1,'Expected exactly one selected matrix')
    study=m.read(files[0])
    m.require(study['system_unchanged'] and study['primary_restored_bitwise'],'Fixed probe changed its production system/solution')
    m.write_new(session/'capture_gate.json',dict(passed=True,restore=restored,
        study=m.record(files[0]),system_unchanged=True,primary_restored_bitwise=True))
    print(json.dumps(dict(capture_complete=True,study=str(files[0]))))


if __name__=='__main__':main()
