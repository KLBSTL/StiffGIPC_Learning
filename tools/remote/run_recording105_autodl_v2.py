"""Corrected Linux environment adapter; preserve the original zero-frame attempt."""
import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
original=ROOT/'tools/remote/run_recording105_autodl.py'
spec=importlib.util.spec_from_file_location('original_autodl_controller',original)
base=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=base
spec.loader.exec_module(base)
base.SESSION=ROOT/'runs/autodl_recording105_20261013_v2'
base.OUTPUT=ROOT/'reports/autodl_recording105_20261013_v2'

original_env=base.linux.env_for
def environment(task,out,exe_hash,checkpoint_root=None):
    # The local controller also explicitly sets forcing off. Protocol's Linux
    # launcher requires this metadata dictionary, unlike the local old_task.
    return original_env(task|dict(extra_environment={'GIPC_IPC_PCG_FORCING':'off'}),
                        out,exe_hash,checkpoint_root)
base.linux.env_for=environment
base.__file__=__file__

if __name__=='__main__':
    action=sys.argv[1]
    {'run':base.run,'analyze':base.analyze_results,'render':base.render_results}[action]()
