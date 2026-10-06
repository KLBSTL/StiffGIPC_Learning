"""Carry the three verified v38 typed kernels into v39, with distinct names."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
source=(r/'sources/mas_replay_v38/native_kernels.cuh').read_text()
pieces=[]
for name in ['__collectFinalZ_new','__buildMultiLevelR_optimized_new','_schwarzLocalXSym6']:
    start=source.index(name+'(');start=source.rfind('template<',0,start)
    opening=source.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    pieces.append(source[start:end].replace(name,name+'_wide'))
p=r/'sources/stiff_perf_v39/StiffGIPC/solver/mas_wide_kernels.inl'
assert not p.exists()
p.write_text('// Derived from frozen v38 typed kernels; upstream MAS MPL provenance retained.\n'+'\n\n'.join(pieces))
