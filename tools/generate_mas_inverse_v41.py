from pathlib import Path
r=Path(__file__).resolve().parents[1]
source=(r/'sources/mas_replay_v40/native_kernels.cuh').read_text()
name='__inverse6_P96x96';start=source.index(name+'(');start=source.rfind('template<',0,start)
opening=source.index('{',start);end=opening+1;depth=1
while depth:
    depth+=(source[end]=='{')-(source[end]=='}');end+=1
target=r/'sources/stiff_perf_v41/StiffGIPC/solver/mas_inverse64_kernel.inl'
assert not target.exists()
target.write_text('// Frozen v40 typed inverse; upstream MAS MPL provenance retained.\n'+source[start:end].replace(name,name+'_typed')+'\n')
