"""Read-only Windows build seals, with no historical Python-code imports."""
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from local_plan import ROOT,plan
from config import read,digest
from linux_runner import require,sha,inventory,child,write_new

def record(p):
    p=Path(p).resolve();before=p.stat();value=sha(p);after=p.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Identity file changed while hashing')
    return {'path':str(p),'bytes':after.st_size,'sha256':value}

def verify(rows):
    for r in rows:require(record(Path(r['path']))==r,'Identity changed: '+r['path'])

def tree(folder):
    require(folder.is_dir() and not folder.is_symlink(),'Missing/linked identity root')
    rows=[]
    for p in sorted(folder.rglob('*')):
        require(not p.is_symlink(),'Linked identity entry')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':rows.append(record(p))
    return rows

def text(p):
    raw=p.read_bytes();return raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig',errors='replace')

def norm(s):return str(s).replace('\\','/').casefold()

def compiler_records(build):
    rows=[]
    for key in ('CMAKE_CXX_COMPILER','CMAKE_CUDA_COMPILER'):
        candidates=[]
        for f in (build/'CMakeFiles').glob('*/'+('CMakeCXXCompiler.cmake' if key=='CMAKE_CXX_COMPILER' else 'CMakeCUDACompiler.cmake')):
            match=re.search(r'set\('+key+r'\s+"([^"]+)"\)',text(f))
            if match:candidates.append(Path(match.group(1)))
        require(len({str(p.resolve()) for p in candidates})==1,'Compiler path unavailable/ambiguous: '+key)
        p=candidates[0].resolve();require(p.is_file(),'Compiler binary missing')
        argv=[str(p),'--version'] if key=='CMAKE_CUDA_COMPILER' else [str(p)]
        result=subprocess.run(argv,capture_output=True,timeout=15)
        output=(result.stdout+result.stderr).decode(errors='replace').strip()
        require(bool(output) and (result.returncode==0 or key=='CMAKE_CXX_COMPILER'),'Compiler version query failed')
        rows.append({'key':key,'binary':record(p),'command':argv,'returncode':result.returncode,'version_output':output})
    return rows

def project_map(root,build):
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};xml=ET.parse(build/'gipc.vcxproj')
    ints=[n.text for n in xml.findall('.//m:IntDir',ns) if 'Release|x64' in n.attrib.get('Condition','')]
    require(len(ints)==1,'Release IntDir unavailable')
    rows=[]
    for tag,field in (('CudaCompile','CompileOut'),('ClCompile','ObjectFileName')):
        for n in xml.findall('.//m:'+tag,ns):
            if not n.attrib.get('Include'):continue
            p=Path(n.attrib['Include']);p=(p if p.is_absolute() else build/p).resolve()
            values=[v.text for v in n.findall('m:'+field,ns) if not v.attrib.get('Condition') or 'Release|x64' in v.attrib['Condition']]
            require(bool(values),'Per-source Release object name unavailable')
            obj=values[-1].replace('$(IntDir)',ints[0]).replace('$(Configuration)','Release').replace('$(Platform)','x64')
            require('$(' not in obj and '%(' not in obj,'Unresolved object expression')
            object_path=(build/Path(obj.replace('\\','/'))).resolve();require(object_path.is_relative_to(build.resolve()),'Object outside selected build')
            rows.append({'source':str(p),'object':str(object_path),'kind':tag})
    native={norm(p.resolve()) for p in (root/'StiffGIPC').rglob('*') if p.suffix.lower() in ('.cu','.cpp')}
    require({norm(r['source']) for r in rows}==native and len(rows)==len(native),'Exact native source coverage failed')
    require(len({norm(r['object']) for r in rows})==len(rows),'Object names collide on Windows')
    return rows

def active_identity(root,build,exe,log):
    require(build.is_relative_to(root/'build') and exe.is_relative_to(build),'Active output outside standalone build')
    cache=text(build/'CMakeCache.txt');home=re.search(r'^CMAKE_HOME_DIRECTORY:[^=]+=([^\r\n]+)',cache,re.M)
    require(home and Path(home.group(1)).resolve()==root,'Active CMake source root differs')
    mapping=project_map(root,build);logtext=text(log)
    lines=[x.strip() for x in logtext.splitlines() if 'nvcc.exe' in x.lower() and
           re.search(r'(?:\s-c\s|\s--compile(?:\s|$)|\s-x\s+cu\s)',x,re.I) and not re.search(r'--device-link|\s-dlink\s',x,re.I)]
    tracking=build/'gipc.dir/Release/gipc.tlog';cl=tracking/'CL.command.1.tlog';link=tracking/'link.read.1.tlog'
    require(cl.is_file() and link.is_file(),'Actual CL/link tracking evidence missing')
    cltext=norm(text(cl));linktext=norm(text(link));commands=[]
    for r in mapping:
        p=Path(r['source']);obj=Path(r['object']);require(obj.is_file(),'Missing native object: '+str(obj))
        require(norm(obj) in linktext,'Native object absent from gipc link.read: '+str(obj))
        require(p.stat().st_mtime_ns<=obj.stat().st_mtime_ns,'Source newer than compiled object')
        if r['kind']=='CudaCompile':
            matched=[x for x in lines if norm(p) in norm(x)]
            require(matched,'Actual NVCC compile command missing: '+str(p));commands.append({'source':str(p),'commands':matched})
        else:require(norm(p) in cltext,'Actual CL command missing: '+str(p))
    allsources=sum([tree(root/n) for n in ('StiffGIPC','Assets','MeshProcess')],[])+[record(root/'CMakeLists.txt')]
    before_path=build/'build_inputs_before.json';before=read(before_path);verify(before)
    before_map={norm(r['path']):r for r in before}
    require(all(before_map.get(norm(r['path']))==r for r in allsources),'Source/input differs from pre-build snapshot')
    newest_header=max((Path(r['path']).stat().st_mtime_ns for r in allsources if Path(r['path']).suffix in ('.h','.hpp','.cuh','.inl','.inc')),default=0)
    require(newest_header<=min(Path(r['object']).stat().st_mtime_ns for r in mapping),'Header newer than native objects; clean identity ambiguous')
    evidence=[build/'gipc.vcxproj',build/'CMakeCache.txt',before_path,log,*tracking.glob('*.tlog')]
    optional=build/'windows_manifest.json'
    if optional.exists():evidence.append(optional)
    return {'kind':'active','exe':record(exe),'dlls':[record(p) for p in sorted(exe.parent.glob('*.dll'))],
            'source_root':str(root),'sources':allsources,'source_digest':digest(allsources),
            'build_evidence':[record(p) for p in evidence],'objects':[record(Path(r['object'])) for r in mapping],
            'source_object_map':mapping,'cuda_commands':commands,'compilers':compiler_records(build),
            'capabilities':{'resolved_config':True,'actual_velocity':True},
            'scope':'Fresh build owner supplies immutable build; pre-build source snapshot plus exact XML/TU/object/link and actual CUDA/CL command verification. Timestamp fence also rejects newer source/header edits.'}

def base_identity(base_root,manifest_path,exe_path):
    m=read(manifest_path);relative=exe_path.relative_to(base_root).as_posix();expected=m['binaries'][relative]
    require(sha(exe_path)==expected['sha256'],'Frozen Stiff executable differs')
    source=base_root/'sources/stiff_base';rows=[]
    for r in m['files']:
        if r['path'].startswith('sources/stiff_base/'):
            p=child(base_root,r['path']);require(sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'Frozen Stiff source differs');rows.append(record(p))
    require(rows,'Frozen Stiff source identity missing')
    # These are the retained original executable and immutable inputs, not a copied historical runner.
    current=tree(source);known={r['path']:r for r in rows}
    for r in current:
        if r['path'] not in known:rows.append(r)
    build=exe_path.parent.parent;cache=build/'CMakeCache.txt';home=re.search(r'^CMAKE_HOME_DIRECTORY:[^=]+=([^\r\n]+)',text(cache),re.M)
    require(home and Path(home.group(1)).resolve()==source.resolve(),'Frozen Stiff build source root differs')
    stopping=text(source/'StiffGIPC/core/GIPC.cu')
    require(re.search(r'Kmin\s*=\s*6\s*;',stopping) and re.search(r'beta\s*<=\s*Newton_solver_threshold',stopping),'Frozen stopping source differs')
    evidence=[manifest_path,cache,build/'gipc.vcxproj']+list(build.glob('*.log'))+list((build/'gipc.dir/Release/gipc.tlog').glob('*.tlog'))
    return {'kind':'base','exe':record(exe_path),'dlls':[record(p) for p in sorted(exe_path.parent.glob('*.dll'))],
            'source_root':str(source),'sources':rows,'source_digest':digest(rows),'build_evidence':[record(p) for p in evidence],
            'original_manifest':record(manifest_path),'original_source_digest':m['source_digest'],'compilers':compiler_records(build),
            'capabilities':{'resolved_config':False,'actual_velocity':False},
            'scope':'Original retained Windows Stiff executable hash from perf_v34. Historical source plus current Assets/build evidence; not claimed newly rebuilt. Runtime min6 source-confirmed only; no velocity/breakdown telemetry.'}

def tool_inventory(root):
    own=('local_plan.py','local_analysis.py','local_identity.py','windows_runner.py','contracts_test.py','README.md')
    return [record(root/'tools/local'/n) for n in own]+sum([tree(root/'tools'/n) for n in ('bench','diagnostic')],[])

# Exact historical caches which the current 13-run plan does not load. This is
# not a directory/suffix exception: any other baseline-only entry is rejected.
UNUSED_BASE_CACHES=frozenset({
    'sorted_mesh/cipc_table_sorted.16.obj','sorted_mesh/cipc_table_sorted.16.part',
    'sorted_mesh/cloth_high_sorted.16.obj','sorted_mesh/cloth_high_sorted.16.part',
    'sorted_mesh/cube_sorted.16.msh','sorted_mesh/cube_sorted.16.part',
    'sorted_mesh/high_mat_sorted.16.msh','sorted_mesh/high_mat_sorted.16.part'})

def assets_equivalence(active_assets,base_assets):
    inventories=[inventory(folder,[p for p in folder.rglob('*') if p.is_file()]) for folder in (active_assets,base_assets)]
    a,b=({r['path']:r for r in rows} for rows in inventories)
    require(not (a.keys()-b.keys()),'Active-only Assets are forbidden')
    require(all(row==b[name] for name,row in a.items()),'Common Assets differ')
    extras=b.keys()-a.keys();require(extras<=UNUSED_BASE_CACHES,'Unapproved baseline-only Assets')
    selected=sorted({t['config']['scene'] for group in plan()['rounds'].values() for t in group})
    references=set();scenes=[]
    for case in selected:
        relative='benchmark_scenes/'+case+'.json';scene=read(child(active_assets,relative));require(scene['case_id']==case,'Selected scene id differs')
        meshes=[];references.add(relative)
        for obj in scene['objects']:
            relative_mesh=obj['stiff_mesh'];mesh=child(active_assets,relative_mesh)
            require(relative_mesh in a and a[relative_mesh]['sha256']==obj['stiff_mesh_sha256'],'Selected frozen mesh identity differs')
            # metis_sort.cpp uses mesh stem + _sorted.16 + extension and .part.
            caches=['sorted_mesh/'+mesh.stem+'_sorted.16'+mesh.suffix,'sorted_mesh/'+mesh.stem+'_sorted.16.part']
            require(all(name in a for name in caches),'Selected sorted-mesh/cache missing from common Assets')
            references.update([relative_mesh,*caches]);meshes.append({'stiff_mesh':relative_mesh,'sha256':a[relative_mesh]['sha256'],'derived_sorted_cache':caches})
        scenes.append({'scene':case,'scene_sha256':a[relative]['sha256'],'meshes':meshes})
    require(not (references&extras),'Ignored baseline cache is referenced by selected plan')
    ignored=[b[name]|{'reason':'Exact reviewed historical cache; not referenced by selected scene stiff_mesh or its derived .16 sorting cache.'} for name in sorted(extras)]
    return {'passed':True,'files':len(a),'sha256':digest(inventories[0]),'active_only':[],
            'common_files_all_equal':True,'ignored_baseline_only':ignored,'selected_scene_references':scenes,
            'cache_rule':'MeshProcess/metis_partition/src/metis_sort.cpp: stem + _sorted.16 + original extension, and .part',
            'scope':'All active Assets match baseline bytes. Only the eight named unused historical caches may exist solely in baseline; they remain independently hashed in baseline source inventory.'}

def create_seal(root,build,exe,log,base_root,base_manifest,base_exe,guard,out):
    g=read(guard);require(g['allow_next_round'] is False,'Original failed quality gate must remain false')
    active=active_identity(root,build,exe,log);base=base_identity(base_root,base_manifest,base_exe)
    assets=assets_equivalence(Path(active['source_root'])/'Assets',Path(base['source_root'])/'Assets')
    seal={'schema':'windows_cloth_seal.v1','plan':plan(),'programs':{'active':active,'base':base},
          'prior_guard':record(guard),'old_quality_gate_unchanged':True,'quality_protocol':record(root/'tools/bench/quality_protocol.json'),
          'tools':tool_inventory(root),'assets_equivalence':assets,
          'runtime_limits':'Adjacent application DLLs and compiler identities recorded. OS/driver DLL closure is not fully sealed; all local timings remain diagnostic.'}
    write_new(out,seal);return seal

def verify_seal(root,path):
    s=read(path);require(s['schema']=='windows_cloth_seal.v1' and s['plan']==plan(),'Seal plan mismatch')
    require(s['tools']==tool_inventory(root),'Runner/analysis tools changed')
    verify([s['prior_guard'],s['quality_protocol']]);require(read(s['prior_guard']['path'])['allow_next_round'] is False,'Old quality gate altered')
    for ident in s['programs'].values():
        verify(ident['sources']+ident['build_evidence']+[ident['exe']]+ident['dlls']+ident.get('objects',[]))
        source=Path(ident['source_root'])
        current=(sum([tree(source/n) for n in ('StiffGIPC','Assets','MeshProcess')],[])+[record(source/'CMakeLists.txt')]
                 if ident['kind']=='active' else tree(source))
        require({r['path']:r for r in current}=={r['path']:r for r in ident['sources']},'Source/input inventory changed')
        verify([r['binary'] for r in ident['compilers']])
    return s
