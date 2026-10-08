"""Read-only Windows build seals, with no historical Python-code imports."""
import re
import hashlib
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

def _verify_build_source_records(identity,build):
    """Bind original source bytes to both retained build inventories."""
    before={norm(r['path']):r for r in read(build/'build_inputs_before.json')}
    manifest=read(build/'windows_manifest.json')
    require(manifest['kind']=='active' and manifest['status']=='completed' and
            Path(manifest['source_root']).resolve()==Path(identity['source_root']).resolve() and
            Path(manifest['build_dir']).resolve()==build.resolve() and
            manifest['exe']==identity['exe'],'Frozen build manifest differs')
    compiled={norm(r['path']):r for r in manifest['source_inventory']}
    originals=identity.get('frozen_source',{}).get('original_sources',identity['sources'])
    require(len({norm(r['path']) for r in originals})==len(originals),'Duplicate original source records')
    require(all(before.get(norm(r['path']))==r and compiled.get(norm(r['path']))==r for r in originals),
            'Source bytes differ from pre-build or compiled manifest inventory')

def _verified_active_build(identity,root,build,exe,log):
    require(identity['kind']=='active' and Path(identity['source_root']).resolve()==root and
            identity['exe']==record(exe) and digest(identity['sources'])==identity['source_digest'] and
            not identity.get('frozen_source'),'Previously verified active identity differs')
    verify(identity['build_evidence']+identity['objects']+[identity['exe']]+identity['dlls']+
           [r['binary'] for r in identity['compilers']])
    evidence={Path(r['path']).resolve() for r in identity['build_evidence']}
    link=build/'gipc.dir/Release/gipc.tlog/link.read.1.tlog'
    require({build/'build_inputs_before.json',build/'windows_manifest.json',build/'gipc.vcxproj',log,link}<=evidence,
            'Original compiled build evidence incomplete')
    mapping=identity['source_object_map'];objects={norm(r['path']) for r in identity['objects']}
    native={norm(r['path']) for r in identity['sources'] if
            Path(r['path']).is_relative_to(root/'StiffGIPC') and Path(r['path']).suffix.lower() in ('.cu','.cpp')}
    require(len(mapping)==len(native) and {norm(r['source']) for r in mapping}==native and
            len(objects)==len(mapping) and {norm(r['object']) for r in mapping}==objects,
            'Original source/object coverage differs')
    linktext=norm(text(link))
    require(all(norm(r['object']) in linktext for r in mapping),'Original native object absent from actual link evidence')
    _verify_build_source_records(identity,build)

def _git_source_bytes(root,commit,relative):
    resolved=subprocess.run(['git','-C',str(root),'rev-parse','--verify',commit+'^{commit}'],
                            capture_output=True,check=True,timeout=15).stdout.decode('ascii').strip()
    require(re.fullmatch(r'[0-9a-f]{40}',resolved) is not None,'Git source commit unresolved')
    raw=subprocess.run(['git','-C',str(root),'cat-file','blob',resolved+':'+relative.as_posix()],
                       capture_output=True,check=True,timeout=15).stdout
    return raw,{'kind':'git_blob','commit':resolved,'path':relative.as_posix()}

def freeze_active_identity(root,build,exe,log,snapshot,from_verified_identity=None,git_commit=None):
    """Freeze verified native inputs once; continue checking actual runtime Assets."""
    root,build,exe,log,snapshot=(Path(p).resolve() for p in (root,build,exe,log,snapshot))
    require(snapshot.is_relative_to(root/'build') and snapshot!=root/'build' and not snapshot.exists(),
            'Fresh private snapshot below build/ required')
    prior=None
    if from_verified_identity is None:identity=active_identity(root,build,exe,log)
    else:
        prior=record(from_verified_identity);identity=read(from_verified_identity)
    _verified_active_build(identity,root,build,exe,log)
    snapshot.mkdir(parents=True)
    for folder in ('StiffGIPC','MeshProcess'):(snapshot/folder).mkdir()
    rows=[];mapping=[]
    for original in identity['sources']:
        source=Path(original['path']);relative=source.relative_to(root)
        if relative.parts[0]=='Assets':
            rows.append(original);continue
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes() if source.is_file() else None
        matches=raw is not None and len(raw)==original['bytes'] and hashlib.sha256(raw).hexdigest()==original['sha256']
        if matches:origin={'kind':'current_verified_bytes','path':str(source)}
        else:
            require(git_commit is not None,'Original source bytes unavailable: '+str(source))
            raw,origin=_git_source_bytes(root,git_commit,relative)
        require(len(raw)==original['bytes'] and hashlib.sha256(raw).hexdigest()==original['sha256'],
                'Recovered source does not exactly match compiled bytes: '+str(source))
        with target.open('xb') as stream:stream.write(raw)
        frozen=record(target)
        require((frozen['bytes'],frozen['sha256'])==(original['bytes'],original['sha256']),
                'Source changed while copying snapshot: '+str(source))
        rows.append(frozen);mapping.append({'original':original,'frozen':frozen,'origin':origin})
    verify([r for r in identity['sources'] if Path(r['path']).is_relative_to(root/'Assets')])
    _verified_active_build(identity,root,build,exe,log)
    if prior:verify([prior])
    frozen=dict(identity,sources=rows,frozen_source={'root':str(snapshot),
        'original_sources':identity['sources'],'mapping':mapping,
        'build_inputs_before':record(build/'build_inputs_before.json'),
        'windows_manifest':record(build/'windows_manifest.json')})
    if prior:frozen['frozen_source']['verified_identity']=prior
    frozen['scope']+=' Native source bytes are retained in a private snapshot; actual Assets remain verified at their runtime paths.'
    verify_active_sources(frozen)
    write_new(snapshot/'snapshot_identity.json',frozen)
    return frozen

def verify_active_sources(identity):
    source=Path(identity['source_root']);frozen=identity.get('frozen_source')
    if frozen:
        snapshot=Path(frozen['root'])
        require(snapshot.resolve().is_relative_to(source.resolve()/'build') and not snapshot.is_symlink(),
                'Frozen snapshot outside private build directory')
        require(digest(frozen['original_sources'])==identity['source_digest'],'Original source digest differs')
        verify([frozen['build_inputs_before'],frozen['windows_manifest']])
        if frozen.get('verified_identity'):verify([frozen['verified_identity']])
        _verify_build_source_records(identity,Path(frozen['windows_manifest']['path']).parent)
        original_code=[r for r in frozen['original_sources'] if not Path(r['path']).is_relative_to(source/'Assets')]
        require({r['path']:r for r in original_code}=={m['original']['path']:m['original'] for m in frozen['mapping']}
                and len(original_code)==len(frozen['mapping']),'Frozen source mapping incomplete')
        for m in frozen['mapping']:
            original,saved=m['original'],m['frozen']
            require(Path(saved['path'])==snapshot/Path(original['path']).relative_to(source) and
                    (saved['bytes'],saved['sha256'])==(original['bytes'],original['sha256']),
                    'Frozen source mapping differs')
        expected=[m['frozen'] for m in frozen['mapping']]+[
            r for r in frozen['original_sources'] if Path(r['path']).is_relative_to(source/'Assets')]
        require({r['path']:r for r in expected}=={r['path']:r for r in identity['sources']},
                'Frozen source records do not match original bytes')
        current=sum([tree(snapshot/n) for n in ('StiffGIPC','MeshProcess')],[])+[
            record(snapshot/'CMakeLists.txt')]+tree(source/'Assets')
    else:
        current=sum([tree(source/n) for n in ('StiffGIPC','Assets','MeshProcess')],[])+[record(source/'CMakeLists.txt')]
    require({r['path']:r for r in current}=={r['path']:r for r in identity['sources']},'Source/input inventory changed')

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
    s=read(path)
    if s.get('schema')=='windows_execution_observation.v1':
        return verify_execution_observation_seal(root,s)
    require(s['schema']=='windows_cloth_seal.v1' and s['plan']==plan(),'Seal plan mismatch')
    require(s['tools']==tool_inventory(root),'Runner/analysis tools changed')
    verify([s['prior_guard'],s['quality_protocol']]);require(read(s['prior_guard']['path'])['allow_next_round'] is False,'Old quality gate altered')
    for ident in s['programs'].values():
        verify(ident['sources']+ident['build_evidence']+[ident['exe']]+ident['dlls']+ident.get('objects',[]))
        source=Path(ident['source_root'])
        if ident['kind']=='active':verify_active_sources(ident)
        else:require({r['path']:r for r in tree(source)}=={r['path']:r for r in ident['sources']},'Source/input inventory changed')
        verify([r['binary'] for r in ident['compilers']])
    return s

def execution_observation_tools(root):
    rows=tool_inventory(root)+[record(root/'tools/local'/name) for name in
        ('profiler_windows.py','profiler_windows_contracts_test.py','windows_owned_job.py')]
    require(len({norm(r['path']) for r in rows})==len(rows),'Duplicate observation tool identity')
    return rows

def _verify_observation_program(root,identity):
    require(identity['kind']=='active' and Path(identity['source_root']).resolve()==root.resolve(),
            'Observation requires an active workspace identity')
    exe=Path(identity['exe']['path']).resolve();build=exe.parent.parent
    require(build.is_relative_to(root.resolve()/'build') and exe.parent.name=='Release',
            'Observation requires an independently built Release program')
    verify(identity['sources']+identity['build_evidence']+identity['objects']+[identity['exe']]+identity['dlls'])
    verify_active_sources(identity)
    original=dict(identity)
    if original.get('frozen_source'):
        original['sources']=original.pop('frozen_source')['original_sources']
    require(digest(original['sources'])==original['source_digest'],'Observation source digest differs')
    _verified_active_build(original,root.resolve(),build,exe,build/'build.log')

def verify_execution_observation_seal(root,seal):
    fields={'schema','programs','plan_record','prior_guard','tools','performance_certified',
            'quality_certified','profile_attempts_per_program'}
    require(set(seal)==fields and seal['schema']=='windows_execution_observation.v1' and
            set(seal['programs'])=={'active'},'Observation seal schema/fields differ')
    require(seal['performance_certified'] is False and seal['quality_certified'] is False and
            type(seal['profile_attempts_per_program']) is int and seal['profile_attempts_per_program']==2,
            'Observation certification or capture budget differs')
    require(seal['tools']==execution_observation_tools(root),'Observation tools changed')
    verify([seal['plan_record'],seal['prior_guard']])
    require(read(seal['prior_guard']['path'])['allow_next_round'] is False,'Original failed quality gate altered')
    _verify_observation_program(root,seal['programs']['active'])
    return seal

def create_execution_observation_seal(root,identity,plan_path,prior_guard,out):
    """Seal an observation plan without impersonating the historical 13-run plan."""
    root=Path(root).resolve();out=Path(out)
    require(not out.exists(),'Observation seal exists; preserve immutable evidence')
    seal={'schema':'windows_execution_observation.v1','programs':{'active':identity},
        'plan_record':record(plan_path),'prior_guard':record(prior_guard),
        'tools':execution_observation_tools(root),'performance_certified':False,
        'quality_certified':False,'profile_attempts_per_program':2}
    verify_execution_observation_seal(root,seal)
    write_new(out,seal)
    return seal
