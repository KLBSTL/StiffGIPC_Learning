"""Recover historical bytes; accept only exact recorded SHA256 matches."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent/'StiffGIPC_Learning'
ORIGIN=ROOT.parent/'stiff_toi_cudagraph_20260929'
MANIFEST=ORIGIN/'manifests/3777a77b831b67fa3009c3c4825db40b752bfabfbc3a8ec9b1b7cb4ff48ee287.json'
HISTORY=Path('C:/Users/86189/.codex/sessions/2026/09/26/rollout-2026-09-26T19-01-26-01a0dd60-87aa-7e41-a022-b5f8bcf62f32.jsonl')
DEST=ROOT/'build/graph_v4_recovered_source'
PREFIX='sources/stiff_fused/'


def sections(patch,prefix=PREFIX):
    lines=patch.splitlines();out=[];current=None
    for line in lines:
        match=re.match(r'\*\*\* (Add|Update|Delete) File: (.*)',line)
        if match:
            path=match[2].replace('\\','/')
            current=None
            if prefix in path:
                current=[match[1],path.split(prefix,1)[1],[]];out.append(current)
        elif line.startswith('*** '):
            current=None
        elif current is not None:
            current[2].append(line)
    return out


def update(text, lines, reverse=False, preserve=False):
    blocks=[];block=[]
    for line in lines:
        if line.startswith('@@'):
            if block:blocks.append(block)
            block=[]
        elif line and line[0] in ' +-':block.append(line)
    if block:blocks.append(block)
    raw=text.splitlines(keepends=True)
    values=[s.rstrip('\r\n') for s in raw]
    for block in reversed(blocks) if reverse else blocks:
        old=[s[1:] for s in block if s[0]!='+']
        new=[s[1:] for s in block if s[0]!='-']
        if reverse:old,new=new,old
        if not reverse and new!=old and new and any(values[i:i+len(new)]==new for i in range(len(values)-len(new)+1)):
            continue
        starts=[i for i in range(len(values)-len(old)+1) if values[i:i+len(old)]==old]
        if len(starts)!=1:raise ValueError('Hunk absent or ambiguous')
        at=starts[0]
        if preserve:
            replacement=[];offset=0
            for item in block:
                sign=item[0]
                if reverse:sign='-' if sign=='+' else '+' if sign=='-' else sign
                if sign==' ':replacement.append(item[1:]+'\n' if preserve==2 else raw[at+offset]);offset+=1
                elif sign=='-':offset+=1
                elif sign=='+':replacement.append(item[1:]+'\n')
            raw[at:at+len(old)]=replacement
        values[at:at+len(old)]=new
    return ''.join(raw) if preserve else '\n'.join(values)+'\n'


def main():
    expected={r['path'][len(PREFIX):]:r for r in json.loads(MANIFEST.read_text())['files'] if r['path'].startswith(PREFIX)}
    raw=subprocess.check_output(['git','-C',str(REPO),'archive','archive/pre-cleanup-20261006','sources/stiff_fused','sources/stiff_base'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        blobs={m.name:archive.extractfile(m).read() for m in archive if m.isfile()}
    found={};evidence={};patches=[];base_patches=[];closest={}
    def consider(name,data,origin):
        if name not in expected or data is None:return
        for mode,b in [('raw',data),('lf',data.replace(b'\r\n',b'\n')),('crlf',data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))]:
            difference=abs(len(b)-expected[name]['bytes'])
            if name not in closest or difference<closest[name][0]:closest[name]=(difference,b,origin,mode)
            if len(b)==expected[name]['bytes'] and hashlib.sha256(b).hexdigest()==expected[name]['sha256']:
                found[name]=b;evidence[name]=dict(origin=origin,newline_mode=mode)
    for name in expected:
        cached=DEST/name
        if cached.is_file():consider(name,cached.read_bytes(),'previous hash-verified recovery')
        for prefix in ('sources/stiff_fused/','sources/stiff_base/'):
            consider(name,blobs.get(prefix+name),'Git archive '+prefix)
        for folder in (ORIGIN/'sources/stiff_base',ROOT,ROOT/'baseline'):
            p=folder/name
            if p.is_file():consider(name,p.read_bytes(),'current verified candidate '+str(p))
    for lineno,line in enumerate(HISTORY.open(encoding='utf-8'),1):
        data=json.loads(line)
        if data.get('type')!='response_item':continue
        payload=data.get('payload',{})
        if payload.get('type') not in ('custom_tool_call','function_call'):continue
        source=payload.get('input',payload.get('arguments',''))
        if PREFIX not in source and '"stiff_fused"' not in source and 'sources/stiff_base/' not in source:continue
        variables={}
        for declaration in re.finditer(r'(?:const|let)\s+(\w+)\s*=\s*',source):
            try:
                value,_=json.JSONDecoder().raw_decode(source[declaration.end():])
                if isinstance(value,str) and value.startswith('*** Begin Patch'):variables[declaration[1]]=value
            except ValueError:pass
        for match in re.finditer(r'apply_patch\(',source):
            try:
                argument=source[match.end():]
                if argument.startswith('`'):
                    stop=argument.index('`',1)
                    patch=argument[1:stop].replace('${root}',str(ORIGIN/'sources/stiff_fused').replace('\\','/'))
                else:
                    variable=re.match(r'(\w+)\s*\)',argument)
                    if variable and variable[1] in variables:patch=variables[variable[1]]
                    else:patch,_=json.JSONDecoder().raw_decode(argument)
                if not isinstance(patch,str):continue
                for item in sections(patch):patches.append((lineno,item))
                base_patch=patch.replace(str(ORIGIN/'sources/stiff_fused').replace('\\','/'),str(ORIGIN/'sources/stiff_base').replace('\\','/')) if argument.startswith('`') else patch
                for item in sections(base_patch,'sources/stiff_base/'):
                    base_patches.append((lineno,item))
            except (ValueError,TypeError):pass
    failures=[];pcg_states=[]
    for prefix,reverse,preserve in [('sources/stiff_base/',False,False),('sources/stiff_fused/',True,False),
                                  ('disk_base',False,True),('disk_base',False,False),('disk_base',False,2),
                                  ('disk_fused',True,True),('disk_fused',True,2),
                                  ('precopy_base',False,True),('precopy_base',False,2)]:
        if prefix in ('disk_base','precopy_base'):
            state={name:(ORIGIN/'sources/stiff_base'/name).read_bytes().decode('utf-8-sig')
                   for name in expected if not name.startswith('Assets/') and (ORIGIN/'sources/stiff_base'/name).is_file()}
            if prefix=='precopy_base':
                for lineno,(kind,name,lines) in reversed(base_patches):
                    if not 5467<lineno<6916 or kind!='Update' or name not in state:continue
                    try:state[name]=update(state[name],lines,True,False)
                    except ValueError:pass
                state={name:text.replace('\r\n','\n').replace('\n','\r\n') for name,text in state.items()}
        elif prefix=='disk_fused':
            state={name:(ORIGIN/'sources/stiff_robust_port'/name).read_bytes().decode('utf-8-sig')
                   for name in expected if not name.startswith('Assets/') and (ORIGIN/'sources/stiff_robust_port'/name).is_file()}
        else:
            state={name:raw.decode('utf-8-sig') for name in expected if (raw:=blobs.get(prefix+name)) is not None and '/Assets/' not in prefix+name}
        for lineno,(kind,name,lines) in reversed(patches) if reverse else patches:
            if name not in expected or name.startswith('Assets/'):continue
            try:
                if kind=='Add':
                    if reverse:state.pop(name,None);continue
                    state[name]='\n'.join(s[1:] for s in lines if s.startswith('+'))+'\n'
                elif kind=='Update':state[name]=update(state[name],lines,reverse,preserve)
                else:continue
                consider(name,state[name].encode('utf-8'),dict(history_line=lineno,reverse=reverse,preserve=preserve))
                if name=='StiffGIPC/linear_system/solver/pcg_solver.cu' and lineno<=6916:
                    raw=state[name].encode('utf-8');lf=raw.replace(b'\r\n',b'\n')
                    pcg_states.append(dict(line=lineno,prefix=prefix,reverse=reverse,preserve=preserve,
                                          raw=len(raw),lf=len(lf),crlf=len(lf.replace(b'\n',b'\r\n')),
                                          host_repeat=b'host_repeat_relative' in raw))
                    if prefix=='precopy_base':
                        path=DEST/'recovery_candidates'/f'pcg_{lineno}_{preserve}.cu'
                        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            except (ValueError,KeyError) as error:
                failures.append(dict(name=name,line=lineno,reverse=reverse,error=str(error)))
    DEST.mkdir(parents=True,exist_ok=True)
    for name in expected:
        if name in found or name not in closest:continue
        _,data,origin,mode=closest[name]
        delta=expected[name]['bytes']-len(data)
        if not 0<abs(delta)<=2:continue
        locations=[i for i,c in enumerate(data) if c==10 and (i==0 or data[i-1]!=13)] if delta>0 else [i for i,c in enumerate(data) if c==13 and i+1<len(data) and data[i+1]==10]
        if len(locations)>500:continue
        for positions in itertools.combinations(locations,abs(delta)):
            pieces=[];at=0
            for position in positions:
                pieces.append(data[at:position]);pieces.append(b'\r' if delta>0 else b'');at=position if delta>0 else position+1
            pieces.append(data[at:]);candidate=b''.join(pieces)
            if hashlib.sha256(candidate).hexdigest()==expected[name]['sha256']:
                consider(name,candidate,dict(origin=origin,newline_byte_recovery=list(positions)))
                break
    for name,raw in found.items():
        p=(DEST/name).resolve()
        if not p.is_relative_to(DEST.resolve()):raise ValueError('Unsafe relative path')
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=raw:raise ValueError('Recovered bytes changed')
        p.write_bytes(raw)
    missing=[name for name in expected if name not in found]
    candidate_dir=DEST/'recovery_candidates'
    for name in missing:
        if name not in closest:continue
        p=candidate_dir/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(closest[name][1])
    report=dict(expected=len(expected),recovered=len(found),missing=missing,evidence=evidence,
                historical_manifest=str(MANIFEST),historical_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                recovery_history=str(HISTORY),patch_failures=failures,closest={k:dict(bytes=len(v[1]),difference=v[0],origin=v[2],mode=v[3]) for k,v in closest.items() if k in missing},
                exact_native_complete=not any(not x.startswith('Assets/') for x in missing))
    report['pcg_states']=pcg_states
    (DEST/'RECOVERY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(expected=len(expected),recovered=len(found),missing=missing,exact_native_complete=report['exact_native_complete'])))


if __name__=='__main__':main()
