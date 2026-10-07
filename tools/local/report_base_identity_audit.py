"""Compare retained baseline and read-only observer build evidence, no GPU.

Matching cache flags do not prove old objects were freshly compiled. The
original retained executable remains the denominator, never the observer.
"""
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from local_identity import record,text,norm
from linux_runner import write_new

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/report_execution_20261007'
LEGACY=ROOT.parent/'stiff_toi_cudagraph_20260929/builds/local-base'
OBSERVER=ROOT/'build/report_base_observer_20261007_v2'

def cache(path):
    keys=('CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_FLAGS_RELEASE','CMAKE_CXX_FLAGS_RELEASE',
          'CMAKE_CUDA_COMPILER','CMAKE_CXX_COMPILER','CMAKE_MSVC_RUNTIME_LIBRARY')
    text=path.read_text(encoding='utf-8-sig')
    return {k:next(iter(re.findall(r'^'+k+r':[^=]+=(.*)$',text,re.M)),None) for k in keys}

def units(path):
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};tree=ET.parse(path)
    sources=[n.attrib['Include'] for kind in ('CudaCompile','ClCompile')
             for n in tree.findall('.//m:'+kind,ns) if n.attrib.get('Include')]
    counts=Counter(Path(p).name.casefold() for p in sources)
    return dict(source_count=len(sources),basename_collisions={k:v for k,v in counts.items() if v>1})

def main():
    import sys
    if '--object-paths' in sys.argv:
        ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
        project=ET.parse(LEGACY/'gipc.vcxproj')
        intdir=next(n.text for n in project.findall('.//m:IntDir',ns)
                    if 'Release|x64' in n.attrib.get('Condition',''))
        fallback=next(c.find('m:CompileOut',ns).text
            for group in project.findall('.//m:ItemDefinitionGroup',ns)
            if 'Release|x64' in group.attrib.get('Condition','')
            for c in group.findall('m:CudaCompile',ns))
        mapping=[]
        for n in project.findall('.//m:CudaCompile',ns):
            if not n.attrib.get('Include'):continue
            overrides=n.findall('m:CompileOut',ns)
            value=(overrides[-1].text if overrides else fallback).replace('$(IntDir)',intdir)
            value=value.replace('%(Filename)',Path(n.attrib['Include']).stem)
            if '$(' in value or '%(' in value:raise ValueError('Unresolved legacy CUDA object path')
            mapping.append(dict(source=str((LEGACY/n.attrib['Include']).resolve()),
                object=str((LEGACY/value.replace('\\','/')).resolve())))
        linked=text(LEGACY/'gipc.dir/Release/gipc.tlog/link.read.1.tlog')
        output=dict(schema='report.baseline.object.audit.v1',
            project=record(LEGACY/'gipc.vcxproj'),link_tracking=record(LEGACY/'gipc.dir/Release/gipc.tlog/link.read.1.tlog'),
            mapping=mapping,unique_resolved_object_paths=len({norm(r['object']) for r in mapping})==len(mapping),
            all_objects_present=all(Path(r['object']).is_file() for r in mapping),
            all_objects_in_link=all(norm(r['object']) in norm(linked) for r in mapping),
            scope='CUDA-only per-source Release object paths and link tracking. Legacy C++ has no per-source object override and is not inferred here. Basename duplicates do not imply collisions; original PCG CUDA object directories differ. Historical complete compilation and observer neutrality are not inferred.')
        write_new(REPORT/'BASE_OBJECT_PATH_AUDIT.json',output)
        print(json.dumps({k:v for k,v in output.items() if k in ('unique_resolved_object_paths','all_objects_present','all_objects_in_link')}))
        return
    a,b=cache(LEGACY/'CMakeCache.txt'),cache(OBSERVER/'CMakeCache.txt')
    output=dict(schema='report.baseline.identity.audit.v1',
        source_records=[record(p) for p in (LEGACY/'CMakeCache.txt',LEGACY/'gipc.vcxproj',
                                          OBSERVER/'CMakeCache.txt',OBSERVER/'frozen_base/gipc.vcxproj')],
        selected_flags_original=a,selected_flags_observer=b,
        selected_flags_equal={k:a[k]==b[k] for k in a},
        original_units=units(LEGACY/'gipc.vcxproj'),observed_units=units(OBSERVER/'frozen_base/gipc.vcxproj'),
        observation_neutrality_certified=False,
        scope='Current cached flags/source inventories only. Observer identity.json proves its fresh commands and linked unique objects. Retained original hash is frozen; its historical partial-build logs do not prove a fresh matching build. Material/work/trajectory differences remain separately reported, without attributing them to a compiler or export operation.')
    write_new(REPORT/'BASE_IDENTITY_AUDIT.json',output)
    print(json.dumps({k:v for k,v in output.items() if k in ('selected_flags_equal','original_units','observed_units')}))

if __name__=='__main__':main()
