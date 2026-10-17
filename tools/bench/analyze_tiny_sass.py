"""Static compiled evidence only; never interpret sites as dynamic transactions."""
import collections
import re
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_tiny_prefetch import RAW,BUILD,read
from local_identity import record,verify
from linux_runner import require,write_new

def main():
    require(not (RAW/'sass_analysis.json').exists(),'No implicit overwrite')
    tool=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuobjdump.exe')
    identity=read(BUILD/'identity.json');verify([identity['exe']])
    symbol_command=[str(tool),'--dump-elf-symbols',identity['exe']['path']]
    symbols=subprocess.run(symbol_command,capture_output=True,text=True,check=True,timeout=30).stdout
    require(symbols==(RAW/'elf_symbols.txt').read_text(encoding='utf-8'),'Symbol dump differs')
    names=re.findall(r'^STT_FUNC\s+STB_LOCAL\s+STO_ENTRY\s+(\S*balanced_tiny_solve\S*)$',symbols,re.M)
    require(len(names)==2 and len(set(names))==2,'Expected exact two actual symbols')
    command=[str(tool),'--dump-sass','--function',','.join(names),identity['exe']['path']]
    result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30)
    require(not result.stderr.strip(),'Unexpected disassembly stderr')
    source=RAW/'tiny_sass_verified.txt';require(result.stdout==source.read_text(encoding='utf-8'),'Actual selected SASS differs')
    rows=[]
    for symbol,body in re.findall(r'Function : (\S+)\s*(.*?)(?=Function :|\Z)',result.stdout,re.S):
        require(symbol in names,'Unexpected function');method='prefetch' if 'prefetch' in symbol else 'original'
        counts=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P[0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b',body))
        require(counts and counts['LDG.E.64']>0 and counts['STG.E.64']>0,'No parsed load/store instructions')
        rows.append(dict(method=method,symbol=symbol,total_instruction_sites=sum(counts.values()),opcode_sites=dict(counts)))
    require({r['method'] for r in rows}=={'original','prefetch'},'Missing paired SASS')
    write_new(RAW/'sass_analysis.json',dict(passed=True,scope='Static instruction sites including loops/predicates; not executed instruction counts, memory transactions, cycles or isolated latency',
        source=record(Path(__file__)),tool=record(tool),exe=identity['exe'],symbol_command=symbol_command,command=command,
        artifacts=[record(RAW/'elf_symbols.txt'),record(source)],rows=rows))
    print([dict(method=r['method'],total_sites=r['total_instruction_sites'],LDG=r['opcode_sites'].get('LDG.E.64',0),
        LDS=r['opcode_sites'].get('LDS.64',0),STS=r['opcode_sites'].get('STS.64',0),STG=r['opcode_sites'].get('STG.E.64',0),DFMA=r['opcode_sites'].get('DFMA',0)) for r in rows])

if __name__=='__main__':main()
