"""Publication-style tables from measured CSVs, without changing raw outputs."""
import argparse,csv,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def number(row,key,digits=3):return f'{float(row[key]):.{digits}f}'
def plot(output,name,headers,rows,widths,groups=None,caption='',group_key=None):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11})
    height=.31*len(rows)+1.6+(bool(groups)*.4)
    fig,ax=plt.subplots(figsize=(sum(widths),height));ax.set_xlim(0,sum(widths));ax.set_ylim(-.8,len(rows)+2.5)
    ax.axis('off');edges=[0]
    for w in widths:edges.append(edges[-1]+w)
    centers=[(a+b)/2 for a,b in zip(edges,edges[1:])]
    y=len(rows)+1.4
    if groups:
        for start,end,label in groups:
            ax.text((edges[start]+edges[end])/2,y+.9,label,ha='center',va='center')
            ax.plot([edges[start]+.08,edges[end]-.08],[y+.55,y+.55],color='black',lw=.7)
    ax.plot([0,edges[-1]],[y+(1.35 if groups else .7)]*2,color='black',lw=1)
    for x,h in zip(centers,headers):ax.text(x,y,h,ha='center',va='center')
    ax.plot([0,edges[-1]],[y-.45]*2,color='black',lw=.8)
    for i,row in enumerate(rows):
        current=y-1-i
        if group_key is not None and i and row[group_key]!=rows[i-1][group_key]:
            ax.plot([0,edges[-1]],[current+.48]*2,color='black',lw=.6)
        for x,v in zip(centers,row):ax.text(x,current,v,ha='center',va='center')
    ax.plot([0,edges[-1]],[y-len(rows)-.5]*2,color='black',lw=1)
    fig.text(.5,.04,caption,ha='center',va='bottom',fontsize=8)
    fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.14)
    fig.savefig(output/(name+'.png'),dpi=200);fig.savefig(output/(name+'.pdf'));plt.close(fig)

def make(source,output):
    output.mkdir(parents=True,exist_ok=False)
    for name in ('phase_table','graph_table','checkpoint_table'):
        shutil.copyfile(source/(name+'.csv'),output/(name+'.csv'))
        if name!='checkpoint_table':shutil.copyfile(source/(name+'.tex'),output/(name+'.tex'))
    phase=load(source/'phase_table.csv')
    rows=[[r['scene'],'Base' if r['method']=='stiff' else 'Ours',
           *[number(r,k) for k in ('Hess','LSolver','CCD','LineS','Misc','TimeTot')],
           number(r,'P_bar',2),number(r,'N_bar',2),str(int(float(r['P_tot']))),number(r,'S')]
           for r in phase]
    plot(output,'phase_table',['Scene','Method','Hess','LSolver','CCD','LineS','Misc','TimeTot',r'$\overline{P}$',r'$\overline{N}$',r'$P_{tot}$','S'],
         rows,[.9,.8,.75,.85,.75,.75,.75,.9,.8,.8,.9,.75],group_key=0,
         caption='AutoDL RTX 4090; 100 physical frames/run; median of 3 rounds. Phase times: seconds/100 frames.\n'
                 'Base: frozen StiffGIPC. Ours: combined execution + compensated TOI + residual gate. S: median paired Base/Ours.')
    graph=load(source/'graph_table.csv')
    rows=[[r['scene'],*[number(r,k,1) for k in ('Base_ms','Graph_ms')],number(r,'T_B_over_T_G'),
           *[number(r,k,2) for k in ('Base_N','Graph_N','Base_P','Graph_P')]] for r in graph]
    plot(output,'graph_table',['Scene','Base','Graph',r'$T_B/T_G$','Base','Graph','Base','Graph'],
         rows,[1.2,1,1,1,1,1,1,1],groups=[(1,4,r'$\overline{t}$ (ms/step)'),(4,6,r'$\overline{N}$'),(6,8,r'$\overline{P}$')],
         caption='Same active binary and outer rule; host vs graph execution only. 100 frames x 3 rounds.\n'
                 'Base in this table means active host execution, distinct from frozen Base in the main table.')
    matched=load(source/'checkpoint_table.csv')
    names={'original':'Original (Graph)','pure':'Pure TOI','ours':'Residual gate (Ours)'}
    rows=[[r['state'],names[r['method']],str(int(float(r['N']))),number(r,'P',1),number(r,'t',4),
           number(r,'e_RMS_mm',6),number(r,'e_max_mm',4)] for r in matched]
    plot(output,'checkpoint_table',['State','Outer rule','N','P','t (s)',r'$e_x^{RMS}$ (mm)',r'$e_x^{max}$ (mm)'],
         rows,[1.65,2.5,.5,.8,.8,1.6,1.6],group_key=0,
         caption='All rows cold-reconstructed; exact initial checkpoint input gate passed. Medians over 3 candidate rounds.\n'
                 'Errors: worst of 9 pairings against 3 movement-only, PCG rho=1e-6 references; RMS includes all vertices.\n'
                 'Hot/cold endpoint 1e-8 continuity test failed; physical-quality equivalence is not certified.')
    latex=[r'\begin{tabular}{llrrrrr}',r'\hline',
           r'State & Outer rule & $N$ & $P$ & $t$ (s) & $e_x^{\mathrm{RMS}}$ (mm) & $e_x^{\max}$ (mm) \\',r'\hline']
    for i,row in enumerate(rows):
        if i and row[0]!=rows[i-1][0]:latex.append(r'\hline')
        latex.append(' & '.join(row)+r' \\')
    latex.extend([r'\hline',r'\end{tabular}'])
    (output/'checkpoint_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf8')
    print('Created 3 measured CSV/TeX/PNG/PDF tables')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();make(a.source,a.output)
