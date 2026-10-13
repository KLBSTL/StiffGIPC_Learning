"""Actual native keyframes: retained/current combinations and frozen Stiff, scale one."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/paper_suite'),str(ROOT/'tools/bench')]
from run_combination_suite import SCENES,KEYFRAMES,CHECKPOINTS
from render import geometry,positions,draw,export_obj
from config import read
from linux_runner import require,write_new,sha

ARMS=('stiff','old','ours')
LABELS={'stiff':'Base','old':'Frozen combination','ours':'Current combination'}


def bounds(states):
    values=np.concatenate(states)[:,[0,2,1]]
    center=(values.min(axis=0)+values.max(axis=0))/2
    radius=max(float(np.ptp(values,axis=0).max())*.55,.01)
    return center-radius,center+radius


def render(session,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ledger_path=session/'performance_delivery_ledger.json'
    ledger=read(ledger_path if ledger_path.exists() else session/'performance_ledger.json')
    require(ledger['complete'],'Complete selected performance matrix required')
    selected=ledger.get('selected_scenes',list(SCENES))
    require(read(session/'checkpoint_input_gate.json')['passed'],'Exact matched inputs required')
    output.mkdir(parents=True,exist_ok=False)
    evidence=[];figures=[]
    for repeat in (1,2,3):
        data={}
        for scene in selected:
            values={}
            for arm in ARMS:
                folder=session/f'performance_{scene.lower()}_{arm}_r{repeat}'
                require(read(folder/'result.json')['status']=='completed','No failed frame illustrations')
                mesh=geometry(folder);x={f:positions(folder,f,mesh['nv']) for f in KEYFRAMES[scene]}
                values[arm]=dict(folder=folder,mesh=mesh,states=x)
                for frame,state in x.items():
                    obj=output/f'{scene.lower()}_{arm}_r{repeat}_f{frame:04d}.obj';export_obj(obj,mesh,state)
                    evidence.append(dict(scene=scene,arm=arm,repeat=repeat,frame=frame,
                        state=str(folder/f'trace/state_{frame:04d}.bin'),state_sha256=sha(folder/f'trace/state_{frame:04d}.bin'),
                        topology_sha256=sha(folder/'trace/topology.bin'),obj=obj.name,obj_sha256=sha(obj)))
            common=bounds([x for r in values.values() for x in r['states'].values()])
            data[scene]=dict(arms=values,bounds=common)
        primary=selected
        for frame in (25,50,75,100):
            for candidate in ('old','ours'):
                pair_arms=('stiff',candidate);fig=plt.figure(figsize=(3*len(primary),6),facecolor='white')
                for row,arm in enumerate(pair_arms):
                    for col,scene in enumerate(primary):
                        d=data[scene];r=d['arms'][arm];ax=fig.add_subplot(2,len(primary),row*len(primary)+col+1,projection='3d')
                        draw(ax,r['mesh'],r['states'][frame],scene,d['bounds'])
                        if row==0:ax.set_title(f'({chr(97+col)}) {scene}',fontsize=10)
                        ax.text2D(.5,.02,LABELS[arm],transform=ax.transAxes,ha='center',fontsize=9)
                fig.text(.5,.02,f'Actual frame {frame}, repeat {repeat}; dt=.01, rho=1e-4; shared orthographic camera; deformation scale 1',ha='center',fontsize=9)
                fig.subplots_adjust(wspace=0,hspace=0,bottom=.05,top=.95)
                stem=f'scenes_{candidate}_f{frame:04d}_r{repeat}'
                for ext in ('png','pdf'):fig.savefig(output/(stem+'.'+ext),dpi=180);figures.append(stem+'.'+ext)
                plt.close(fig)
        fig=plt.figure(figsize=(3*len(primary),9),facecolor='white')
        for row,arm in enumerate(ARMS):
            for col,scene in enumerate(primary):
                d=data[scene];r=d['arms'][arm];ax=fig.add_subplot(3,len(primary),row*len(primary)+col+1,projection='3d')
                draw(ax,r['mesh'],r['states'][100],scene,d['bounds'])
                if row==0:ax.set_title(scene,fontsize=10)
                ax.text2D(.5,.02,LABELS[arm],transform=ax.transAxes,ha='center',fontsize=9)
        fig.text(.5,.02,f'{len(primary)} scenes, actual frame 100, repeat {repeat}; dt=.01, rho=1e-4; deformation scale 1',ha='center',fontsize=10)
        fig.subplots_adjust(wspace=0,hspace=0,bottom=.05,top=.95)
        stem=f'all_scenes_f0100_r{repeat}'
        for ext in ('png','pdf'):fig.savefig(output/(stem+'.'+ext),dpi=160);figures.append(stem+'.'+ext)
        plt.close(fig)
        for scene,d in data.items():
            frames=KEYFRAMES[scene];fig=plt.figure(figsize=(3*len(frames),9),facecolor='white')
            for row,arm in enumerate(ARMS):
                r=d['arms'][arm]
                for col,frame in enumerate(frames):
                    ax=fig.add_subplot(3,len(frames),row*len(frames)+col+1,projection='3d')
                    draw(ax,r['mesh'],r['states'][frame],scene,d['bounds'])
                    if row==0:ax.set_title(f'{scene}, frame {frame}',fontsize=10)
                    ax.text2D(.5,.02,LABELS[arm],transform=ax.transAxes,ha='center',fontsize=9)
            fig.text(.5,.01,f'Actual native frames, repeat {repeat}; dt=.01, rho=1e-4; scale 1',ha='center',fontsize=9)
            fig.subplots_adjust(wspace=0,hspace=0,bottom=.04,top=.95);stem=f'{scene.lower()}_timeline_r{repeat}'
            for ext in ('png','pdf'):fig.savefig(output/(stem+'.'+ext),dpi=140);figures.append(stem+'.'+ext)
            plt.close(fig)
        fig=plt.figure(figsize=(12,7),facecolor='white')
        for row,(scene,frame) in enumerate(CHECKPOINTS.items()):
            rows=[]
            for arm in ('original','pure','ours'):
                folder=session/f'matched_{scene.lower()}_{arm}_r{repeat}';mesh=geometry(folder);x=positions(folder,1,mesh['nv']);rows.append((mesh,x))
                obj=output/f'matched_{scene.lower()}_{arm}_r{repeat}.obj';export_obj(obj,mesh,x)
                evidence.append(dict(stage='matched',scene=scene,arm=arm,repeat=repeat,physical_frame=frame,
                    state=str(folder/'trace/state_0001.bin'),state_sha256=sha(folder/'trace/state_0001.bin'),
                    topology_sha256=sha(folder/'trace/topology.bin'),obj=obj.name,obj_sha256=sha(obj)))
            common=bounds([x for _,x in rows])
            for col,((mesh,x),label) in enumerate(zip(rows,('Movement only','Cumulative TOI .01','Residual gate, beta=.01'))):
                ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');draw(ax,mesh,x,scene,common)
                ax.set_title(f'{scene}, step {frame}: {label}',fontsize=10)
        fig.text(.5,.02,f'Cold matched inputs, repeat {repeat}; current combination; dt=.01, rho=1e-4; scale 1',ha='center',fontsize=9)
        fig.subplots_adjust(wspace=0,hspace=0,bottom=.05,top=.94);stem=f'matched_checkpoints_r{repeat}'
        for ext in ('png','pdf'):fig.savefig(output/(stem+'.'+ext),dpi=180);figures.append(stem+'.'+ext)
        plt.close(fig)
        print(f'Rendered repeat {repeat}',flush=True)
    write_new(output/'render_manifest.json',dict(complete=True,selected_scenes=selected,excluded_scenes=ledger.get('excluded_scenes',[]),figures=figures,evidence=evidence,
        camera=dict(projection='orthographic',elevation=26,azimuth=-65,display_axis_order=[0,2,1],deformation_scale=1),
        identities=sha(session/'identities.json'),physical_quality_certified=False))
    print(f'Rendered {len(figures)//2} PNG/PDF figures and {len(evidence)} actual OBJ states',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.session.resolve(),a.output.resolve())
