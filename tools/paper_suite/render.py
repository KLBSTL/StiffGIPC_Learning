"""CPU-only orthographic views and lossless OBJ exports of actual trace frames."""
import argparse
from pathlib import Path
import numpy as np
from protocol import SCENES,KEYFRAMES,CHECKPOINTS,read,write,sha,require


def geometry(folder):
    raw=np.fromfile(folder/'trace/topology.bin',dtype='<u4')
    require(raw.size>=3,'Missing topology');nv,nf,nt=map(int,raw[:3])
    require(raw.size==3+3*nf+4*nt and nv>0,'Malformed topology')
    faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    require(not raw[3:].size or raw[3:].max()<nv,'Out-of-range triangle')
    boundary=np.fromfile(folder/'trace/boundary_types.bin',dtype='<i4')
    require(boundary.size==nv,'Missing boundary types')
    volume=np.zeros(nv,bool);volume[tets.ravel()]=True
    parent=np.arange(nv)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    for a,b,c in faces:
        root=find(int(a));parent[find(int(b))]=root;parent[find(int(c))]=root
    components=np.array([find(int(f[0])) for f in faces]);_,ids=np.unique(components,return_inverse=True)
    fixed=(boundary[faces]!=0).all(axis=1)
    return dict(nv=nv,faces=faces,solid=volume[faces].any(axis=1),fixed=fixed,components=ids)


def positions(folder,frame,nv):
    p=folder/f'trace/state_{frame:04d}.bin'
    require(p.is_file() and p.stat().st_size==nv*24,'Missing/partial keyframe: '+str(p))
    x=np.fromfile(p,dtype='<f8').reshape(nv,3);require(np.isfinite(x).all(),'Nonfinite keyframe')
    return x


def colors(mesh,scene,x):
    import matplotlib.colors as mc
    base={'Sphere':'#79bdd2','Bunny':'#d9a4bd','Table':'#94c6b9','Hang':'#b4a3d4','Mixed':'#8fc5cf','Boxes':'#81bbd0'}
    palette=np.array([mc.to_rgb(c) for c in ['#bba3c8','#d6ba79','#9bc1b9','#dba58e','#9cbbd8','#c4c68e','#b7a5c9','#b3d1c1','#e0b690','#8dadd0']])
    c=palette[mesh['components']%len(palette)] if scene=='Stack' else np.tile(mc.to_rgb(base[scene]),(len(mesh['faces']),1))
    obstacle=mesh['solid']|mesh['fixed']
    c[obstacle]=mc.to_rgb('#c7a574' if scene in ('Sphere','Table','Boxes') else '#b8a4ce')
    points=x[mesh['faces']][:,:,[0,2,1]]
    n=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    light=np.array([-.5,-.2,1.]);light/=np.linalg.norm(light)
    shade=.62+.38*np.abs(n@light)
    return np.clip(c*shade[:,None],0,1)


def draw(ax,mesh,x,scene,bounds):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(x[mesh['faces']][:,:,[0,2,1]],facecolors=colors(mesh,scene,x),
        edgecolors='none',linewidths=0,rasterized=True,zsort='average'))
    for setter,lo,hi in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),bounds[0],bounds[1]):setter(lo,hi)
    ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=26,azim=-65);ax.set_axis_off()


def export_obj(path,mesh,x):
    with path.open('x',encoding='ascii') as stream:
        stream.write('# Actual unscaled world coordinates; topology from native trace\n')
        for v in x:stream.write('v '+' '.join(format(float(z),'.17g') for z in v)+'\n')
        for face in mesh['faces']:stream.write('f '+' '.join(str(int(v)+1) for v in face)+'\n')


def matched_views(session,output,repeat,evidence,diagnostic=False):
    import matplotlib.pyplot as plt
    gate=session/'checkpoint_input_gate.json'
    if not gate.exists() or not read(gate)['passed']:return []
    rows=[];arms=('original','pure','ours')
    for scene,step in CHECKPOINTS.items():
        values=[]
        for arm in arms:
            folder=session/f'matched_{scene.lower()}_{arm}_r{repeat}'
            status=folder/'final_status.json'
            if not status.exists():status=folder/'result.json'
            require(status.exists() and read(status)['status']=='completed','Missing completed matched frame')
            mesh=geometry(folder);x=positions(folder,1,mesh['nv']);values.append((mesh,x))
            target=output/f'{scene.lower()}_{arm}_r{repeat}_physical_f{step:04d}.obj'
            export_obj(target,mesh,x)
            evidence.append(dict(scene=scene,arm=arm,frame=step,trace_frame=1,scope='cold matched checkpoint',
                state_sha256=sha(folder/'trace/state_0001.bin'),topology_sha256=sha(folder/'trace/topology.bin'),
                obj=target.name,obj_sha256=sha(target)))
        xyz=np.concatenate([x for _,x in values])[:,[0,2,1]]
        center=(xyz.min(axis=0)+xyz.max(axis=0))/2;radius=max(float(np.ptp(xyz,axis=0).max())*.55,.01)
        rows.append((scene,step,values,(center-radius,center+radius)))
    fig=plt.figure(figsize=(10,7),facecolor='white')
    for row,(scene,step,values,bounds) in enumerate(rows):
        for col,((mesh,x),label) in enumerate(zip(values,('Original (Graph)','Pure TOI','Residual gate (Ours)'))):
            ax=fig.add_subplot(len(rows),3,row*3+col+1,projection='3d');draw(ax,mesh,x,scene,bounds)
            ax.set_title(f'{scene}, step {step}: {label}',fontsize=10)
    note=f'Cold matched checkpoint, repeat {repeat}; actual states, deformation scale 1; quality not certified'
    if diagnostic:note='LOCAL DIAGNOSTIC: '+note
    fig.text(.5,.02,note,ha='center',fontsize=8);fig.subplots_adjust(wspace=0,hspace=0,bottom=.05,top=.95)
    stem=f'matched_checkpoints_r{repeat}'
    fig.savefig(output/(stem+'.png'),dpi=200);fig.savefig(output/(stem+'.pdf'));plt.close(fig)
    return [stem+'.png',stem+'.pdf']


def render(session,output,stage='performance',frames=None,repeat=1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=False)
    evidence=[];data={};selected=list(SCENES)
    for s in selected:
        available={}
        for arm in ('stiff','ours'):
            folder=session/f'{stage}_{s.lower()}_{arm}_r{repeat}'
            if not (folder/'result.json').exists():continue
            status=read(folder/'final_status.json') if (folder/'final_status.json').exists() else read(folder/'result.json')
            if status['status']!='completed':continue
            mesh=geometry(folder);wanted=frames if frames is not None else KEYFRAMES[s]
            values={f:positions(folder,f,mesh['nv']) for f in wanted}
            available[arm]=(mesh,values,folder)
            for f,x in values.items():
                path=output/f'{s.lower()}_{arm}_r{repeat}_f{f:04d}.obj'
                export_obj(path,mesh,x)
                evidence.append(dict(scene=s,arm=arm,frame=f,state_sha256=sha(folder/f'trace/state_{f:04d}.bin'),
                    topology_sha256=sha(folder/'trace/topology.bin'),obj=path.name,obj_sha256=sha(path)))
        if len(available)==2:
            all_x=np.concatenate([v for _,values,_ in available.values() for v in values.values()])[:,[0,2,1]]
            lo=all_x.min(axis=0);hi=all_x.max(axis=0);center=(lo+hi)/2;radius=max(float((hi-lo).max())*.55,.01)
            data[s]=(available,(center-radius,center+radius))
    common=frames if frames is not None else [25,50,75,100]
    primary=[s for s in list(SCENES)[:5] if s in data]
    for f in common:
        if len(primary)!=5:continue
        fig=plt.figure(figsize=(15,6),facecolor='white')
        for row,arm in enumerate(('stiff','ours')):
            for col,s in enumerate(primary):
                mesh,values,_=data[s][0][arm];ax=fig.add_subplot(2,5,row*5+col+1,projection='3d')
                draw(ax,mesh,values[f],s,data[s][1])
                if row==0:ax.set_title(f'({chr(97+col)}) {s}',fontsize=10)
                ax.text2D(.5,.02,'Base' if arm=='stiff' else 'Ours',transform=ax.transAxes,ha='center',fontsize=9)
        note=f'Actual native frame {f}, repeat {repeat}; shared camera and bounds per scene'
        if stage!='performance':note=f'LOCAL {stage.upper()} PREVIEW: {f} frames, not performance data'
        fig.text(.5,.02,note,ha='center',fontsize=9);fig.subplots_adjust(wspace=0,hspace=0,bottom=.05,top=.95)
        fig.savefig(output/f'five_scenes_f{f:04d}_r{repeat}.png',dpi=200)
        fig.savefig(output/f'five_scenes_f{f:04d}_r{repeat}.pdf');plt.close(fig)
    for s,(arms,bounds) in data.items():
        wanted=frames if frames is not None else KEYFRAMES[s]
        fig=plt.figure(figsize=(3*len(wanted),6),facecolor='white')
        for row,arm in enumerate(('stiff','ours')):
            mesh,values,_=arms[arm]
            for col,f in enumerate(wanted):
                ax=fig.add_subplot(2,len(wanted),row*len(wanted)+col+1,projection='3d');draw(ax,mesh,values[f],s,bounds)
                if row==0:ax.set_title(f'{s}, frame {f}',fontsize=10)
                ax.text2D(.5,.01,'Base' if arm=='stiff' else 'Ours',transform=ax.transAxes,ha='center',fontsize=9)
        fig.subplots_adjust(wspace=0,hspace=0);fig.savefig(output/f'{s.lower()}_timeline_r{repeat}.png',dpi=160);plt.close(fig)
    matched=matched_views(session,output,repeat,evidence,diagnostic=stage!='performance')
    write(output/'render_manifest.json',dict(stage=stage,repeat=repeat,scenes=list(data),evidence=evidence,matched_figures=matched,
        camera=dict(projection='orthographic',elevation=26,azimuth=-65,display_axis_order=[0,2,1],deformation_scale=1),
        physical_quality_certified=False,missing_paired_scenes=[s for s in SCENES if s not in data]))
    print(f'Rendered {len(data)} paired scenes from actual states; {len(evidence)} OBJ exports')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('performance','smoke'),default='performance');p.add_argument('--frames',nargs='+',type=int)
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=1);a=p.parse_args();render(a.session.resolve(),a.output.resolve(),a.stage,a.frames,a.repeat)
