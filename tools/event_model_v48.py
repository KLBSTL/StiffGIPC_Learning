"""Independent CPU reconstruction of frozen affine AL and Stable NH1 terms."""
import json
from pathlib import Path
import numpy as np

CONTACT=np.dtype({'names':['kind','ids','grad','anchor','offset','lambda','gamma','slack','penalty_mu','friction_lambda','friction_gamma','release_age','coordinates'],
    'formats':['<i4',('<i4',4),('<f8',(4,3)),('<f8',(4,3)),'<f8','<f8','<f8','<f8','<f8','<f8','<f8','<i4',('<f8',2)],
    'offsets':[0,4,24,120,216,224,232,240,248,256,264,272,288],'itemsize':304})

class EventModel:
    def __init__(self,prefix):
        self.prefix=Path(prefix);self.run=self.prefix.parent.parent
        self.e=json.loads(Path(str(prefix)+'_line_search.json').read_text())
        self.c=np.fromfile(str(prefix)+'_contacts.bin',dtype=CONTACT)
        self.x=self.load('vertices').reshape(-1,3);self.d=self.load('direction').reshape(-1,3)
        self.q=self.load('abd_q').reshape(-1,12);self.dq=self.load('abd_dq').reshape(-1,12)
        self.abd=json.loads((self.run/'trace/metadata.json').read_text())['abd_point_num']
        self.ids=np.fromfile(self.run/'trace/body_ids.bin',dtype='<i4')
        self.fixed=np.fromfile(self.run/'trace/boundary_types.bin',dtype='<i4')==1
        self.n=len(self.x);self.active=np.arange(4)[None,:]<np.where(self.c['kind']==2,1,4)[:,None]
        assert self.e['contact_bytes']==CONTACT.itemsize and len(self.c)==self.e['contacts']
        assert np.isin(self.c['kind'],[0,1,2]).all()
        assert ((self.c['ids'][self.active]>=0)&(self.c['ids'][self.active]<self.n)).all()
        assert not np.any(self.c['penalty_mu']>0),'Per-contact mu requires initial_mu export'
        assert np.all(np.isfinite(self.c['grad'])) and np.all(self.c['gamma']>=0)
        self.cids=np.where(self.active,self.c['ids'],0)
        self.cg=self.c['grad']*self.active[:,:,None]
        self.mu=self.e['mu'];self.weight=self.mu*self.c['gamma']
        self.xbar=np.zeros((self.abd,3))
        for body in np.unique(self.ids[:self.abd]):
            assert 0<=body<len(self.q)
            use=np.flatnonzero(self.ids[:self.abd]==body);q=self.q[body]
            self.xbar[use]=np.linalg.solve(q[3:].reshape(3,3),(self.x[use]-q[:3]).T).T
        top=np.fromfile(self.run/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,top[:3]);assert nv==self.n
        self.tets=top[3+3*nf:].reshape(-1,4);assert len(self.tets)==nt
        self.tets=self.tets[np.all(self.tets>=self.abd,axis=1)]
        initial=np.fromfile(self.run/'trace/state_0000.bin',dtype='<f8').reshape(-1,3)
        rest=self.edges(initial);self.inv=np.linalg.inv(rest);self.vol=np.abs(np.linalg.det(rest))/6
        scene=json.loads((self.run/'output/scene.json').read_text())
        fem=[o for o in scene['objects'] if o['dimension']==3 and o['body_type']=='FEM']
        if len(self.tets):
            assert len(fem)==1,'Only one homogeneous FEM tet material is supported'
            young=fem[0]['tet_young_pa'];poisson=scene['effective_scalar_fields']['poisson_ratio']
            lame_mu=young/(2*(1+poisson));self.u=4*lame_mu/3
            self.v=young*poisson/((1+poisson)*(1-2*poisson))+5*lame_mu/6
        else:self.u=self.v=1.
        self.dt=self.e['dt']

    def load(self,name):return np.fromfile(str(self.prefix)+'_'+name+'.bin',dtype='<f8')
    def edges(self,x):
        t=x[self.tets];return np.stack([t[:,i]-t[:,0] for i in [1,2,3]],axis=-1)
    def constraint(self,x):
        return self.c['offset']+np.einsum('nki,nki->n',self.cg,x[self.cids]-self.c['anchor'])
    def al(self,x,gradient=False):
        value=self.constraint(x)-self.c['slack']
        per=self.c['gamma']*(.5*self.mu*value**2-self.c['lambda']*value)
        if not gradient:return float(per.sum())
        scale=self.c['gamma']*(self.mu*value-self.c['lambda'])
        g=np.zeros_like(x)
        np.add.at(g,self.cids.ravel(),(scale[:,None,None]*self.cg).reshape(-1,3))
        return float(per.sum()),g,per
    def al_curvature(self,d):return float(np.sum(self.weight*np.einsum('nki,nki->n',self.cg,d[self.cids])**2))
    def fem(self,x,gradient=False):
        F=self.edges(x)@self.inv;J=np.linalg.det(F);s=J-1-self.u/self.v
        per=.5*(self.u*(np.sum(F*F,axis=(1,2))-3)+self.v*s*s)*self.vol*self.dt**2
        if not gradient:return float(per.sum())
        cofactor=np.stack([np.cross(F[:,:,1],F[:,:,2]),np.cross(F[:,:,2],F[:,:,0]),np.cross(F[:,:,0],F[:,:,1])],axis=-1)
        P=self.u*F+self.v*s[:,None,None]*cofactor
        local=(P@self.inv.transpose(0,2,1))*(self.vol*self.dt**2)[:,None,None]
        g=np.zeros_like(x)
        np.add.at(g,self.tets[:,0],-local.sum(axis=2))
        for j in range(3):np.add.at(g,self.tets[:,j+1],local[:,:,j])
        return float(per.sum()),g,per
    def lift(self,g):
        result=np.zeros((len(self.q),12))
        for body in np.unique(self.ids[:self.abd]):
            use=np.flatnonzero(self.ids[:self.abd]==body);result[body,:3]=g[use].sum(axis=0)
            result[body,3:]=(g[use].T@self.xbar[use]).ravel()
        gf=g[self.abd:].copy();gf[self.fixed[self.abd:]]=0
        return np.r_[result.ravel(),gf.ravel()]
    def physical(self,generalized):
        offset=12*len(self.q);d=np.zeros_like(self.x);dq=generalized[:offset].reshape(-1,12)
        d[self.abd:]=generalized[offset:].reshape(-1,3)
        for body in np.unique(self.ids[:self.abd]):
            use=np.flatnonzero(self.ids[:self.abd]==body)
            d[use]=dq[body,:3]+self.xbar[use]@dq[body,3:].reshape(3,3).T
        return d
