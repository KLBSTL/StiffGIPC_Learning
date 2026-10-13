"""Depth-tested offscreen OpenGL views of the unchanged exported simulation meshes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import moderngl

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/paper_suite'),str(ROOT/'tools/bench')]
import render_combination_suite as suite
from render import colors,geometry,positions
from linux_runner import require,sha


class DepthRenderer:
    def __init__(self,size=768):
        self.size=size
        self.context=moderngl.create_standalone_context(require=330)
        c=self.context
        self.program=c.program(vertex_shader='''#version 330
            in vec3 in_position;
            in vec3 in_color;
            out vec3 color;
            void main() { gl_Position=vec4(in_position,1.0); color=in_color; }
            ''',fragment_shader='''#version 330
            in vec3 color;
            out vec4 frag_color;
            void main() { frag_color=vec4(color,1.0); }
            ''')
        self.buffer=c.buffer(reserve=24)
        self.vao=c.vertex_array(self.program,[(self.buffer,'3f 3f','in_position','in_color')])
        self.color=c.renderbuffer((size,size),components=3,samples=4)
        self.depth=c.depth_renderbuffer((size,size),samples=4)
        self.fbo=c.framebuffer([self.color],self.depth)
        self.resolved_color=c.renderbuffer((size,size),components=3)
        self.resolved_depth=c.depth_renderbuffer((size,size))
        self.resolved=c.framebuffer([self.resolved_color],self.resolved_depth)
        azimuth=np.deg2rad(-65);elevation=np.deg2rad(26)
        direction=np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
        right=np.cross([0.,0.,1.],direction);right/=np.linalg.norm(right)
        up=np.cross(direction,right)
        self.basis=np.column_stack([right,up,-direction])
        self.info=dict(backend='Offscreen OpenGL 3.3; real depth test; four-sample MSAA',
                       renderer=c.info['GL_RENDERER'],depth_func='LESS',culling=False,blending=False,
                       projection='orthographic',elevation=26,azimuth=-65,display_axis_order=[0,2,1],
                       deformation_scale=1,geometric_smoothing=False,panel_size=[size,size],
                       projection_halfspan_radius_ratio=1.75,shader_input='interleaved 3f position + 3f face RGB; 24-byte stride',
                       numeric_scope='Only rasterizer uploads float32; native double coordinates and all solver data unchanged')

    def raster(self,vertices,rgb):
        c=self.context
        require(vertices.shape==rgb.shape and vertices.shape[1]==3,'Bad raster vertex layout')
        packed=np.ascontiguousarray(np.column_stack([vertices,rgb]),dtype='f4')
        self.buffer.orphan(packed.nbytes);self.buffer.write(packed.tobytes())
        self.fbo.use();c.viewport=(0,0,self.size,self.size)
        c.enable(moderngl.DEPTH_TEST);c.disable(moderngl.CULL_FACE|moderngl.BLEND);c.depth_func='<'
        self.fbo.clear(1,1,1,1,depth=1)
        self.vao.render(mode=moderngl.TRIANGLES,vertices=len(vertices))
        c.copy_framebuffer(self.resolved,self.fbo)
        pixels=self.resolved.read(components=3,alignment=1)
        return np.asarray(Image.frombytes('RGB',(self.size,self.size),pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM))

    def mesh(self,mesh,x,scene,bounds):
        lo,hi=bounds;center=(lo+hi)/2;radius=float((hi-lo).max())/2
        projected=(x[:,[0,2,1]]-center)@self.basis
        projected[:,:2]/=1.75*radius;projected[:,2]/=3*radius
        require(np.isfinite(projected).all() and np.abs(projected).max()<1,'Camera clips actual geometry')
        rgb=np.repeat(colors(mesh,scene,x),3,axis=0)
        return self.raster(projected[mesh['faces']].reshape(-1,3),rgb)

    def draw(self,ax,mesh,x,scene,bounds):
        from matplotlib.image import BboxImage
        from matplotlib.transforms import Bbox,TransformedBbox
        pixels=self.mesh(mesh,x,scene,bounds)
        ax.set_axis_off()
        artist=BboxImage(TransformedBbox(Bbox.unit(),ax.transAxes),interpolation='bilinear',origin='upper',zorder=0)
        artist.set_data(pixels);ax.add_artist(artist)

    def verify_depth(self):
        tri=np.array([[-.6,-.6,0],[.6,-.6,0],[0,.6,0]])
        near=tri+np.array([0,0,-.25]);far=tri+np.array([0,0,.25])
        red=np.tile([1.,0.,0.],(3,1));green=np.tile([0.,1.,0.],(3,1))
        a=self.raster(np.concatenate([near,far]),np.concatenate([red,green]))
        b=self.raster(np.concatenate([far,near]),np.concatenate([green,red]))
        require(np.array_equal(a,b),'Depth output depends on triangle draw order')
        require(np.array_equal(a[self.size//2,self.size//2],[255,0,0]),'Far polygon incorrectly occludes near polygon')
        self.info['depth_order_smoke_passed']=True

    def release(self):
        for resource in (self.vao,self.buffer,self.program,self.fbo,self.resolved,self.color,self.depth,
                         self.resolved_color,self.resolved_depth,self.context):resource.release()


def main(session,output,smoke):
    renderer=DepthRenderer()
    try:
        renderer.verify_depth()
        if smoke:
            require(not output.exists(),'Smoke output already exists');output.mkdir(parents=True)
            for scene in ('Sphere','Bunny','Table'):
                folder=session/f'performance_{scene.lower()}_ours_r1';mesh=geometry(folder)
                xs=[positions(folder,f,mesh['nv']) for f in suite.KEYFRAMES[scene]]
                bounds=suite.bounds(xs)
                for frame in (0,100):
                    x=positions(folder,frame,mesh['nv'])
                    Image.fromarray(renderer.mesh(mesh,x,scene,bounds)).save(output/f'{scene.lower()}_f{frame:04d}.png')
            (output/'renderer_smoke.json').write_text(json.dumps(renderer.info,indent=2)+'\n',encoding='utf8')
        else:
            suite.draw=renderer.draw
            suite.render(session,output)
            path=output/'render_manifest.json';manifest=json.loads(path.read_text(encoding='utf8'))
            manifest['camera'].update(renderer.info)
            manifest['renderer_script']=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
            path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(renderer.info),flush=True)
    finally:
        renderer.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    main(args.session.resolve(),args.output.resolve(),args.smoke)
