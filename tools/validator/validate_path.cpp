// Independent CPU broad phase + Tight-Inclusion narrow phase. This program
// reads exported accepted paths; it does not call the simulator's BVH or ACCD.
#include <tight_inclusion/inclusion_ccd.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
using V=Eigen::Vector3d;
using J=nlohmann::json;
struct Box
{
    V lo=V::Constant(1e300),hi=V::Constant(-1e300);
    void add(const V& p){lo=lo.cwiseMin(p);hi=hi.cwiseMax(p);}
    void add(const Box& b){add(b.lo);add(b.hi);}
    bool overlaps(const Box& b)const
    {return (lo.array()<=b.hi.array()+1e-12).all() && (b.lo.array()<=hi.array()+1e-12).all();}
};
struct Node {Box box;int left=-1,right=-1,primitive=-1;};
struct Tree
{
    std::vector<Node> nodes;std::vector<int> ids;const std::vector<Box>& boxes;
    Tree(const std::vector<Box>& b):boxes(b){ids.resize(b.size());std::iota(ids.begin(),ids.end(),0);if(!b.empty())build(0,b.size());}
    int build(int begin,int end)
    {
        int index=nodes.size();nodes.push_back({});Box b;
        for(int i=begin;i<end;++i)b.add(boxes[ids[i]]);
        nodes[index].box=b;
        if(end-begin==1){nodes[index].primitive=ids[begin];return index;}
        int axis=0;(b.hi-b.lo).maxCoeff(&axis);int middle=(begin+end)/2;
        std::nth_element(ids.begin()+begin,ids.begin()+middle,ids.begin()+end,[&](int a,int c){return boxes[a].lo[axis]+boxes[a].hi[axis]<boxes[c].lo[axis]+boxes[c].hi[axis];});
        int l=build(begin,middle),r=build(middle,end);nodes[index].left=l;nodes[index].right=r;return index;
    }
    template<class F>void query(const Box& b,F visit)const
    {
        if(nodes.empty())return;std::vector<int> stack{0};
        while(!stack.empty())
        {
            int i=stack.back();stack.pop_back();const auto& n=nodes[i];if(!b.overlaps(n.box))continue;
            if(n.primitive>=0)visit(n.primitive);else {stack.push_back(n.left);stack.push_back(n.right);}
        }
    }
};
template<class T>std::vector<T> load(const std::filesystem::path& p,size_t n)
{
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Missing "+p.string());
    std::vector<T> out(n);f.read(reinterpret_cast<char*>(out.data()),n*sizeof(T));
    if(!f)throw std::runtime_error("Truncated "+p.string());return out;
}
int main(int argc,char** argv)
try
{
    if(argc==2 && std::string(argv[1])=="--self-test")
    {
        std::array<Scalar,3> error{-1,-1,-1};double toi,tol;
        V a(0,0,0),b(1,0,0),c(0,1,0),p(.25,.25,1),q(.25,.25,-1);
        bool crossing=inclusion_ccd::vertexFaceCCD_double(p,a,b,c,q,a,b,c,error,0,toi,1e-9,1,-1,tol);
        double crossing_toi=toi;
        bool separated=inclusion_ccd::vertexFaceCCD_double(p,a,b,c,p+V(0,0,1),a,b,c,error,0,toi,1e-9,1,-1,tol);
        V e0(-1,0,0),e1(1,0,0),f0(0,-1,1),f1(0,1,1);
        bool ee=inclusion_ccd::edgeEdgeCCD_double(e0,e1,f0,f1,e0,e1,f0-V(0,0,2),f1-V(0,0,2),error,0,toi,1e-9,1,-1,tol);
        J result={{"vf_crossing_detected",crossing},{"vf_crossing_toi",crossing_toi},
                  {"separated_rejected",!separated},{"ee_crossing_detected",ee}};
        std::cout<<result.dump()<<"\n";
        return crossing && !separated && ee && std::abs(crossing_toi-.5)<1e-6?0:1;
    }
    if(argc<3)throw std::runtime_error("Usage: validate_path TRACE_DIR REPORT_JSON [substeps] [--stable-nh1]");
    std::filesystem::path root=argv[1],output=argv[2];
    std::ifstream tf(root/"topology.bin",std::ios::binary);uint32_t counts[3];tf.read(reinterpret_cast<char*>(counts),sizeof(counts));
    if(!tf)throw std::runtime_error("Invalid topology");
    using Face=std::array<uint32_t,3>;using Tet=std::array<uint32_t,4>;using Edge=std::array<uint32_t,2>;
    std::vector<Face> faces(counts[1]);std::vector<Tet> tets(counts[2]);
    tf.read(reinterpret_cast<char*>(faces.data()),faces.size()*sizeof(Face));
    tf.read(reinterpret_cast<char*>(tets.data()),tets.size()*sizeof(Tet));
    std::set<Edge> unique;for(auto f:faces)for(int i=0;i<3;++i){Edge e{f[i],f[(i+1)%3]};if(e[0]>e[1])std::swap(e[0],e[1]);unique.insert(e);}
    std::vector<Edge> edges(unique.begin(),unique.end());
    auto body=load<int>(root/"body_ids.bin",counts[0]);auto fixed=load<int>(root/"boundary_types.bin",counts[0]);
    J meta;std::ifstream mf(root/"metadata.json");mf>>meta;int abd=meta.at("abd_point_num");
    std::vector<std::filesystem::path> paths;
    bool substeps=false,stable_nh1=false;
    for(int i=3;i<argc;++i)
    {
        std::string option=argv[i];
        if(option=="substeps")substeps=true;
        else if(option=="--stable-nh1")stable_nh1=true;
        else throw std::runtime_error("Unknown validator option: "+option);
    }
    auto state_dir=substeps?root/"substeps":root;
    for(auto p:std::filesystem::directory_iterator(state_dir))
        if(p.path().filename().string().rfind(substeps?"safe_":"state_",0)==0)paths.push_back(p.path());
    std::sort(paths.begin(),paths.end());if(paths.size()<2)throw std::runtime_error("Need two accepted states");
    J report={{"validator","independent CPU BVH + Tight-Inclusion"},{"scope",substeps?"accepted substeps":"frame chords"},
        {"vertices",counts[0]},{"faces",faces.size()},{"edges",edges.size()},{"paths_checked",0},
        {"vf_queries",0},{"ee_queries",0},{"conservative_collision_flags",0},{"finite",true},{"tet_inversions",0},
        {"ground_min_gap",1e300},{"ccd_tolerance",1e-9}};
    auto initial=load<V>(paths[0],counts[0]);std::vector<double> rest_det(tets.size());
    auto determinant=[](const std::vector<V>& x,Tet t){return (x[t[1]]-x[t[0]]).dot((x[t[2]]-x[t[0]]).cross(x[t[3]]-x[t[0]]));};
    for(size_t i=0;i<tets.size();++i)rest_det[i]=determinant(initial,tets[i]);
    uint64_t vf_queries=0,ee_queries=0,flags=0,inversions=0,abd_inversions=0;double ground=1e300,fixed_ground=1e300;
    std::vector<V> start=initial;
    for(size_t frame=1;frame<paths.size();++frame)
    {
        auto end=load<V>(paths[frame],counts[0]);
        for(size_t v=0;v<end.size();++v)
        {
            if(!end[v].allFinite())throw std::runtime_error("Nonfinite vertex");
            // The native ground model excludes fixed geometry (e.g. table
            // legs crossing the plane). Validate every simulated vertex.
            if(fixed[v]==1)fixed_ground=std::min(fixed_ground,end[v].y()+1);
            else ground=std::min(ground,end[v].y()+1);
        }
        // Minimize the cubic determinant over the entire linear segment.
        for(size_t i=0;i<tets.size();++i)
        {
            auto t=tets[i];V a=start[t[1]]-start[t[0]],b=start[t[2]]-start[t[0]],c=start[t[3]]-start[t[0]];
            V da=end[t[1]]-end[t[0]]-a,db=end[t[2]]-end[t[0]]-b,dc=end[t[3]]-end[t[0]]-c;
            double s=rest_det[i]>0?1:-1;
            double d0=s*a.dot(b.cross(c));
            double d1=s*(da.dot(b.cross(c))+a.dot(db.cross(c)+b.cross(dc)));
            double d2=s*(da.dot(db.cross(c)+b.cross(dc))+a.dot(db.cross(dc)));
            double d3=s*da.dot(db.cross(dc));
            auto eval=[&](double x){return ((d3*x+d2)*x+d1)*x+d0;};
            double minimum=std::min(eval(0),eval(1));
            if(std::abs(d3)>1e-30)
            {
                double disc=4*d2*d2-12*d3*d1;
                if(disc>=0)for(double r:{(-2*d2-std::sqrt(disc))/(6*d3),(-2*d2+std::sqrt(disc))/(6*d3)})
                    if(r>0&&r<1)minimum=std::min(minimum,eval(r));
            }
            else if(std::abs(d2)>1e-30)
            {double r=-d1/(2*d2);if(r>0&&r<1)minimum=std::min(minimum,eval(r));}
            if(minimum<=0)
            {
                ++inversions;
                if(t[0]<abd && t[1]<abd && t[2]<abd && t[3]<abd)++abd_inversions;
            }
        }
        std::vector<Box> vertex_boxes(counts[0]),face_boxes(faces.size()),edge_boxes(edges.size());
        for(size_t i=0;i<counts[0];++i){vertex_boxes[i].add(start[i]);vertex_boxes[i].add(end[i]);}
        for(size_t i=0;i<faces.size();++i)for(int v:faces[i])face_boxes[i].add(vertex_boxes[v]);
        for(size_t i=0;i<edges.size();++i)for(int v:edges[i])edge_boxes[i].add(vertex_boxes[v]);
        Tree face_tree(face_boxes),edge_tree(edge_boxes);
        auto ignored=[&](const std::array<int,4>& ids){
            bool same_abd=true,all_fixed=true;
            for(int v:ids){same_abd=same_abd && v<abd && body[v]==body[ids[0]];all_fixed=all_fixed && fixed[v]==1;}
            return same_abd || all_fixed;
        };
        std::array<Scalar,3> error{-1,-1,-1};
        for(int v=0;v<static_cast<int>(counts[0]);++v)
            face_tree.query(vertex_boxes[v],[&](int i){
                auto f=faces[i];if(v==f[0]||v==f[1]||v==f[2])return;
                if(ignored({v,int(f[0]),int(f[1]),int(f[2])}))return;
                double toi,tolerance; ++vf_queries;
                if(inclusion_ccd::vertexFaceCCD_double(start[v],start[f[0]],start[f[1]],start[f[2]],
                    end[v],end[f[0]],end[f[1]],end[f[2]],error,0,toi,1e-9,1,-1,tolerance))++flags;
            });
        for(size_t i=0;i<edges.size();++i)
            edge_tree.query(edge_boxes[i],[&](int j){
                if(j<=i)return;auto a=edges[i],b=edges[j];
                if(a[0]==b[0]||a[0]==b[1]||a[1]==b[0]||a[1]==b[1])return;
                if(ignored({int(a[0]),int(a[1]),int(b[0]),int(b[1])}))return;
                double toi,tolerance;++ee_queries;
                if(inclusion_ccd::edgeEdgeCCD_double(start[a[0]],start[a[1]],start[b[0]],start[b[1]],
                    end[a[0]],end[a[1]],end[b[0]],end[b[1]],error,0,toi,1e-9,1,-1,tolerance))++flags;
            });
        start.swap(end);report["paths_checked"]=frame;
    }
    report["vf_queries"]=vf_queries;report["ee_queries"]=ee_queries;report["conservative_collision_flags"]=flags;
    report["tet_inversions"]=inversions;report["ground_min_gap"]=ground;
    report["fixed_ground_min_gap_diagnostic"]=fixed_ground;
    report["ground_scope"]="nonfixed vertices; fixed geometry follows native scene semantics";
    // The approved quality plan requires positive J for materials that need
    // it. Stable NH1 is inversion tolerant; CCD and ground gates still apply.
    report["declared_material_policy"]=stable_nh1?"Stable NH1, inversion tolerant":"positive volume required";
    report["positive_volume_required"]=!stable_nh1;
    report["abd_tet_inversions"]=abd_inversions;
    report["fem_tet_inversions"]=inversions-abd_inversions;
    report["abd_orientation_required"]=true;
    report["passed"]=flags==0 && abd_inversions==0 && (stable_nh1 || inversions==0) && ground>=-1e-10;
    std::ofstream out(output);out<<report.dump(2);std::cout<<report.dump()<<"\n";
    return report["passed"].get<bool>()?0:1;
}
catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
