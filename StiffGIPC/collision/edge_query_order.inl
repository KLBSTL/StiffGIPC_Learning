// Included once in GIPC.cu after the original host query wrapper.
// All probe buffers/events are private. Input trees/geometry are const to kernels;
// no ordinary/swept storage is selected, rebuilt or otherwise changed here.
#include <chrono>
#include <set>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cmath>
#include <vector>
#include <functional>
#include <type_traits>

namespace
{
template<bool PerFace>
void edge_order_launch(bool leaf,const lbvh_f& f,const lbvh_e& e,
                       const double3* positions,int* any,int* per_face,double dHat)
{
    const unsigned int block=default_threads; // unchanged production block size
    const int count=static_cast<int>(f.face_number);
    const int blocks=(count+block-1)/block;
    if(e.edge_number==1)
    {
        if(leaf)_edgeTriIntersectionQueryOrdered<true,PerFace,true><<<blocks,block>>>(
            e._btype,positions,e._edges,f._faces,e._bvs.data(),e._nodes.data(),any,dHat,count,f._nodes.data(),per_face);
        else _edgeTriIntersectionQueryOrdered<false,PerFace,true><<<blocks,block>>>(
            e._btype,positions,e._edges,f._faces,e._bvs.data(),e._nodes.data(),any,dHat,count,f._nodes.data(),per_face);
    }
    else if(leaf)
        _edgeTriIntersectionQueryOrdered<true,PerFace,false><<<blocks,block>>>(
            e._btype,positions,e._edges,f._faces,e._bvs.data(),e._nodes.data(),any,dHat,count,f._nodes.data(),per_face);
    else if constexpr(PerFace)
        _edgeTriIntersectionQueryOrdered<false,true,false><<<blocks,block>>>(
            e._btype,positions,e._edges,f._faces,e._bvs.data(),e._nodes.data(),any,dHat,count,f._nodes.data(),per_face);
    else
        _edgeTriIntersectionQuery<<<blocks,block>>>(
            e._btype,positions,e._edges,f._faces,e._bvs.data(),e._nodes.data(),any,dHat,count);
    CUDA_SAFE_CALL(cudaGetLastError());
}

void edge_order_require_trees(const lbvh_f& f,const lbvh_e& e)
{
    if(f.face_number>static_cast<unsigned>(std::numeric_limits<int>::max())
       || e.edge_number>static_cast<unsigned>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Edge query order count overflow");
    // The mapping is the current ordinary tree's original-face permutation.
    // It may not be borrowed from a swept cache or replaced by another sort.
    if(f.swept_storage_active || e.swept_storage_active
       || f.discrete_state.mode!=gipc::DiscreteBVHTreeMode::discrete
       || e.discrete_state.mode!=gipc::DiscreteBVHTreeMode::discrete
       || f._nodes.size()!=2*size_t(f.face_number)-1
       || e._nodes.size()!=2*size_t(e.edge_number)-1
       || e._bvs.size()!=2*size_t(e.edge_number)-1
       || !f._faces || !e._edges || !e._btype)
        throw std::runtime_error("Edge query order requires proven ordinary tree storage");
}

struct EdgeProbeEvents
{
    cudaEvent_t start=nullptr,end=nullptr;
    EdgeProbeEvents()
    {
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&end));
    }
    ~EdgeProbeEvents(){if(end)cudaEventDestroy(end);if(start)cudaEventDestroy(start);}
};

void edge_order_probe(const lbvh_f& f,const lbvh_e& e,const double3* positions,
                      double dHat,int frame,const gipc::EdgeQueryOrderOptions& options)
{
    static thread_local std::set<int> attempted;
    static thread_local bool output_started=false;
    if(!options.selected(frame) || !attempted.insert(frame).second)return;
    if(!output_started && std::filesystem::exists(options.probe_file))
        throw std::runtime_error("Edge probe output already exists; no overwrite/retry");
    const auto started=std::chrono::steady_clock::now();
    gipc::Json result={{"schema","gipc.edge_order_probe.v1"},{"frame",frame},
        {"first_query_only",true},{"production_order",options.leaf?"leaf":"raw"},
        {"face_count",f.face_number},{"edge_count",e.edge_number},
        {"passed",false},{"performance_certified",false},
        {"inputs_read_only",true},{"production_scratch_untouched",true},
        {"probe_scope","same_geometry_same_ordinary_tree_private_outputs"},
        {"kernel_timing_includes_memset_or_host_readback",false}};
    double readback_ms=0;
    auto copy=[&](void* dst,const void* src,size_t bytes){
        const auto t=std::chrono::steady_clock::now();
        if(bytes)CUDA_SAFE_CALL(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost));
        readback_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    };
    try
    {
        if(!f.face_number || !e.edge_number)
        {
            result["empty_query"]=true;result["raw_any"]=false;result["leaf_any"]=false;
            result["compared_faces"]=0;result["timed_pairs"]=0;result["passed"]=true;
            result["speed_evidence"]=false;
        }
        else
        {
            edge_order_require_trees(f,e);
            std::vector<Node> leaves(f.face_number);
            copy(leaves.data(),f._nodes.data()+f.face_number-1,leaves.size()*sizeof(Node));
            std::vector<uint32_t> ids;ids.reserve(leaves.size());
            for(const auto& node:leaves)ids.push_back(node.element_idx);
            std::sort(ids.begin(),ids.end());
            for(size_t i=0;i<ids.size();++i)
                if(ids[i]!=i)throw std::runtime_error("Face leaf IDs are not a complete permutation");
            result["leaf_permutation_valid"]=true;
            cudatool::DeviceBuffer<int> raw_any,leaf_any,raw_faces,leaf_faces;
            raw_any.resize(1);leaf_any.resize(1);
            raw_faces.resize(f.face_number);leaf_faces.resize(f.face_number);
            auto reset=[](int* p,size_t n,int value){CUDA_SAFE_CALL(cudaMemset(p,value,n*sizeof(int)));};
            auto run=[&](bool leaf,int* any,int* per_face,bool diagnostic){
                reset(any,1,0);
                if(diagnostic){
                    reset(per_face,f.face_number,0xff); // poison: every mapped face must overwrite
                    edge_order_launch<true>(leaf,f,e,positions,any,per_face,dHat);
                }
                else edge_order_launch<false>(leaf,f,e,positions,any,nullptr,dHat);
            };
            int truth=0,raw_hit=0,leaf_hit=0;
            run(false,raw_any.data(),nullptr,false);copy(&truth,raw_any.data(),sizeof(int));
            run(false,raw_any.data(),raw_faces.data(),true);
            run(true,leaf_any.data(),leaf_faces.data(),true);
            std::vector<int> raw(f.face_number),leaf(f.face_number);
            copy(&raw_hit,raw_any.data(),sizeof(int));copy(&leaf_hit,leaf_any.data(),sizeof(int));
            copy(raw.data(),raw_faces.data(),raw.size()*sizeof(int));
            copy(leaf.data(),leaf_faces.data(),leaf.size()*sizeof(int));
            size_t mismatches=0,invalid=0,hits=0;
            for(size_t i=0;i<raw.size();++i){
                mismatches+=raw[i]!=leaf[i];invalid+=(raw[i]!=0 && raw[i]!=1)||(leaf[i]!=0 && leaf[i]!=1);
                hits+=raw[i]==1;
            }
            result["raw_any"]=truth<0;result["raw_diagnostic_any"]=raw_hit<0;result["leaf_any"]=leaf_hit<0;
            result["raw_per_face_hits"]=raw;result["leaf_per_face_hits"]=leaf;
            result["compared_faces"]=raw.size();result["per_face_hit_count"]=hits;
            result["per_face_mismatches"]=mismatches;result["invalid_or_unwritten_faces"]=invalid;
            if(mismatches || invalid || raw_hit!=truth || leaf_hit!=truth || (hits>0)!=(truth<0))
                throw std::runtime_error("Raw/leaf edge query output mismatch");
            // Warm both production variants twice; private diagnostic writes are
            // absent from all timed launches. No geometry/control state advances.
            for(int i=0;i<2;++i){
                run(false,raw_any.data(),nullptr,false);run(true,leaf_any.data(),nullptr,false);
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            EdgeProbeEvents events;
            auto timed=[&](bool leaf,int* any){
                reset(any,1,0);
                CUDA_SAFE_CALL(cudaEventRecord(events.start));
                edge_order_launch<false>(leaf,f,e,positions,any,nullptr,dHat);
                CUDA_SAFE_CALL(cudaEventRecord(events.end));
                CUDA_SAFE_CALL(cudaEventSynchronize(events.end));
                float ms=0;CUDA_SAFE_CALL(cudaEventElapsedTime(&ms,events.start,events.end));
                int hit=0;copy(&hit,any,sizeof(int));
                if(hit!=truth || !std::isfinite(ms) || ms<0)
                    throw std::runtime_error("Timed edge query differs or has invalid event time");
                return ms;
            };
            result["pairs"]=gipc::Json::array();
            for(int i=0;i<7;++i){
                float raw_ms,leaf_ms;
                if(i%2==0){raw_ms=timed(false,raw_any.data());leaf_ms=timed(true,leaf_any.data());}
                else{leaf_ms=timed(true,leaf_any.data());raw_ms=timed(false,raw_any.data());}
                result["pairs"].push_back({{"index",i},{"first",i%2==0?"raw":"leaf"},
                    {"raw_kernel_ms",raw_ms},{"leaf_kernel_ms",leaf_ms}});
            }
            result["timed_pairs"]=7;result["warmups_per_mode"]=2;
            result["speed_evidence"]=e.edge_number>1 && f.face_number>1;
            result["passed"]=true;
        }
    }
    catch(const std::exception& error){result["error"]=error.what();}
    result["host_readback_ms"]=readback_ms;
    result["diagnostic_host_ms"]=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-started).count();
    std::ofstream file(options.probe_file,std::ios::app);
    file<<result.dump()<<'\n';file.close();
    if(!file)throw std::runtime_error("Cannot write edge-order probe evidence");
    output_started=true;
    if(!result["passed"].get<bool>())throw std::runtime_error("Edge-order probe failed");
}

bool edge_order_query(const lbvh_f& f,const lbvh_e& e,const double3* positions,
                      int* production_any,double dHat,int frame,
                      const gipc::EdgeQueryOrderOptions& options)
{
    // Both orders explicitly define zero primitives as no intersection. The old
    // >=2-edge path is unchanged; the old one-edge traversal was undefined.
    if(!f.face_number || !e.edge_number){
        edge_order_probe(f,e,positions,dHat,frame,options);return false;
    }
    edge_order_require_trees(f,e);
    edge_order_probe(f,e,positions,dHat,frame,options);
    CUDA_SAFE_CALL(cudaMemset(production_any,0,sizeof(int)));
    edge_order_launch<false>(options.leaf,f,e,positions,production_any,nullptr,dHat);
    int hit=0;CUDA_SAFE_CALL(cudaMemcpy(&hit,production_any,sizeof(int),cudaMemcpyDeviceToHost));
    return hit<0;
}
} // namespace

// Finite correctness fixture. Synthetic balanced trees are test inputs only.
namespace
{
__global__ void edge_fixture_predicates(const double3* x,const uint3* faces,
                                       const uint2* edges,const int* type,int* result)
{
    const auto f=faces[0];const auto e=edges[0];AABB a,b;
    const uint32_t fv[3]={f.x,f.y,f.z},ev[2]={e.x,e.y};
    for(int i=0;i<3;++i){const auto v=fv[i];a.combines(x[v].x,x[v].y,x[v].z);}
    for(int i=0;i<2;++i){const auto v=ev[i];b.combines(x[v].x,x[v].y,x[v].z);}
    const bool shared=f.x==e.x||f.x==e.y||f.y==e.x||f.y==e.y||f.z==e.x||f.z==e.y;
    const bool fixed=type[f.x]>=2&&type[f.y]>=2&&type[f.z]>=2&&type[e.x]>=2&&type[e.y]>=2;
    *result=(_overlap(a,b,0)?1:0)|(shared?2:0)|(fixed?4:0)
        |(segTriIntersect(x[e.x],x[e.y],x[f.x],x[f.y],x[f.z])?8:0);
}
template<class Element>
void edge_fixture_tree(lbvh& tree,const std::vector<Element>& elements,const std::vector<double3>& x)
{
    const uint32_t n=static_cast<uint32_t>(elements.size());
    if(!n){tree._nodes.resize(0);tree._bvs.resize(0);return;}
    std::vector<Node> nodes(2*size_t(n)-1);std::vector<AABB> boxes(nodes.size());
    uint32_t next=0;
    std::function<uint32_t(uint32_t,uint32_t)> build=[&](uint32_t begin,uint32_t end){
        if(end-begin==1){
            const uint32_t id=n-1+begin,original=n-1-begin;
            nodes[id]={0xffffffffu,0xffffffffu,0xffffffffu,original};
            const auto e=elements[original];
            auto add=[&](uint32_t v){boxes[id].combines(x[v].x,x[v].y,x[v].z);};
            add(e.x);add(e.y);if constexpr(std::is_same_v<Element,uint3>)add(e.z);
            return id;
        }
        const uint32_t id=next++,mid=begin+(end-begin)/2;
        const auto left=build(begin,mid),right=build(mid,end);
        nodes[id]={0xffffffffu,left,right,0xffffffffu};
        nodes[left].parent_idx=id;nodes[right].parent_idx=id;
        for(const auto child:{left,right}){
            const auto b=boxes[child];boxes[id].combines(b.lower.x,b.lower.y,b.lower.z);
            boxes[id].combines(b.upper.x,b.upper.y,b.upper.z);
        }
        return id;
    };
    if(build(0,n)!=0)throw std::runtime_error("Fixture root differs");
    tree._nodes.copy_from(nodes);tree._bvs.copy_from(boxes);
    tree.discrete_state.mode=gipc::DiscreteBVHTreeMode::discrete;tree.swept_storage_active=false;
}
gipc::Json edge_fixture_case(const char* name,int nf,int ne,bool crossing,bool shared,
                             int boundary,bool partial,bool real_construct=false)
{
    std::vector<double3> x={make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0),
        make_double3(.2,.2,-1),make_double3(.2,.2,1),make_double3(3,0,0),
        make_double3(4,0,0),make_double3(3,1,0),make_double3(10,10,-1),make_double3(10,10,1)};
    if(!crossing){x[3]=make_double3(2,2,-1);x[4]=make_double3(2,2,1);}
    // Make shared-vertex rejection non-vacuous: strict AABB overlap and the
    // unfiltered segment/triangle predicate both succeed at the shared vertex.
    if(shared){x[1]=make_double3(1,0,1);x[2]=make_double3(0,1,1);}
    std::vector<int> types(x.size(),boundary),body(x.size(),-1);if(partial)types[3]=1;
    std::vector<uint3> faces;for(int i=0;i<nf;++i)faces.push_back(i%2?make_uint3(5,6,7):make_uint3(0,1,2));
    std::vector<uint2> edges;for(int i=0;i<ne;++i)edges.push_back(i==0?make_uint2(shared?0:3,4):make_uint2(8,9));
    std::vector<uint32_t> surface(x.size());std::iota(surface.begin(),surface.end(),0);
    cudatool::DeviceBuffer<double3> dx(x);
    cudatool::DeviceBuffer<int> dt(types),db(body),ra,la,rf,lf;
    cudatool::DeviceBuffer<uint3> df(faces);cudatool::DeviceBuffer<uint2> de(edges);
    cudatool::DeviceBuffer<uint32_t> ds(surface);
    ra.resize(1);la.resize(1);rf.resize(nf);lf.resize(nf);
    lbvh_f f;lbvh_e e;
    f.init(db.data(),dt.data(),dx.data(),df.data(),ds.data(),nf,static_cast<int>(x.size()));
    e.init(db.data(),dt.data(),dx.data(),dx.data(),de.data(),ne,static_cast<int>(x.size()));
    if(real_construct){f.ConstructRebuild();e.ConstructRebuild();}
    else{edge_fixture_tree(f,faces,x);edge_fixture_tree(e,edges,x);}
    gipc::EdgeQueryOrderOptions ro,lo;lo.leaf=true;
    const bool expected=nf>0 && ne>0 && crossing && !shared && (boundary<2 || partial);
    const bool raw=edge_order_query(f,e,dx.data(),ra.data(),1e-6,0,ro);
    const bool leaf=edge_order_query(f,e,dx.data(),la.data(),1e-6,0,lo);
    bool correct=true,poison=true,permutation=true,ordinary=true,guard=true,predicates=true;
    int predicate_mask=-1;
    if(nf && ne){
        cudatool::DeviceBuffer<int> primitive;primitive.resize(1);
        edge_fixture_predicates<<<1,1>>>(dx.data(),df.data(),de.data(),dt.data(),primitive.data());
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaMemcpy(&predicate_mask,primitive.data(),sizeof(int),cudaMemcpyDeviceToHost));
        const int expected_mask=(crossing?9:0)|(shared?2:0)|((boundary>=2&&!partial)?4:0);
        predicates=predicate_mask==expected_mask;
        ordinary=f.discrete_state.mode==gipc::DiscreteBVHTreeMode::discrete && e.discrete_state.mode==gipc::DiscreteBVHTreeMode::discrete;
        std::vector<Node> mapping(nf);CUDA_SAFE_CALL(cudaMemcpy(mapping.data(),f._nodes.data()+nf-1,nf*sizeof(Node),cudaMemcpyDeviceToHost));
        std::vector<uint32_t> ids;for(const auto n:mapping)ids.push_back(n.element_idx);
        std::sort(ids.begin(),ids.end());for(int i=0;i<nf;++i)permutation&=ids[i]==static_cast<uint32_t>(i);
        CUDA_SAFE_CALL(cudaMemset(ra.data(),0,sizeof(int)));CUDA_SAFE_CALL(cudaMemset(la.data(),0,sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(rf.data(),0xff,nf*sizeof(int)));CUDA_SAFE_CALL(cudaMemset(lf.data(),0xff,nf*sizeof(int)));
        edge_order_launch<true>(false,f,e,dx.data(),ra.data(),rf.data(),1e-6);
        edge_order_launch<true>(true,f,e,dx.data(),la.data(),lf.data(),1e-6);
        std::vector<int> a(nf),b(nf);int ah=0,bh=0;
        CUDA_SAFE_CALL(cudaMemcpy(a.data(),rf.data(),nf*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(b.data(),lf.data(),nf*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&ah,ra.data(),sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&bh,la.data(),sizeof(int),cudaMemcpyDeviceToHost));
        correct=(ah<0)==expected && (bh<0)==expected;
        for(int i=0;i<nf;++i){
            const int expected_face=expected && i%2==0;
            correct&=a[i]==expected_face && b[i]==expected_face;
            poison&=(a[i]==0||a[i]==1)&&(b[i]==0||b[i]==1);
        }
        f.swept_storage_active=true;bool refused=false;
        try{edge_order_query(f,e,dx.data(),la.data(),1e-6,0,lo);}
        catch(const std::runtime_error&){refused=true;}
        f.swept_storage_active=false;guard=refused;
    }
    return {{"name",name},{"faces",nf},{"edges",ne},{"expected_any",expected},{"raw_any",raw},{"leaf_any",leaf},
        {"per_face_expected_and_equal",correct},{"poison_cleared",poison},{"leaf_permutation",permutation},
        {"ordinary_mode",ordinary},{"reject_exposed_swept_storage",guard},{"real_D0_ConstructRebuild",real_construct},
        {"primitive_predicate_mask",predicate_mask},{"primitive_predicates_expected",predicates},
        {"performance_certified",false},{"passed",raw==expected && leaf==expected && correct && poison && permutation && ordinary && guard && predicates}};
}
} // namespace
namespace gipc
{
int edge_query_order_fixture(const char* output)
{
    Json report={{"schema","gipc.edge_order_fixture.v1"},{"passed",false},{"performance_certified",false},{"cases",Json::array()}};
    try{
        if(!output || !*output)throw std::runtime_error("Edge fixture output required");
        if(discrete_bvh_config().enabled)throw std::runtime_error("Edge fixture requires GIPC_DISCRETE_BVH_REFIT=0");
        auto add=[&](const char* name,int nf,int ne,bool cross,bool shared,int fixed,bool partial,bool build=false){
            report["cases"].push_back(edge_fixture_case(name,nf,ne,cross,shared,fixed,partial,build));
        };
        add("empty_both",0,0,false,false,0,false);add("empty_faces",0,3,false,false,0,false);
        add("empty_edges",3,0,false,false,0,false);add("one_edge_hit",1,1,true,false,0,false);
        add("one_edge_miss",1,1,false,false,0,false);add("shared_vertex",3,3,true,true,0,false);
        add("all_fixed_2",3,3,true,false,2,false);add("all_fixed_3",3,3,true,false,3,false);
        add("partial_fixed_1",3,3,true,false,3,true);add("negative_boundary",3,3,true,false,-1,false);
        add("tail257_mixed_hit_miss",257,3,true,false,0,false);
        add("D0_real_ordinary_build",3,3,true,false,0,false,true);
        bool passed=true;for(const auto& row:report["cases"])passed&=row["passed"].get<bool>();
        report["passed"]=passed;report["expected_case_count"]=12;
        report["scope"]="Analytic hit/miss and production raw/leaf booleans; no material or speed certification";
    }catch(const std::exception& error){report["error"]=error.what();}
    if(!output || !*output)return 2;
    std::ofstream file(output);file<<report.dump(2)<<'\n';file.close();
    return file && report["passed"].get<bool>()?0:2;
}
} // namespace gipc
