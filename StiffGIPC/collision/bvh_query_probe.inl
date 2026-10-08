// Included only by mlbvh.cu after the query wrappers and finite tree helpers.
#include <filesystem>
#include <set>
#include <numeric>

namespace
{
struct BvhQueryProbeOwner
{
    cudatool::DeviceBuffer<int4> pairs,ccd;
    cudatool::DeviceBuffer<int> matrix;
    cudatool::DeviceBuffer<uint32_t> counts;
    cudatool::DeviceBuffer<gipc::BvhQueryRecord> vf,ee;
    cudatool::DeviceBuffer<uint32_t> vf_trace;
    uint32_t vf_trace_stride=0;
    void prepare(const lbvh_f& f,const lbvh_e& e,uint32_t capacity,bool swept)
    {
        if(capacity>uint32_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("BVH probe pair count exceeds signed range");
        ccd.resize_discard(capacity);counts.resize_discard(5);
        if(!swept){pairs.resize_discard(capacity);matrix.resize_discard(capacity);}
        vf.resize_discard(f.face_number?f.vert_number:0);
        ee.resize_discard(e.edge_number>1?e.edge_number:0);
        CUDA_SAFE_CALL(cudaMemset(counts.data(),0,5*sizeof(uint32_t)));
        if(vf.size())CUDA_SAFE_CALL(cudaMemset(vf.data(),0xff,vf.size()*sizeof(gipc::BvhQueryRecord)));
        if(ee.size())CUDA_SAFE_CALL(cudaMemset(ee.data(),0xff,ee.size()*sizeof(gipc::BvhQueryRecord)));
    }
    uint64_t bytes() const
    {return uint64_t(pairs.capacity()+ccd.capacity())*sizeof(int4)+uint64_t(matrix.capacity())*sizeof(int)
        +uint64_t(counts.capacity())*sizeof(uint32_t)+uint64_t(vf.capacity()+ee.capacity())*sizeof(gipc::BvhQueryRecord);}
};
template<class T> std::vector<T> bvh_probe_copy(const T* input,size_t count)
{
    if(count&&!input)throw std::runtime_error("BVH probe missing production range");
    std::vector<T> output(count);
    if(count)CUDA_SAFE_CALL(cudaMemcpy(output.data(),input,count*sizeof(T),cudaMemcpyDeviceToHost));
    return output;
}
std::vector<gipc::BvhTypedPair> bvh_probe_typed(const int4* pairs,const int4* ccd,
    const int* matrix,const std::array<uint32_t,5>& counts)
{
    const auto p=bvh_probe_copy(pairs,counts[0]),q=bvh_probe_copy(ccd,counts[0]);
    const auto ranks=bvh_probe_copy(matrix,counts[0]);
    std::vector<gipc::BvhTypedPair> tuples;tuples.reserve(p.size());
    for(size_t i=0;i<p.size();++i){
        const int type=(p[i].x>=0||p[i].y<0)?4:(p[i].z<0?2:(p[i].w<0?3:4));
        tuples.push_back({type,p[i].x,p[i].y,p[i].z,p[i].w,q[i].x,q[i].y,q[i].z,q[i].w});
    }
    return gipc::bvh_query_typed_multiset(std::move(tuples),ranks,counts);
}
std::vector<std::array<int,4>> bvh_probe_ccd(const int4* pairs,uint32_t count)
{
    const auto input=bvh_probe_copy(pairs,count);std::vector<std::array<int,4>> output;
    output.reserve(input.size());for(auto p:input)output.push_back({p.x,p.y,p.z,p.w});
    std::sort(output.begin(),output.end());return output;
}
void bvh_probe_launch(const lbvh_f& f,const lbvh_e& e,const double3* direction,
    double alpha,double dHat,uint32_t capacity,bool swept,bool instrumented,BvhQueryProbeOwner& out)
{
    constexpr unsigned block=256; // same as the existing VF/EE wrappers
    if(f.face_number&&f.vert_number){
        const unsigned blocks=(f.vert_number+block-1)/block;
        if(swept){
            if(instrumented)_selfQuery_vf_ccd_probe<<<blocks,block>>>(f._bodyId,f._btype,f._vertexes,
                direction,alpha,f._faces,f._surfVerts,f._bvs.data(),f._nodes.data(),out.ccd.data(),
                out.counts.data(),dHat,capacity,f.vert_number,out.vf.data(),out.vf_trace.data(),out.vf_trace_stride);
            else fullCCDselfQuery_vf(f._bodyId,f._btype,f._vertexes,direction,alpha,f._faces,f._surfVerts,
                f._bvs.data(),f._nodes.data(),out.ccd.data(),out.counts.data(),dHat,capacity,f.vert_number);
        }else{
            if(instrumented)_selfQuery_vf_probe<<<blocks,block>>>(f._bodyId,f._btype,f._vertexes,
                f._faces,f._surfVerts,f._bvs.data(),f._nodes.data(),out.pairs.data(),out.ccd.data(),
                out.counts.data(),out.matrix.data(),dHat,capacity,f.vert_number,out.vf.data(),out.vf_trace.data(),out.vf_trace_stride);
            else selfQuery_vf(f._bodyId,f._btype,f._vertexes,f._faces,f._surfVerts,f._bvs.data(),
                f._nodes.data(),out.pairs.data(),out.ccd.data(),out.counts.data(),out.matrix.data(),dHat,capacity,f.vert_number);
        }
    }
    if(e.edge_number>1){
        const unsigned blocks=(e.edge_number+block-1)/block;
        if(swept){
            if(instrumented)_selfQuery_ee_ccd_probe<<<blocks,block>>>(e._bodyId,e._btype,e._vertexes,
                direction,alpha,e._edges,e._bvs.data(),e._nodes.data(),out.ccd.data(),out.counts.data(),
                dHat,capacity,e.edge_number,out.ee.data());
            else fullCCDselfQuery_ee(e._bodyId,e._btype,e._vertexes,direction,alpha,e._edges,
                e._bvs.data(),e._nodes.data(),out.ccd.data(),out.counts.data(),dHat,capacity,e.edge_number);
        }else{
            if(instrumented)_selfQuery_ee_probe<<<blocks,block>>>(e._bodyId,e._btype,e._vertexes,
                e._rest_vertexes,e._edges,e._bvs.data(),e._nodes.data(),out.pairs.data(),out.ccd.data(),
                out.counts.data(),out.matrix.data(),dHat,capacity,e.edge_number,out.ee.data());
            else selfQuery_ee(e._bodyId,e._btype,e._vertexes,e._rest_vertexes,e._edges,e._bvs.data(),
                e._nodes.data(),out.pairs.data(),out.ccd.data(),out.counts.data(),out.matrix.data(),dHat,capacity,e.edge_number);
        }
    }
    CUDA_SAFE_CALL(cudaGetLastError());
}
gipc::Json bvh_probe_record_json(const std::vector<gipc::BvhQueryRecord>& records,bool swept,const char* kind,bool raw_records)
{
    auto result=gipc::bvh_query_record_summary(records,swept);result["kind"]=kind;
    result["raw_records_included"]=raw_records;
    if(raw_records){
        result["record_columns"]={"launch_thread","original_id","node_aabb_tests","internal_pops","leaf_overlaps","narrow_calls","max_pending_stack"};
        result["records"]=gipc::Json::array();
        for(size_t i=0;i<records.size();++i){const auto& r=records[i];
            result["records"].push_back({i,r.original_id,r.node_aabb_tests,r.internal_pops,r.leaf_overlaps,r.narrow_calls,r.max_stack});}
    }
    std::vector<size_t> order(records.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return records[a].node_aabb_tests>records[b].node_aabb_tests;});
    result["heaviest_queries"]=gipc::Json::array();
    for(size_t j=0;j<std::min(size_t(16),order.size());++j){const size_t i=order[j];
        result["heaviest_queries"].push_back({{"launch_thread",i},{"original_id",records[i].original_id},
            {"node_aabb_tests",records[i].node_aabb_tests},{"narrow_calls",records[i].narrow_calls}});}
    return result;
}
gipc::Json bvh_probe_compare(const lbvh_f& f,const lbvh_e& e,const double3* direction,
    double alpha,double dHat,bool swept,const gipc::BvhDcdProductionView* dcd,
    const gipc::BvhFullCcdProductionView* ccd,bool raw_records=false)
{
    const uint32_t capacity=swept?ccd->count:dcd->counts[0];
    BvhQueryProbeOwner private_output;private_output.prepare(f,e,capacity,swept);
    bvh_probe_launch(f,e,direction,alpha,dHat,capacity,swept,true,private_output);
    std::array<uint32_t,5> actual{};
    CUDA_SAFE_CALL(cudaMemcpy(actual.data(),private_output.counts.data(),sizeof(actual),cudaMemcpyDeviceToHost));
    if(swept){
        if(actual[0]!=capacity)throw std::runtime_error("BVH probe FullCCD count mismatch");
        if(bvh_probe_ccd(ccd->ccd,capacity)!=bvh_probe_ccd(private_output.ccd.data(),capacity))
            throw std::runtime_error("BVH probe FullCCD tuple multiset mismatch");
    }else{
        std::array<uint32_t,5> expected{};std::copy(dcd->counts,dcd->counts+5,expected.begin());
        if(actual!=expected)throw std::runtime_error("BVH probe DCD typed count mismatch");
        if(bvh_probe_typed(dcd->pairs,dcd->ccd,dcd->matrix_indices,expected)
           !=bvh_probe_typed(private_output.pairs.data(),private_output.ccd.data(),private_output.matrix.data(),actual))
            throw std::runtime_error("BVH probe DCD typed tuple multiset mismatch");
    }
    std::vector<gipc::BvhQueryRecord> vf,ee;private_output.vf.copy_to(vf);private_output.ee.copy_to(ee);
    const auto surface=bvh_probe_copy(f._surfVerts,vf.size());
    for(size_t i=0;i<vf.size();++i)
        if(vf[i].original_id!=surface[i])throw std::runtime_error("BVH probe VF original query ID mismatch");
    if(!ee.empty()){
        const auto leaves=bvh_probe_copy(e._nodes.data()+e.edge_number-1,ee.size());
        for(size_t i=0;i<ee.size();++i)
            if(ee[i].original_id!=leaves[i].element_idx)throw std::runtime_error("BVH probe EE original query ID mismatch");
    }
    return {{"production_pairs",capacity},{"private_counts",actual},{"typed_multiset_equal",true},
        {"atomic_array_order_compared",false},{"private_device_bytes",private_output.bytes()},
        {"workload_readback_bytes",uint64_t(vf.size()+ee.size())*sizeof(gipc::BvhQueryRecord)},
        {"queries",{bvh_probe_record_json(vf,swept,swept?"FullCCD_VF":"DCD_VF",raw_records),
                    bvh_probe_record_json(ee,swept,swept?"FullCCD_EE":"DCD_EE",raw_records)}}};
}
void bvh_probe_sample(const lbvh_f& f,const lbvh_e& e,const double3* direction,
    double alpha,double dHat,bool swept,const gipc::BvhDcdProductionView* dcd,
    const gipc::BvhFullCcdProductionView* ccd,int frame)
{
    const auto& options=gipc::bvh_query_probe_options();
    if(!options.selected(frame)||std::strcmp(gipc::cost_trace_state().sample_kind,"production"))return;
    static thread_local std::set<std::pair<int,bool>> attempted;
    static thread_local bool output_started=false;
    if(!attempted.insert({frame,swept}).second)return;
    if(!output_started&&std::filesystem::exists(options.file))
        throw std::runtime_error("BVH query probe output already exists; no overwrite/retry");
    gipc::CostSampleGuard sample("diagnostic_bvh_query_workload");
    gipc::CostScope scope("diagnostic.bvh_query_workload");
    const auto start=std::chrono::steady_clock::now();
    gipc::Json result={{"schema","gipc.bvh_query_workload.v1"},{"frame",frame},
        {"phase",swept?"FullCCD":"DCD"},{"alpha",swept?gipc::Json(alpha):gipc::Json(nullptr)},
        {"dHat",dHat},{"first_successful_query_only",true},{"production_outputs_untouched",true},
        {"tree_storage_untouched",true},{"performance_certified",false},{"passed",false}};
    gipc::attach_solve_context(result);result["frame"]=frame;
    try{result["sample"]=bvh_probe_compare(f,e,direction,alpha,dHat,swept,dcd,ccd,options.raw_records);result["passed"]=true;}
    catch(const std::exception& error){result["error"]=error.what();}
    result["diagnostic_host_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::ofstream file(options.file,std::ios::app);file<<result.dump()<<'\n';file.close();
    if(!file)throw std::runtime_error("Cannot write BVH workload probe evidence");
    output_started=true;
    if(!result["passed"].get<bool>())throw std::runtime_error("BVH workload probe failed: "+result.value("error",std::string("unknown")));
}
gipc::Json bvh_probe_three_level_fixture(bool swept)
{
    // One eligible face, then shared-vertex, same x-key body and all-fixed
    // rejections. Four leaves form a root / internal / leaf three-level tree.
    std::vector<double3> x={make_double3(.2,.2,.01),make_double3(0,0,0),make_double3(1,0,0),
        make_double3(0,1,0),make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0),
        make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0)};
    std::vector<double3> direction(x.size(),make_double3(0,0,0));
    std::vector<uint3> faces={make_uint3(1,2,3),make_uint3(0,2,3),make_uint3(4,5,6),make_uint3(7,8,9)};
    std::vector<uint2> edges;std::vector<uint32_t> surface={0};
    std::vector<int> body(x.size(),-1),types(x.size(),0);body[0]=7;body[4]=7;
    types[0]=3;types[7]=types[8]=types[9]=3;
    cudatool::DeviceBuffer<double3> dx(x),rest(x),dd(direction);
    cudatool::DeviceBuffer<uint3> df(faces);cudatool::DeviceBuffer<uint32_t> ds(surface);
    cudatool::DeviceBuffer<int> db(body),dt(types);
    lbvh_f f;lbvh_e e;f.face_number=4;f.vert_number=1;e.edge_number=0;e.vert_number=uint32_t(x.size());
    f._vertexes=dx.data();f._bodyId=db.data();f._btype=dt.data();f._faces=df.data();f._surfVerts=ds.data();
    e._vertexes=dx.data();e._rest_vertexes=rest.data();e._bodyId=db.data();e._btype=dt.data();e._edges=nullptr;
    cp_fixture_tree(f,faces,x,direction,swept?1.:0.);cp_fixture_tree(e,edges,x,direction,0);
    std::vector<Node> nodes;std::vector<AABB> boxes;f._nodes.copy_to(nodes);f._bvs.copy_to(boxes);
    if(nodes.size()!=7)throw std::runtime_error("BVH trace fixture must have four leaves");
    AABB query;query.upper=query.lower=x[0];
    std::vector<uint32_t> pending={0},expected;
    while(!pending.empty()){
        const uint32_t node=pending.back();pending.pop_back();
        for(uint32_t child:{nodes[node].left_idx,nodes[node].right_idx}){
            if(!overlap(query,boxes[child],.1))continue;
            if(nodes[child].element_idx!=0xffffffffu)expected.push_back(nodes[child].element_idx);
            else pending.push_back(child);
        }
    }
    BvhQueryProbeOwner reference;reference.prepare(f,e,4,swept);
    bvh_probe_launch(f,e,dd.data(),1,.01,4,swept,false,reference);
    std::array<uint32_t,5> counts{};
    CUDA_SAFE_CALL(cudaMemcpy(counts.data(),reference.counts.data(),sizeof(counts),cudaMemcpyDeviceToHost));
    if(counts[0]!=1)throw std::runtime_error("BVH trace fixture filter truth must leave one pair");
    gipc::BvhDcdProductionView dcd;dcd.pairs=reference.pairs.data();dcd.ccd=reference.ccd.data();dcd.matrix_indices=reference.matrix.data();
    std::copy(counts.begin(),counts.end(),dcd.counts);gipc::BvhFullCcdProductionView ccd{reference.ccd.data(),counts[0]};
    auto result=bvh_probe_compare(f,e,dd.data(),1,.01,swept,&dcd,&ccd);
    BvhQueryProbeOwner traced;traced.prepare(f,e,1,swept);traced.vf_trace_stride=4;
    traced.vf_trace.resize_discard(4);CUDA_SAFE_CALL(cudaMemset(traced.vf_trace.data(),0xff,4*sizeof(uint32_t)));
    bvh_probe_launch(f,e,dd.data(),1,.01,1,swept,true,traced);
    std::vector<uint32_t> actual;std::vector<gipc::BvhQueryRecord> records;
    traced.vf_trace.copy_to(actual);traced.vf.copy_to(records);
    gipc::bvh_query_record_summary(records,swept);
    if(actual!=expected||expected.size()!=4||records.size()!=1||records[0].node_aabb_tests!=6
       ||records[0].internal_pops!=3||records[0].leaf_overlaps!=4||records[0].max_stack!=2
       ||records[0].narrow_calls!=(swept?0u:1u))
        throw std::runtime_error("BVH trace fixture DFS order/workload mismatch");
    result["case"]="three_level_shared_body_fixed_filters";result["phase"]=swept?"FullCCD":"DCD";
    result["expected_leaf_sequence"]=expected;result["actual_leaf_sequence"]=actual;
    result["dfs_leaf_order_equal"]=true;result["expected_eligible_pairs"]=1;
    return result;
}
} // namespace

void gipc::bvh_query_probe_dcd(const lbvh_f& f,const lbvh_e& e,double dHat,
    const BvhDcdProductionView& view,int frame)
{bvh_probe_sample(f,e,nullptr,0,dHat,false,&view,nullptr,frame);}
void gipc::bvh_query_probe_fullccd(const lbvh_f& f,const lbvh_e& e,const double3* direction,
    double alpha,double dHat,const BvhFullCcdProductionView& view,int frame)
{bvh_probe_sample(f,e,direction,alpha,dHat,true,nullptr,&view,frame);}

int gipc::bvh_query_probe_fixture(const char* output)
{
    Json report={{"schema","gipc.bvh_query_probe_fixture.v1"},{"passed",false},
        {"performance_certified",false},{"cases",Json::array()}};
    try{
        for(uint32_t queries:{0u,1u,5u,257u})for(bool fixed:{false,true}){
            std::vector<double3> x={make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0),
                make_double3(.2,.2,.01),make_double3(2,0,0),make_double3(3,0,0),
                make_double3(2.5,-.5,.01),make_double3(2.5,.5,.01)};
            std::vector<double3> direction(x.size(),make_double3(0,0,.002));
            std::vector<uint3> faces=queries?std::vector<uint3>{make_uint3(0,1,2)}:std::vector<uint3>{};
            std::vector<uint2> edges=queries?std::vector<uint2>{make_uint2(4,5),make_uint2(6,7)}:std::vector<uint2>{};
            std::vector<uint32_t> surface(queries,3);std::vector<int> body(x.size(),-1),types(x.size(),fixed?3:0);
            cudatool::DeviceBuffer<double3> dx(x),rest(x),dd(direction);
            cudatool::DeviceBuffer<uint3> df(faces);cudatool::DeviceBuffer<uint2> de(edges);
            cudatool::DeviceBuffer<uint32_t> ds(surface);cudatool::DeviceBuffer<int> db(body),dt(types);
            lbvh_f f;lbvh_e e;f.face_number=uint32_t(faces.size());f.vert_number=queries;
            f._vertexes=dx.data();f._bodyId=db.data();f._btype=dt.data();f._faces=df.data();f._surfVerts=ds.data();
            e.edge_number=uint32_t(edges.size());e.vert_number=uint32_t(x.size());e._vertexes=dx.data();e._rest_vertexes=rest.data();
            e._bodyId=db.data();e._btype=dt.data();e._edges=de.data();
            for(bool swept:{false,true}){
                cp_fixture_tree(f,faces,x,direction,swept?1.:0.);cp_fixture_tree(e,edges,x,direction,swept?1.:0.);
                const uint32_t capacity=queries+4;BvhQueryProbeOwner reference;reference.prepare(f,e,capacity,swept);
                bvh_probe_launch(f,e,dd.data(),1.,.01,capacity,swept,false,reference);
                std::array<uint32_t,5> counts{};CUDA_SAFE_CALL(cudaMemcpy(counts.data(),reference.counts.data(),sizeof(counts),cudaMemcpyDeviceToHost));
                if(counts[0]>capacity)throw std::runtime_error("BVH query fixture unexpected pair overflow");
                BvhDcdProductionView dcd;dcd.pairs=reference.pairs.data();dcd.ccd=reference.ccd.data();dcd.matrix_indices=reference.matrix.data();
                std::copy(counts.begin(),counts.end(),dcd.counts);BvhFullCcdProductionView ccd{reference.ccd.data(),counts[0]};
                auto result=bvh_probe_compare(f,e,dd.data(),1.,.01,swept,&dcd,&ccd);
                result["input_queries"]=queries;result["all_fixed"]=fixed;result["face_count"]=faces.size();
                result["edge_count"]=edges.size();result["phase"]=swept?"FullCCD":"DCD";
                report["cases"].push_back(std::move(result));
            }
        }
        report["cases"].push_back(bvh_probe_three_level_fixture(false));
        report["cases"].push_back(bvh_probe_three_level_fixture(true));
        report["passed"]=true;report["case_count"]=report["cases"].size();
    }catch(const std::exception& error){report["error"]=error.what();}
    if(!output||!*output)return 2;
    std::ofstream file(output);file<<report.dump(2)<<'\n';file.close();
    return file&&report["passed"].get<bool>()?0:2;
}
