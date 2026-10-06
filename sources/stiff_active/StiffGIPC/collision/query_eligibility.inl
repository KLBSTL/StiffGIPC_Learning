// Included once after the legacy raw-query helpers. No geometry, bounds,
// cached tree state, production pair buffers, or original _flags are written.
#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <limits>

namespace
{
template<class Element>
__global__ void eligibility_init_leaves(const Node* nodes,const Element* elements,
    const int* body,const int* btype,int count,gipc::QueryEligibilitySummary* summaries,
    uint32_t* arrivals)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=static_cast<uint32_t>(count))return;
    if(i<static_cast<uint32_t>(count-1))arrivals[i]=0;
    const uint32_t leaf=i+count-1;
    const uint32_t original=nodes[leaf].element_idx;
    const auto element=elements[original];
    bool fixed=btype[element.x]>=2 && btype[element.y]>=2;
    if constexpr(std::is_same_v<Element,uint3>)fixed=fixed && btype[element.z]>=2;
    summaries[leaf]={body[element.x],1,static_cast<uint32_t>(fixed),original};
}

__global__ void eligibility_merge_internal(const Node* nodes,int count,
    gipc::QueryEligibilitySummary* summaries,uint32_t* arrivals,unsigned long long* diagnostics)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=static_cast<uint32_t>(count))return;
    uint32_t parent=nodes[i+count-1].parent_idx;
    while(parent!=0xffffffffu)
    {
        // Every arrival releases the completed child summary. The second
        // child acquires the first child's writes before reading both children.
        // Leaves came from the preceding same-stream initialization kernel.
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> ready(arrivals[parent]);
        const uint32_t old=ready.fetch_add(1,cuda::memory_order_acq_rel);
        if(old==0)return;
        if(old!=1){if(diagnostics)atomicAdd(diagnostics,1ULL);return;}
        const auto left=summaries[nodes[parent].left_idx];
        const auto right=summaries[nodes[parent].right_idx];
        const uint32_t homogeneous=left.homogeneous && right.homogeneous && left.body==right.body;
        summaries[parent]={left.body,homogeneous,left.all_fixed && right.all_fixed,
            left.max_original_id>right.max_original_id?left.max_original_id:right.max_original_id};
        parent=nodes[parent].parent_idx;
    }
}

template<class Element>
const gipc::QueryEligibilitySummary* eligibility_prepare_raw(QueryEligibilityScratch& scratch,
    const Node* nodes,const Element* elements,const int* body,const int* btype,uint32_t count,bool diagnostic)
{
    if(!count)return nullptr;
    if(count>static_cast<uint32_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("BVH eligibility exceeds signed leaf range");
    scratch.summaries.resize_discard(size_t(count)*2-1);
    scratch.arrivals.resize_discard(count-1);
    if(diagnostic)
    {
        scratch.diagnostics.resize_discard(5);
        CUDA_SAFE_CALL(cudaMemsetAsync(scratch.diagnostics.data(),0,5*sizeof(unsigned long long),cudaStreamPerThread));
    }
    const unsigned blocks=(count-1)/256+1;
    eligibility_init_leaves<<<blocks,256>>>(nodes,elements,body,btype,static_cast<int>(count),
        scratch.summaries.data(),scratch.arrivals.data());
    if(count>1)eligibility_merge_internal<<<blocks,256>>>(nodes,static_cast<int>(count),
        scratch.summaries.data(),scratch.arrivals.data(),diagnostic?scratch.diagnostics.data():nullptr);
    return scratch.summaries.data();
}

inline uint32_t eligibility_leaf_count(const lbvh_f& t){return t.face_number;}
inline uint32_t eligibility_leaf_count(const lbvh_e& t){return t.edge_number;}
inline const uint3* eligibility_elements(const lbvh_f& t){return t._faces;}
inline const uint2* eligibility_elements(const lbvh_e& t){return t._edges;}
inline bool eligibility_empty(const lbvh_f& t){return !t.face_number || !t.vert_number;}
inline bool eligibility_empty(const lbvh_e& t){return t.edge_number<=1;}

template<class Tree>
const gipc::QueryEligibilitySummary* eligibility_prepare_query(Tree& tree,int kind)
{
    auto& stats=gipc::query_eligibility_stats()[kind];
    ++stats.query_calls;
    const auto& config=gipc::query_eligibility_config();
    if(!config.enabled)return nullptr;
    ++stats.enabled_queries;
    if(eligibility_empty(tree)){++stats.empty_queries;return nullptr;}
    const uint32_t count=eligibility_leaf_count(tree);
    if(tree._nodes.size()!=size_t(count)*2-1 || !tree._bodyId || !tree._btype || !eligibility_elements(tree))
        throw std::runtime_error("BVH eligibility invalid query storage");
    gipc::CostScope scope("collision.eligibility_prepare");
    const auto* result=eligibility_prepare_raw(tree.query_eligibility_scratch,tree._nodes.data(),
        eligibility_elements(tree),tree._bodyId,tree._btype,count,config.validate);
    ++stats.prepare_calls;stats.summary_nodes+=size_t(count)*2-1;
    stats.scratch_bytes_peak=std::max<uint64_t>(stats.scratch_bytes_peak,tree.query_eligibility_scratch.capacity_bytes());
    return result;
}

inline void eligibility_raw_query(lbvh_f& t,bool swept,double gap,const double3* move,double alpha,
    int4* pairs,int4* ccd,uint32_t* counts,int* indices,uint32_t capacity,
    const gipc::QueryEligibilitySummary* summary,unsigned long long* diagnostic)
{
    if(eligibility_empty(t))return;
    if(swept)fullCCDselfQuery_vf(t._bodyId,t._btype,t._vertexes,move,alpha,t._faces,t._surfVerts,
        t._bvs,t._nodes,ccd,counts,gap,capacity,t.vert_number,summary,diagnostic);
    else selfQuery_vf(t._bodyId,t._btype,t._vertexes,t._faces,t._surfVerts,t._bvs,t._nodes,
        pairs,ccd,counts,indices,gap,capacity,t.vert_number,summary,diagnostic);
}
inline void eligibility_raw_query(lbvh_e& t,bool swept,double gap,const double3* move,double alpha,
    int4* pairs,int4* ccd,uint32_t* counts,int* indices,uint32_t capacity,
    const gipc::QueryEligibilitySummary* summary,unsigned long long* diagnostic)
{
    if(eligibility_empty(t))return;
    if(swept)fullCCDselfQuery_ee(t._bodyId,t._btype,t._vertexes,move,alpha,t._edges,
        t._bvs,t._nodes,ccd,counts,gap,capacity,t.edge_number,summary,diagnostic);
    else selfQuery_ee(t._bodyId,t._btype,t._vertexes,t._rest_vertexes,t._edges,t._bvs,t._nodes,
        pairs,ccd,counts,indices,gap,capacity,t.edge_number,summary,diagnostic);
}

struct EligibilityPairSnapshot
{
    std::array<uint32_t,5> counts{};
    std::vector<std::array<int,9>> pairs;
};
template<class Tree>
EligibilityPairSnapshot eligibility_snapshot(Tree& tree,int kind,double gap,const double3* move,
    double alpha,bool enabled)
{
    auto& stats=gipc::query_eligibility_stats()[kind];
    const bool swept=kind>=2;
    auto* summary=enabled?tree.query_eligibility_scratch.summaries.data():nullptr;
    auto* diagnostic=enabled?tree.query_eligibility_scratch.diagnostics.data():nullptr;
    cudatool::DeviceBuffer<int4> pairs,ccd;
    cudatool::DeviceBuffer<int> indices;
    cudatool::DeviceBuffer<uint32_t> counts(5);
    EligibilityPairSnapshot result;
    uint32_t capacity=0;
    // Both paths start at zero capacity. Their independent overflow/retry loops
    // cannot touch the caller's existing append counts or production capacity.
    for(;;)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(counts.data(),0,5*sizeof(uint32_t),cudaStreamPerThread));
        eligibility_raw_query(tree,swept,gap,move,alpha,pairs.data(),ccd.data(),counts.data(),indices.data(),capacity,summary,diagnostic);
        if(enabled)++stats.new_query_passes;else ++stats.old_query_passes;
        CUDA_SAFE_CALL(cudaMemcpy(result.counts.data(),counts.data(),5*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        if(result.counts[0]<=capacity)break;
        if(result.counts[0]>static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("BVH eligibility validation exceeds signed pair range");
        capacity=result.counts[0];
        ccd.resize_discard(capacity);
        if(!swept){pairs.resize_discard(capacity);indices.resize_discard(capacity);}
        if(enabled){++stats.new_overflow_retries;stats.new_peak_capacity=std::max<uint64_t>(stats.new_peak_capacity,capacity);}
        else{++stats.old_overflow_retries;stats.old_peak_capacity=std::max<uint64_t>(stats.old_peak_capacity,capacity);}
    }
    const auto& c=result.counts;
    ccd.resize(c[0]);std::vector<int4> hc;ccd.copy_to(hc);
    if(swept)
    {
        if(c[1] || c[2] || c[3] || c[4])throw std::runtime_error("BVH eligibility swept count format");
        for(const auto& q:hc)result.pairs.push_back({kind,q.x,q.y,q.z,q.w,0,0,0,0});
    }
    else
    {
        if(uint64_t(c[2])+c[3]+c[4]!=c[0] || c[1])
            throw std::runtime_error("BVH eligibility incomplete typed counts");
        pairs.resize(c[0]);indices.resize(c[0]);
        std::vector<int4> hp;std::vector<int> hi;pairs.copy_to(hp);indices.copy_to(hi);
        std::array<std::vector<int>,5> ranks;
        for(size_t i=0;i<hp.size();++i)
        {
            const auto p=hp[i],q=hc[i];
            const int type=(p.x>=0 || p.y<0)?4:(p.z<0?2:(p.w<0?3:4));
            ranks[type].push_back(hi[i]);
            result.pairs.push_back({type,p.x,p.y,p.z,p.w,q.x,q.y,q.z,q.w});
        }
        for(int type=2;type<=4;++type)
        {
            auto& r=ranks[type];std::sort(r.begin(),r.end());
            if(r.size()!=c[type])throw std::runtime_error("BVH eligibility pair type/count mismatch");
            for(size_t i=0;i<r.size();++i)if(r[i]!=static_cast<int>(i))
                throw std::runtime_error("BVH eligibility MatIndex is not a type-local bijection");
        }
    }
    std::sort(result.pairs.begin(),result.pairs.end()); // retain all duplicates
    return result;
}

template<class Tree>
void eligibility_validate_query(Tree& tree,int kind,double gap,const double3* move=nullptr,double alpha=0)
{
    if(!gipc::query_eligibility_config().validate || eligibility_empty(tree))return;
    auto& stats=gipc::query_eligibility_stats()[kind];
    ++stats.validation_calls;
    const auto start=std::chrono::steady_clock::now();
    gipc::CostSampleGuard sample("diagnostic_query_eligibility");
    gipc::CostScope scope("diagnostic.query_eligibility");
    try
    {
        const auto old=eligibility_snapshot(tree,kind,gap,move,alpha,false);
        const auto candidate=eligibility_snapshot(tree,kind,gap,move,alpha,true);
        if(old.counts!=candidate.counts || old.pairs!=candidate.pairs)
            throw std::runtime_error("BVH eligibility changed the typed collision multiset");
        std::array<unsigned long long,5> diagnostic{};
        CUDA_SAFE_CALL(cudaMemcpy(diagnostic.data(),tree.query_eligibility_scratch.diagnostics.data(),
            sizeof(diagnostic),cudaMemcpyDeviceToHost));
        if(diagnostic[0])throw std::runtime_error("BVH eligibility invalid parent arrival count");
        stats.diagnostic_nodes_tested+=diagnostic[1];stats.diagnostic_pruned_body+=diagnostic[2];
        stats.diagnostic_pruned_fixed+=diagnostic[3];stats.diagnostic_pruned_id+=diagnostic[4];
        stats.pairs_compared+=old.pairs.size();++stats.validation_passed;
    }
    catch(...)
    {
        ++stats.validation_failed;
        stats.diagnostic_host_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        throw;
    }
    stats.diagnostic_host_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
} // namespace

namespace
{
std::vector<Node> eligibility_fixture_nodes(uint32_t count)
{
    if(!count)return {};
    std::vector<Node> nodes(size_t(count)*2-1,Node{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu});
    std::vector<uint32_t> original(count);std::iota(original.begin(),original.end(),0u);
    std::mt19937 random(731u+count);std::shuffle(original.begin(),original.end(),random);
    uint32_t next=0;
    std::function<uint32_t(uint32_t,uint32_t,uint32_t)> build=[&](uint32_t lo,uint32_t hi,uint32_t parent)
    {
        if(hi-lo==1)
        {
            const uint32_t index=count-1+lo;
            nodes[index]={parent,0xffffffffu,0xffffffffu,original[lo]};return index;
        }
        const uint32_t index=next++;
        const uint32_t mid=lo+(hi-lo)/2;
        nodes[index].parent_idx=parent;
        nodes[index].left_idx=build(lo,mid,index);
        nodes[index].right_idx=build(mid,hi,index);
        return index;
    };
    if(build(0,count,0xffffffffu)!=0 || next!=count-1)
        throw std::runtime_error("eligibility fixture invalid generated topology");
    return nodes;
}

template<class Element>
void eligibility_fixture_metadata(nlohmann::json& result,uint32_t count)
{
    constexpr bool face=std::is_same_v<Element,uint3>;
    const uint32_t width=face?3:2;
    auto nodes=eligibility_fixture_nodes(count);
    std::vector<Element> elements(count);
    for(uint32_t i=0;i<count;++i)
    {
        if constexpr(face)elements[i]=make_uint3(width*i,width*i+1,width*i+2);
        else elements[i]=make_uint2(width*i,width*i+1);
    }
    std::vector<int> body(count*width),fixed(count*width);
    cudatool::DeviceBuffer<Node> dn(nodes);
    cudatool::DeviceBuffer<Element> de(elements);
    cudatool::DeviceBuffer<int> db(body),df(fixed);
    QueryEligibilityScratch scratch;
    auto check=[&](const char* label)
    {
        // Poison retained capacity so a missing refresh cannot inherit a pass.
        if(scratch.summaries.size())CUDA_SAFE_CALL(cudaMemsetAsync(scratch.summaries.data(),0xff,
            scratch.summaries.size()*sizeof(gipc::QueryEligibilitySummary),cudaStreamPerThread));
        if(scratch.arrivals.size())CUDA_SAFE_CALL(cudaMemsetAsync(scratch.arrivals.data(),0xff,
            scratch.arrivals.size()*sizeof(uint32_t),cudaStreamPerThread));
        const auto* prepared=eligibility_prepare_raw(scratch,dn.data(),de.data(),db.data(),df.data(),count,true);
        if(!count)
        {
            if(prepared || scratch.capacity_bytes())throw std::runtime_error("empty metadata allocated or returned storage");
            result.push_back({{"primitive",face?"face":"edge"},{"leaves",count},{"case",label},
                {"nodes_checked",0},{"pass",true}});return;
        }
        std::vector<gipc::QueryEligibilitySummary> actual;scratch.summaries.copy_to(actual);
        std::vector<uint32_t> arrivals;scratch.arrivals.copy_to(arrivals);
        std::vector<unsigned long long> diagnostic;scratch.diagnostics.copy_to(diagnostic);
        if(diagnostic[0])throw std::runtime_error("fixture parent publication received a third arrival");
        for(const auto n:arrivals)if(n!=2)throw std::runtime_error("fixture parent did not receive both children");
        // Independent reference: enumerate every node's descendant leaves, then
        // inspect original primitive vertices. No recursive summary merge reuse.
        for(size_t root=0;root<nodes.size();++root)
        {
            std::vector<uint32_t> todo{static_cast<uint32_t>(root)},leaves;
            std::vector<bool> seen(nodes.size(),false);
            while(!todo.empty())
            {
                const auto node=todo.back();todo.pop_back();
                if(node>=nodes.size() || seen[node])throw std::runtime_error("fixture topology cycle or bad child");
                seen[node]=true;
                if(nodes[node].element_idx!=0xffffffffu)
                {
                    if(nodes[node].element_idx>=count)throw std::runtime_error("fixture original ID outside range");
                    leaves.push_back(nodes[node].element_idx);
                }
                else
                {
                    const auto l=nodes[node].left_idx,r=nodes[node].right_idx;
                    if(l>=nodes.size() || r>=nodes.size() || nodes[l].parent_idx!=node || nodes[r].parent_idx!=node)
                        throw std::runtime_error("fixture parent/child mismatch");
                    todo.push_back(r);todo.push_back(l);
                }
            }
            if(leaves.empty())throw std::runtime_error("fixture empty subtree");
            const int key=body[elements[leaves.front()].x];bool homogeneous=true,all_fixed=true;
            uint32_t max_id=0;
            for(const auto id:leaves)
            {
                const auto e=elements[id];homogeneous=homogeneous && body[e.x]==key;
                all_fixed=all_fixed && fixed[e.x]>=2 && fixed[e.y]>=2;
                // Use the type trait directly: NVCC's host lambda lowering can
                // turn a captured local constexpr into a nonconstant member.
                if constexpr(std::is_same_v<Element,uint3>)
                    all_fixed=all_fixed && fixed[e.z]>=2;
                max_id=std::max(max_id,id);
            }
            const auto a=actual[root];
            if(a.body!=key || a.homogeneous!=uint32_t(homogeneous) || a.all_fixed!=uint32_t(all_fixed)
                || a.max_original_id!=max_id)
                throw std::runtime_error("fixture GPU summary differs from descendant reference at node "+std::to_string(root));
        }
        result.push_back({{"primitive",face?"face":"edge"},{"leaves",count},{"case",label},
            {"nodes_checked",nodes.size()},{"internal_arrivals_all_two",true},{"pass",true}});
    };
    if(!count){check("empty_no_allocation");return;}
    for(uint32_t i=0;i<count;++i)
    {
        body[width*i]=7;
        for(uint32_t j=1;j<width;++j)body[width*i+j]=-19-static_cast<int>(j);
        for(uint32_t j=0;j<width;++j)fixed[width*i+j]=2+(j%2);
    }
    db.copy_from(body);df.copy_from(fixed);check("x_key_only_all_fixed_2_3");
    const auto* body_address=db.data();const auto* fixed_address=df.data();const auto* node_address=dn.data();
    constexpr int keys[]={-1,7,std::numeric_limits<int>::min(),19};
    constexpr int flags[]={-3,1,2,3};
    for(uint32_t i=0;i<count;++i)
    {
        body[width*i]=keys[i%4];
        for(uint32_t j=0;j<width;++j)fixed[width*i+j]=flags[(i+j)%4];
    }
    db.copy_from(body);df.copy_from(fixed);
    if(db.data()!=body_address || df.data()!=fixed_address)throw std::runtime_error("fixture failed same-address mutation setup");
    check("same_address_mixed_body_and_btype");
    for(auto& value:body)value=-1;
    for(auto& value:fixed)value=3;
    db.copy_from(body);df.copy_from(fixed);check("same_address_minus_one_all_fixed_3");
    if(count>1)
    {
        for(uint32_t i=0;i<count;++i)nodes[count-1+i].element_idx=(nodes[count-1+i].element_idx+1)%count;
        for(uint32_t i=0;i<count;++i)body[width*i]=keys[i%4];
        db.copy_from(body);dn.copy_from(nodes);
        if(dn.data()!=node_address)throw std::runtime_error("fixture failed same-address node mapping mutation");
        check("same_address_original_id_remap");
    }
}

void eligibility_fixture_one_face(nlohmann::json& results)
{
    // More query points than leaves deliberately detects face/vertex count mixups.
    std::vector<double3> points={make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0),
        make_double3(.25,.25,.1),make_double3(.25,.25,-.1)};
    std::vector<uint3> faces={make_uint3(0,1,2)};
    std::vector<uint32_t> surf={0,1,2,3,4};
    std::vector<int> body={7,8,9,7,-1},fixed(5,0);
    cudatool::DeviceBuffer<double3> dv(points),move(std::vector<double3>(5,make_double3(0,0,0)));
    cudatool::DeviceBuffer<uint3> df(faces);cudatool::DeviceBuffer<uint32_t> ds(surf);
    cudatool::DeviceBuffer<int> db(body),dt(fixed);
    lbvh_f tree;
    tree.face_number=1;tree.vert_number=5;tree._vertexes=dv.data();tree._bodyId=db.data();
    tree._btype=dt.data();tree._faces=df.data();tree._surfVerts=ds.data();
    tree._nodes.copy_from(eligibility_fixture_nodes(1));
    AABB bounds;bounds.lower=make_double3(0,0,0);bounds.upper=make_double3(1,1,0);
    tree._bvs.copy_from(std::vector<AABB>{bounds});
    auto check=[&](const char* label,uint32_t expected)
    {
        eligibility_prepare_raw(tree.query_eligibility_scratch,tree._nodes.data(),tree._faces,
            tree._bodyId,tree._btype,tree.face_number,true);
        for(int kind:{0,2})
        {
            const auto old=eligibility_snapshot(tree,kind,.25,move.data(),.5,false);
            const auto candidate=eligibility_snapshot(tree,kind,.25,move.data(),.5,true);
            if(old.counts!=candidate.counts || old.pairs!=candidate.pairs || old.counts[0]!=expected)
                throw std::runtime_error("one-face/five-query-point multiset fixture failed");
            results.push_back({{"case",label},{"kind",kind==0?"ordinary_vf":"swept_vf"},
                {"leaves",1},{"query_points",5},{"expected_pairs",expected},
                {"typed_multiset_equal",true},{"pass",true}});
        }
    };
    check("heterogeneous_vertex_body_x_key",1);
    const auto* body_address=db.data();const auto* fixed_address=dt.data();
    body[3]=-1;body[4]=7;fixed={2,3,2,1,3};db.copy_from(body);dt.copy_from(fixed);
    if(db.data()!=body_address || dt.data()!=fixed_address)throw std::runtime_error("single-face mutation changed address");
    check("inplace_minus_one_and_btype_one",1);
    fixed[3]=3;dt.copy_from(fixed);check("all_fixed_at_least_two",0);
    fixed[3]=-1;dt.copy_from(fixed);check("inplace_negative_btype_is_not_fixed",1);
    // Edge query with one leaf has no pair. Exercise the shared raw-query guard
    // with a real root leaf so no internal-child dereference may be required.
    cudatool::DeviceBuffer<uint2> edges(std::vector<uint2>{make_uint2(0,1)});
    lbvh_e edge;edge.edge_number=1;edge.vert_number=5;edge._vertexes=dv.data();edge._rest_vertexes=dv.data();
    edge._edges=edges.data();edge._bodyId=db.data();edge._btype=dt.data();
    edge._nodes.copy_from(eligibility_fixture_nodes(1));edge._bvs.copy_from(std::vector<AABB>{bounds});
    eligibility_prepare_raw(edge.query_eligibility_scratch,edge._nodes.data(),edge._edges,
        edge._bodyId,edge._btype,1,true);
    for(int kind:{1,3})
    {
        const auto old=eligibility_snapshot(edge,kind,.25,move.data(),.5,false);
        const auto candidate=eligibility_snapshot(edge,kind,.25,move.data(),.5,true);
        if(old.counts!=candidate.counts || old.counts[0] || !candidate.pairs.empty())
            throw std::runtime_error("one-edge query guard fixture failed");
        results.push_back({{"case","one_edge_no_query"},{"kind",kind==1?"ordinary_ee":"swept_ee"},
            {"expected_pairs",0},{"pass",true}});
    }
}
} // namespace

int gipc::query_eligibility_fixture(const char* output)
{
    nlohmann::json report={{"fixture","bvh_query_eligibility"},{"pass",false},
        {"metadata_cases",nlohmann::json::array()},{"query_cases",nlohmann::json::array()},
        {"comparison","exact integer summaries and complete typed pair multisets"},
        {"production_scene_validation_required",true},{"performance_claim",false}};
    int code=0;
    try
    {
        for(const uint32_t n:{0u,1u,2u,3u,31u,33u,257u})
        {
            eligibility_fixture_metadata<uint3>(report["metadata_cases"],n);
            eligibility_fixture_metadata<uint2>(report["metadata_cases"],n);
        }
        eligibility_fixture_one_face(report["query_cases"]);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        report["pass"]=true;
    }
    catch(const std::exception& e){report["error"]=e.what();code=1;}
    report["stats"]=query_eligibility_stats_json();
    try
    {
        if(!output || !*output)throw std::runtime_error("fixture requires an output JSON path");
        std::ofstream file(output,std::ios::binary);
        if(!file)throw std::runtime_error("cannot open eligibility fixture output");
        file<<report.dump(2)<<'\n';
        if(!file)throw std::runtime_error("cannot write eligibility fixture output");
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    return code;
}
