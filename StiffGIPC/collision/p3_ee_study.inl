#include <cuda_tools/scoped_cuda_events.h>
namespace
{
struct P3Output
{
    cudatool::DeviceBuffer<int4> pairs,ccd;
    cudatool::DeviceBuffer<int> matrix;
    cudatool::DeviceBuffer<uint32_t> counts;
    void prepare(uint32_t capacity)
    {pairs.resize(capacity);ccd.resize(capacity);matrix.resize(capacity);counts.resize(5);}
    void reset(){CUDA_SAFE_CALL(cudaMemsetAsync(counts.data(),0,5*sizeof(uint32_t),cudaStreamPerThread));}
    std::array<uint32_t,5> read()const
    {
        std::array<uint32_t,5> out{};
        CUDA_SAFE_CALL(cudaMemcpy(out.data(),counts.data(),sizeof(out),cudaMemcpyDeviceToHost));return out;
    }
    void fused(const lbvh_e& e,double dHat,uint32_t cap)
    {p3_fused(e,dHat,pairs.data(),ccd.data(),counts.data(),matrix.data(),cap);}
    P3Result two_stage(const lbvh_e& e,double dHat,P3Queue& q,uint32_t raw_cap,uint32_t cap)
    {return p3_two_stage(e,dHat,q,raw_cap,pairs.data(),ccd.data(),counts.data(),matrix.data(),cap);}
    std::vector<gipc::BvhTypedPair> typed(const std::array<uint32_t,5>& n)const
    {return bvh_probe_typed(pairs.data(),ccd.data(),matrix.data(),n);}
};
gipc::Json p3_resource(const void* kernel)
{
    cudaFuncAttributes attributes{};CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes,kernel));
    int blocks=0,device=0;CUDA_SAFE_CALL(cudaGetDevice(&device));cudaDeviceProp prop{};
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop,device));
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,256,0));
    return {{"registers_per_thread",attributes.numRegs},{"static_local_bytes_per_thread",attributes.localSizeBytes},
        {"static_shared_bytes",attributes.sharedSizeBytes},{"potential_active_blocks_per_sm",blocks},
        {"potential_occupancy",double(blocks*256)/prop.maxThreadsPerMultiProcessor},
        {"measured_local_memory_traffic",nullptr},{"occupancy_kind","resource_limit_not_measured_activity"}};
}
template<class T> gipc::Json p3_dump(const std::filesystem::path& folder,const char* name,
    const T* data,size_t count)
{
    const auto host=bvh_probe_copy(data,count);
    std::ofstream file(folder/name,std::ios::binary);
    if(!file)throw std::runtime_error("Cannot create P3 input snapshot");
    if(count)file.write(reinterpret_cast<const char*>(host.data()),count*sizeof(T));
    file.close();if(!file)throw std::runtime_error("Cannot persist P3 input snapshot");
    return {{"file",name},{"elements",count},{"element_bytes",sizeof(T)}};
}
void p3_study(const lbvh_e& e,double dHat)
{
    gipc::p3_ee_require_edge_count(e.edge_number);
    const auto& options=gipc::p3_ee_options();const int frame=gipc::solve_context().frame;
    static thread_local std::set<int> sampled;
    if(!sampled.insert(frame).second)return;
    const std::filesystem::path file=options.file;
    const auto folder=file.parent_path()/("p3_input_f"+std::to_string(frame));
    if(std::filesystem::exists(folder)||(sampled.size()==1&&std::filesystem::exists(file)))
        throw std::runtime_error("P3 study requires new output paths");
    std::filesystem::create_directories(folder);
    gipc::CostSampleGuard diagnostic("diagnostic_p3_ee");
    gipc::Json result={{"schema","gipc.p3.ee.v1"},{"frame",frame},{"options",options.json()},
        {"dHat_squared_distance",dHat},{"edge_count",e.edge_number},{"production_outputs_untouched",true},
        {"tree_storage_untouched",true},{"physical_quality_certified",false}};
    gipc::attach_solve_context(result);
    const char* binary=std::getenv("GIPC_BINARY_ID");result["binary_id"]=binary?binary:"";
    const auto edges=bvh_probe_copy(e._edges,e.edge_number);size_t nv=0;
    for(auto pair:edges)nv=std::max(nv,size_t(std::max(pair.x,pair.y))+1);
    auto& snapshots=result["input_buffers"];snapshots=gipc::Json::array();
    snapshots.push_back(p3_dump(folder,"positions.bin",e._vertexes,nv));
    snapshots.push_back(p3_dump(folder,"rest.bin",e._rest_vertexes,nv));
    snapshots.push_back(p3_dump(folder,"body.bin",e._bodyId,nv));
    snapshots.push_back(p3_dump(folder,"boundary.bin",e._btype,nv));
    snapshots.push_back(p3_dump(folder,"edges.bin",e._edges,e.edge_number));
    snapshots.push_back(p3_dump(folder,"boxes.bin",e._bvs.data(),e._bvs.size()));
    snapshots.push_back(p3_dump(folder,"nodes.bin",e._nodes.data(),e._nodes.size()));
    result["input_directory"]=folder.string();
    P3Output f,w;f.prepare(0);f.reset();f.fused(e,dHat,0);const uint32_t cap=f.read()[0];
    f.prepare(cap);w.prepare(cap);
    const auto allocate=std::chrono::steady_clock::now();
    P3Queue queue;queue.prepare(options.queue_capacity);
    result["queue_allocation_cpu_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-allocate).count();
    result["queue_allocation_scope"]="header_and_pair_buffer";
    result["queue_device_bytes"]=queue.bytes();
    result["resources"]={{"fused",p3_resource(reinterpret_cast<const void*>(_selfQuery_ee))},
        {"traversal",p3_resource(reinterpret_cast<const void*>(p3_ee_traverse))},
        {"classification",p3_resource(reinterpret_cast<const void*>(p3_ee_classify))}};
    f.reset();f.fused(e,dHat,cap);const auto expected=f.read();const auto tuples=f.typed(expected);
    result["typed_counts"]=expected;result["typed_pairs"]=tuples;
    // This diagnostic replays the EXACT tree/geometry; all outputs are private.
    auto run=[&](bool split,bool check) {
        auto& out=split?w:f;const auto start=std::chrono::steady_clock::now();out.reset();
        P3Result state;
        if(split)state=out.two_stage(e,dHat,queue,options.queue_capacity,cap);else out.fused(e,dHat,cap);
        const auto counts=out.read();const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(counts!=expected)throw std::runtime_error("P3 complete query typed count mismatch");
        if(check&&out.typed(counts)!=tuples)throw std::runtime_error("P3 complete query tuple/MatIndex multiset mismatch");
        return gipc::Json{{"complete_cpu_ms",elapsed},{"raw_pairs",split?gipc::Json(state.raw):gipc::Json(nullptr)},
            {"complete_fused_fallback",state.fallback},{"counts_equal",true},{"typed_multiset_equal",check}};
    };
    for(int warm=0;warm<2;++warm){run(false,true);run(true,true);}
    result["pairs"]=gipc::Json::array();
    for(int i=0;i<7;++i)
    {
        gipc::Json a,b;if(i%2){b=run(true,true);a=run(false,true);}else{a=run(false,true);b=run(true,true);}
        result["pairs"].push_back({{"index",i},{"first",i%2?"W":"F"},{"F",a},{"W",b}});
    }
    // Keep v1 pairs as explicitly labelled historical diagnostics. The v2
    // primary CPU batch does not copy/sort typed arrays between timed calls.
    // Counts readback stays INSIDE each CPU interval, as in production.
    result["legacy_pairs_scope"]="v1; typed D2H and CPU sort between calls; not primary timing";
    auto verify_outputs=[&]() {
        for(auto* out:{&f,&w})
        {
            const auto counts=out->read();
            if(counts!=expected||out->typed(counts)!=tuples)
                throw std::runtime_error("P3 batched boundary tuple/MatIndex mismatch");
        }
    };
    auto batch=[&](bool gpu_events) {
        struct Sample {double cpu=0;P3Result state;std::array<uint32_t,5> counts{};};
        std::array<Sample,gipc::p3_ee_timing_samples> samples{};
        // Event allocation and boundary validation are outside both batches.
        gipc::ScopedCudaEvents<2*gipc::p3_ee_timing_samples> events;
        verify_outputs();
        for(unsigned i=0;i<gipc::p3_ee_timing_samples;++i)
        {
            const bool split=gipc::p3_ee_timing_split(i);auto& out=split?w:f;
            auto& sample=samples[i];const auto start=std::chrono::steady_clock::now();
            if(gpu_events)CUDA_SAFE_CALL(cudaEventRecord(events[2*i],cudaStreamPerThread));
            out.reset();
            if(split)sample.state=out.two_stage(e,dHat,queue,options.queue_capacity,cap);
            else out.fused(e,dHat,cap);
            if(gpu_events)CUDA_SAFE_CALL(cudaEventRecord(events[2*i+1],cudaStreamPerThread));
            sample.counts=out.read();
            sample.cpu=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            // No allocation, JSON construction, typed D2H or CPU sort here.
        }
        verify_outputs();
        gipc::Json rows=gipc::Json::array();
        for(unsigned i=0;i<samples.size();++i)
        {
            const auto& sample=samples[i];
            if(sample.counts!=expected)throw std::runtime_error("P3 batched typed counts mismatch");
            gipc::Json row={{"index",i},{"block",i/4},{"position",i%4},
                {"mode",gipc::p3_ee_timing_split(i)?"W":"F"},{"complete_cpu_ms",sample.cpu},
                {"counts_equal",true},{"complete_fused_fallback",sample.state.fallback}};
            if(gpu_events)
            {
                float elapsed=0;CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed,events[2*i],events[2*i+1]));
                row["gpu_interval_ms"]=elapsed;
            }
            rows.push_back(std::move(row));
        }
        return gipc::Json{{"samples",rows},{"boundary_typed_checks",4},
            {"all_timed_counts_equal",true},{"gpu_events_injected",gpu_events}};
    };
    result["batched_timing"]={{"schema","gipc.p3.ee.balanced.v2"},
        {"schedule","8 alternating FWWF/WFFW blocks; 16 balanced adjacent pairs"},
        {"cpu_primary",batch(false)},{"gpu_separate",batch(true)},
        {"cpu_scope","reset through final typed-count D2H; no events or typed sorting between calls"},
        {"gpu_scope","reset through EE completion, including W intermediate host gap; excludes final count D2H"},
        {"per_call_typed_multiset_checked",false},
        {"correctness_scope","v1 every-call multiset checks plus v2 before/after batch checks"}};
    // GPU stage envelopes are collected separately, never used as the complete
    // query speedup. The queue-count D2H/host gap sits between events 1 and 2.
    gipc::ScopedCudaEvents<4> events;w.reset();
    CUDA_SAFE_CALL(cudaMemsetAsync(queue.state.data(),0,sizeof(P3QueueState),cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaEventRecord(events[0],cudaStreamPerThread));
    if(e.edge_number>1)p3_ee_traverse<<<(e.edge_number+255)/256,256>>>(e._bodyId,e._btype,e._edges,e._bvs.data(),
        e._nodes.data(),dHat,e.edge_number,queue.pairs.data(),options.queue_capacity,queue.state.data());
    CUDA_SAFE_CALL(cudaGetLastError());CUDA_SAFE_CALL(cudaEventRecord(events[1],cudaStreamPerThread));
    P3QueueState raw{};CUDA_SAFE_CALL(cudaMemcpy(&raw,queue.state.data(),sizeof(raw),cudaMemcpyDeviceToHost));
    if(raw.stack_error)throw std::runtime_error("P3 diagnostic traversal stack error");
    CUDA_SAFE_CALL(cudaEventRecord(events[2],cudaStreamPerThread));
    if(raw.count>options.queue_capacity)w.fused(e,dHat,cap);
    else p3_classify(e,dHat,queue,uint32_t(raw.count),w.pairs.data(),w.ccd.data(),w.counts.data(),w.matrix.data(),cap);
    CUDA_SAFE_CALL(cudaEventRecord(events[3],cudaStreamPerThread));const auto final=w.read();
    if(final!=expected||w.typed(final)!=tuples)throw std::runtime_error("P3 stage replay differs");
    float traversal=0,gap=0,classification=0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&traversal,events[0],events[1]));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&gap,events[1],events[2]));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&classification,events[2],events[3]));
    result["gpu_stage_ms"]={{"traversal",traversal},{"queue_readback_host_gap",gap},
        {"classification_or_fallback",classification},{"includes_memsets_and_final_readback",false}};
    if(raw.count<=options.queue_capacity)
    {
        const auto raw_pairs=bvh_probe_copy(queue.pairs.data(),size_t(raw.count));
        std::vector<std::array<uint32_t,2>> raw_ids;raw_ids.reserve(raw_pairs.size());
        for(auto p:raw_pairs)raw_ids.push_back({p.x,p.y});std::sort(raw_ids.begin(),raw_ids.end());
        result["raw_pair_ids"]=raw_ids;
    }
    // Instrumented work counts are outside all timed replays.
    cudatool::DeviceBuffer<gipc::BvhQueryRecord> work;work.resize(e.edge_number>1?e.edge_number:0);
    f.reset();
    if(e.edge_number>1)_selfQuery_ee_probe<<<(e.edge_number+255)/256,256>>>(e._bodyId,e._btype,e._vertexes,
        e._rest_vertexes,e._edges,e._bvs.data(),e._nodes.data(),f.pairs.data(),f.ccd.data(),f.counts.data(),
        f.matrix.data(),dHat,cap,e.edge_number,work.data());
    const auto measured=f.read();if(measured!=expected||f.typed(measured)!=tuples)throw std::runtime_error("P3 instrumented F differs");
    std::vector<gipc::BvhQueryRecord> records;work.copy_to(records);
    result["workload"]=gipc::bvh_query_record_summary(records,false);
    result["raw_count_matches_original_narrow_calls"]=
        raw.count==result["workload"]["narrow_calls"]["sum"].get<uint64_t>();
    if(!result["raw_count_matches_original_narrow_calls"].get<bool>())throw std::runtime_error("P3 missed original narrow calls");
    result["passed"]=true;
    std::ofstream metadata(folder/"snapshot.json");metadata<<result.dump(2);metadata.close();
    std::ofstream stream(file,std::ios::app);stream<<result.dump()<<'\n';stream.close();
    if(!metadata||!stream)throw std::runtime_error("Cannot write P3 query evidence");
}
} // namespace
#include <collision/p3_ee_fixture.inl>
