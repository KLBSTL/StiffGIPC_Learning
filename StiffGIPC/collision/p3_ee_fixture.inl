#include <cstring>
namespace
{
__global__ void p3_fixture_dtype(const double3* x,int* result)
{*result=_dType_EE(x[0],x[1],x[2],x[3]);}
}
int gipc::p3_ee_fixture(const char* output)
{
    auto need=[](bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);};
    std::vector<double3> points;
    auto edge=[&](double3 a,double3 b){points.push_back(a);points.push_back(b);};
    edge(make_double3(0,0,0),make_double3(1,0,0));
    edge(make_double3(.5,-.5,.001),make_double3(.5,.5,.001));
    edge(make_double3(0,.001,0),make_double3(1,.001,0));
    edge(make_double3(1.001,-.5,0),make_double3(1.001,.5,0));
    edge(make_double3(1.001,.001,.001),make_double3(1.1,.1,.1));
    edge(make_double3(0,.002,0),make_double3(1,.002,0));
    edge(make_double3(0,.003,0),make_double3(1,.003,0));
    edge(make_double3(0,.004,0),make_double3(1,.004,0));
    std::vector<uint2> edges;for(uint32_t i=0;i<8;++i)edges.push_back(make_uint2(2*i,2*i+1));
    edges.push_back(make_uint2(0,3)); // shared-vertex rejection; primitive identity retained
    edges.push_back(edges[2]); // repeated primitive geometry must not be silently deduplicated
    std::vector<int> bodies(points.size(),-1),boundary(points.size(),0);
    bodies[0]=bodies[14]=7; // x endpoint body key, matching the original filter
    for(int i=10;i<14;++i)boundary[i]=3;
    cudatool::DeviceBuffer<double3> x(points),rest(points);
    cudatool::DeviceBuffer<uint2> de(edges);
    cudatool::DeviceBuffer<int> db(bodies),dt(boundary);
    lbvh_e e;e.edge_number=uint32_t(edges.size());e.vert_number=uint32_t(points.size());
    e._vertexes=x.data();e._rest_vertexes=rest.data();e._edges=de.data();e._bodyId=db.data();e._btype=dt.data();
    cp_fixture_tree(e,edges,points,std::vector<double3>(points.size(),make_double3(0,0,0)),0);
    constexpr double dHat=.01;
    std::vector<std::array<uint32_t,2>> expected_raw;
    std::vector<uint32_t> per_query(edges.size(),0);
    for(uint32_t i=0;i<edges.size();++i)for(uint32_t j=i+1;j<edges.size();++j)
    {
        const auto a=edges[i],b=edges[j];
        if(bodies[a.x]==bodies[b.x]&&bodies[a.x]!=-1)continue;
        if(a.x==b.x||a.x==b.y||a.y==b.x||a.y==b.y)continue;
        if(boundary[a.x]>=2&&boundary[a.y]>=2&&boundary[b.x]>=2&&boundary[b.y]>=2)continue;
        AABB aa,bb;for(auto k:{a.x,a.y})aa.combines(points[k].x,points[k].y,points[k].z);
        for(auto k:{b.x,b.y})bb.combines(points[k].x,points[k].y,points[k].z);
        if(!overlap(aa,bb,sqrt(dHat)))continue;
        expected_raw.push_back({i,j});++per_query[i];
    }
    need(expected_raw.size()>2&&*std::max_element(per_query.begin(),per_query.end())>1,"P3 fixture lacks a heavy query");
    P3Output f,w;f.prepare(0);f.reset();f.fused(e,dHat,0);const auto contacts=f.read()[0];
    f.prepare(contacts);w.prepare(contacts);f.reset();f.fused(e,dHat,contacts);
    const auto expected=f.read();const auto truth=f.typed(expected);
    need(expected[2]>0&&expected[3]>0&&expected[4]>0,"P3 fixture must exercise PP/PE/EE types");
    P3Queue queue;Json cases=Json::array();
    for(uint32_t capacity:{0u,1u,uint32_t(expected_raw.size()-1),uint32_t(expected_raw.size()),
        uint32_t(expected_raw.size()+1),uint32_t(expected_raw.size()*2),1u})
    {
        // Real device sentinel immediately beyond the DECLARED capacity.
        queue.pairs.resize(capacity+1);const uint2 guard=make_uint2(0xcafebabeu,0xdeadbeefu);
        CUDA_SAFE_CALL(cudaMemcpy(queue.pairs.data()+capacity,&guard,sizeof(guard),cudaMemcpyHostToDevice));
        w.reset();const auto run=w.two_stage(e,dHat,queue,capacity,contacts);const auto counts=w.read();
        uint2 after{};CUDA_SAFE_CALL(cudaMemcpy(&after,queue.pairs.data()+capacity,sizeof(after),cudaMemcpyDeviceToHost));
        need(after.x==guard.x&&after.y==guard.y,"P3 raw queue overrun");
        need(run.raw==expected_raw.size()&&run.fallback==(run.raw>capacity),"P3 capacity boundary/fallback differs");
        need(counts==expected&&w.typed(counts)==truth,"P3 fallback/typed multiset incomplete");
        if(!run.fallback)
        {
            const auto pairs=bvh_probe_copy(queue.pairs.data(),size_t(run.raw));
            std::vector<std::array<uint32_t,2>> actual;for(auto p:pairs)actual.push_back({p.x,p.y});
            std::sort(actual.begin(),actual.end());need(actual==expected_raw,"P3 raw IDs/orientation/filter mismatch");
        }
        cases.push_back({{"capacity",capacity},{"raw",run.raw},{"fallback",run.fallback},{"passed",true}});
    }
    // A pre-existing typed prefix emulates VF counters. F and W must append
    // without clearing, replacing, or incorrectly offsetting MatIndex ranks.
    for(uint32_t raw_capacity:{1u,uint32_t(expected_raw.size())})
    {
      for(bool split:{false,true})
      {
        auto& out=split?w:f;out.prepare(contacts+1);out.reset();
        const std::array<uint32_t,5> prefix={1,0,1,0,0};const int rank=0;
        const int4 pair=make_int4(-1,2,-1,-1),ccd=make_int4(0,1,2,3);
        CUDA_SAFE_CALL(cudaMemcpy(out.counts.data(),prefix.data(),sizeof(prefix),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(out.pairs.data(),&pair,sizeof(pair),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(out.ccd.data(),&ccd,sizeof(ccd),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(out.matrix.data(),&rank,sizeof(rank),cudaMemcpyHostToDevice));
        if(split)out.two_stage(e,dHat,queue,raw_capacity,contacts+1);else out.fused(e,dHat,contacts+1);
      }
      need(f.read()==w.read()&&f.typed(f.read())==w.typed(w.read()),"P3 corrupted VF prefix");
      cases.push_back({{"case","nonzero_vf_prefix"},{"capacity",raw_capacity},{"passed",true}});
    }
    // Typed output shortage must increment full total, avoid OOB writes, and
    // remain recoverable by the unchanged outer host grow-and-requery loop.
    w.prepare(contacts+2);w.reset();const int4 sentinel=make_int4(12345,23456,34567,45678);
    CUDA_SAFE_CALL(cudaMemcpy(w.pairs.data()+1,&sentinel,sizeof(sentinel),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(w.ccd.data()+1,&sentinel,sizeof(sentinel),cudaMemcpyHostToDevice));
    const int matrix_guard=12345678;CUDA_SAFE_CALL(cudaMemcpy(w.matrix.data()+1,&matrix_guard,sizeof(int),cudaMemcpyHostToDevice));
    w.two_stage(e,dHat,queue,uint32_t(expected_raw.size()),1);const auto shortage=w.read();
    need(shortage[0]==contacts&&shortage[2]+shortage[3]+shortage[4]==1,"P3 typed count-only overflow contract differs");
    int4 got{};int got_rank=0;CUDA_SAFE_CALL(cudaMemcpy(&got,w.pairs.data()+1,sizeof(got),cudaMemcpyDeviceToHost));
    need(std::memcmp(&got,&sentinel,sizeof(got))==0,"P3 typed pair overrun");
    CUDA_SAFE_CALL(cudaMemcpy(&got,w.ccd.data()+1,sizeof(got),cudaMemcpyDeviceToHost));
    need(std::memcmp(&got,&sentinel,sizeof(got))==0,"P3 CCD tuple overrun");
    CUDA_SAFE_CALL(cudaMemcpy(&got_rank,w.matrix.data()+1,sizeof(int),cudaMemcpyDeviceToHost));
    need(got_rank==matrix_guard,"P3 matrix rank overrun");
    w.reset();w.two_stage(e,dHat,queue,uint32_t(expected_raw.size()),contacts);
    need(w.read()==expected&&w.typed(w.read())==truth,"P3 typed growth replay lost output");
    cases.push_back({{"case","typed_shortage_and_growth"},{"passed",true}});
    // Duplicate queue contributions are preserved. Compare with two original
    // classifications, not with a deduplicated expected set.
    const auto first=expected_raw.front();std::vector<uint2> duplicated(2,make_uint2(first[0],first[1]));
    queue.pairs.copy_from(duplicated);w.reset();p3_classify(e,dHat,queue,2,w.pairs.data(),w.ccd.data(),w.counts.data(),w.matrix.data(),contacts);
    const auto duplicate_counts=w.read();need(duplicate_counts[0]==2,"P3 duplicate contribution was discarded");
    const auto duplicate_tuples=w.typed(duplicate_counts);need(duplicate_tuples[0]==duplicate_tuples[1],"P3 duplicate tuple identity changed");
    cases.push_back({{"case","duplicate_contributions"},{"passed",true}});
    // A repeat at the old larger size must not consume residual queue entries.
    w.reset();w.two_stage(e,dHat,queue,uint32_t(expected_raw.size()),contacts);
    need(w.read()==expected&&w.typed(w.read())==truth,"P3 repeated query stale output");
    for(uint32_t number:{0u,1u})
    {
        e.edge_number=number;w.reset();const auto empty=w.two_stage(e,dHat,queue,0,contacts);
        need(empty.raw==0&&!empty.fallback&&w.read()[0]==0,"P3 empty/singleton query accessed internal nodes");
        cases.push_back({{"case","empty_or_singleton"},{"edges",number},{"passed",true}});
    }
    // Check all nine original feature classifications and exact distance
    // thresholds on isolated pairs. The diagonal PP threshold remains strictly
    // inside the AABB gap, separating narrow d<dHat from broad-phase rounding.
    auto pair_case=[&](const char* name,double3 b0,double3 b1,double threshold,
        uint32_t expected_contacts,int expected_dtype,int mollified,uint64_t expected_raw=1) {
        std::vector<double3> px={make_double3(0,0,0),make_double3(1,0,0),b0,b1};
        std::vector<uint2> pe={make_uint2(0,1),make_uint2(2,3)};
        cudatool::DeviceBuffer<double3> dx(px),dr(px);
        cudatool::DeviceBuffer<uint2> edges_device(pe);
        cudatool::DeviceBuffer<int> body_device(std::vector<int>(4,-1)),boundary_device(std::vector<int>(4,0));
        lbvh_e tree;tree.edge_number=2;tree.vert_number=4;tree._vertexes=dx.data();tree._rest_vertexes=dr.data();
        tree._edges=edges_device.data();tree._bodyId=body_device.data();tree._btype=boundary_device.data();
        cp_fixture_tree(tree,pe,px,std::vector<double3>(4,make_double3(0,0,0)),0);
        cudatool::DeviceVar<int> dtype_device;p3_fixture_dtype<<<1,1>>>(dx.data(),dtype_device.data());
        CUDA_SAFE_CALL(cudaGetLastError());int dtype=-1;
        CUDA_SAFE_CALL(cudaMemcpy(&dtype,dtype_device.data(),sizeof(dtype),cudaMemcpyDeviceToHost));
        need(dtype==expected_dtype,"P3 fixture did not reach its declared EE feature case");
        P3Output a,b;a.prepare(2);b.prepare(2);a.reset();b.reset();a.fused(tree,threshold,2);
        const auto run=b.two_stage(tree,threshold,queue,1,2);const auto ac=a.read(),bc=b.read();
        need(!run.fallback&&run.raw==expected_raw&&ac==bc&&ac[0]==expected_contacts&&a.typed(ac)==b.typed(bc),"P3 isolated feature/threshold mismatch");
        if(expected_contacts)
        {
            const int type=(dtype==0||dtype==1||dtype==3||dtype==4)?2:(dtype==8?4:3);
            need(ac[type]==1,"P3 isolated contact has wrong typed bucket");
            need((a.typed(ac).front()[4]<=-2)==bool(mollified),"P3 isolated mollifier encoding differs");
        }
        cases.push_back({{"case",name},{"dtype",dtype},{"dHat",threshold},{"raw",run.raw},
            {"typed",ac[0]},{"passed",true}});
    };
    const std::array<std::array<double3,2>,9> features={{
        {make_double3(-.125,.125,0),make_double3(-.125,1.125,0)},
        {make_double3(-.125,1.125,0),make_double3(-.125,.125,0)},
        {make_double3(-.125,-.5,0),make_double3(-.125,.5,0)},
        {make_double3(1.125,.125,0),make_double3(1.125,1.125,0)},
        {make_double3(1.125,1.125,0),make_double3(1.125,.125,0)},
        {make_double3(1.125,-.5,0),make_double3(1.125,.5,0)},
        {make_double3(.5,.125,0),make_double3(.5,1.125,0)},
        {make_double3(.5,1.125,0),make_double3(.5,.125,0)},
        {make_double3(.5,-.5,.125),make_double3(.5,.5,.125)}}};
    for(int dtype=0;dtype<9;++dtype)pair_case(("feature_"+std::to_string(dtype)).c_str(),
        features[dtype][0],features[dtype][1],.25,1,dtype,0);
    const double threshold=1.0/32;
    pair_case("distance_below",features[0][0],features[0][1],std::nextafter(threshold,0.0),0,0,0);
    pair_case("distance_equal",features[0][0],features[0][1],threshold,0,0,0);
    pair_case("distance_above",features[0][0],features[0][1],std::nextafter(threshold,1.0),1,0,0);
    pair_case("aabb_equal",make_double3(0,.125,0),make_double3(1,.125,0),1.0/64,0,0,1,0);
    pair_case("parallel_mollified",make_double3(0,.125,0),make_double3(1,.125,0),threshold,1,0,1);
    pair_case("near_parallel_mollified",make_double3(0,.125,0),make_double3(1,.125+1e-8,0),threshold,1,0,1);
    pair_case("zero_length_second_edge",make_double3(.5,.125,0),make_double3(.5,.125,0),threshold,1,6,0);
    // An intentionally over-deep tree is NOT passed to the unguarded F kernel.
    // Its right spine accumulates one unvisited internal left child per level.
    {
        constexpr uint32_t levels=67,number=2*levels+1,invalid=0xffffffffu;
        std::vector<double3> px;std::vector<uint2> pe;
        for(uint32_t i=0;i<number;++i){px.push_back(make_double3(0,0,0));px.push_back(make_double3(1,0,0));pe.push_back(make_uint2(2*i,2*i+1));}
        std::vector<Node> nodes(2*number-1);uint32_t internal=0,leaf=0;
        auto add_leaf=[&](uint32_t parent){const auto id=number-1+leaf;nodes[id]={parent,invalid,invalid,leaf++};return id;};
        std::function<uint32_t(uint32_t,uint32_t)> chain=[&](uint32_t level,uint32_t parent){
            const auto id=internal++,left=internal++;
            const auto ll=add_leaf(left),lr=add_leaf(left);nodes[left]={id,ll,lr,invalid};
            const auto right=level+1==levels?add_leaf(id):chain(level+1,id);
            nodes[id]={parent,left,right,invalid};return id;
        };
        need(chain(0,invalid)==0&&internal==number-1&&leaf==number,"P3 deep fixture topology invalid");
        AABB box;box.combines(0,0,0);box.combines(1,0,0);
        cudatool::DeviceBuffer<double3> dx(px),dr(px);cudatool::DeviceBuffer<uint2> ed(pe);
        cudatool::DeviceBuffer<int> bd(std::vector<int>(px.size(),-1)),bt(std::vector<int>(px.size(),0));
        lbvh_e deep;deep.edge_number=number;deep._vertexes=dx.data();deep._rest_vertexes=dr.data();
        deep._edges=ed.data();deep._bodyId=bd.data();deep._btype=bt.data();
        deep._nodes.copy_from(nodes);deep._bvs.copy_from(std::vector<AABB>(nodes.size(),box));
        bool rejected=false;try{p3_collect(deep,dHat,queue,1);}
        catch(const std::runtime_error& error){rejected=std::string(error.what()).find("65-entry")!=std::string::npos;}
        need(rejected,"P3 over-deep traversal did not reject before classification");
        e.edge_number=uint32_t(edges.size());w.reset();w.two_stage(e,dHat,queue,uint32_t(expected_raw.size()),contacts);
        need(w.read()==expected&&w.typed(w.read())==truth,"P3 stack error flag survived the next reset");
        cases.push_back({{"case","stack_overflow_rejected_and_next_query_reset"},{"passed",true}});
    }
    std::ofstream stream(output);stream<<Json{{"passed",true},{"raw_pairs",expected_raw.size()},
        {"typed_counts",expected},{"max_raw_per_query",*std::max_element(per_query.begin(),per_query.end())},{"cases",cases}}.dump(2);
    stream.close();return stream?0:1;
}
