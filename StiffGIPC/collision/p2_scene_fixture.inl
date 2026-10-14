// Native lifecycle test: allocations, singleton, ordinary/swept storage swaps,
// refit after geometry changes and CPU consumers following deferred updates.
int gipc::p2_scene_fixture(const char* output)
{
    std::vector<double3> x={make_double3(-1,0,2),make_double3(3,1,-2),
        make_double3(0,-4,1),make_double3(2,2,5)};
    std::vector<double3> direction(x.size(),make_double3(1,-2,3));
    cudatool::DeviceBuffer<double3> dx(x),rest(x),dd(direction);
    cudatool::DeviceBuffer<uint3> df(std::vector<uint3>{make_uint3(0,1,2),make_uint3(1,2,3)});
    cudatool::DeviceBuffer<uint2> de(std::vector<uint2>{make_uint2(0,1),make_uint2(2,3)});
    cudatool::DeviceBuffer<uint32_t> ds(std::vector<uint32_t>{0,1,2,3});
    cudatool::DeviceBuffer<int> db(std::vector<int>(4,-1)),dt(std::vector<int>(4,0));
    lbvh_f f;lbvh_e e;
    f.init(db.data(),dt.data(),dx.data(),df.data(),ds.data(),2,4);
    e.init(db.data(),dt.data(),dx.data(),rest.data(),de.data(),2,4);
    Json rows=Json::array();
    auto check=[&](lbvh& tree,const char* name,double alpha,bool last_vertex=true) {
        AABB expected;
        for(size_t i=0;i<(last_vertex?x.size():3);++i)
        {
            const auto& p=x[i];expected.combines(p.x,p.y,p.z);
            expected.combines(p.x-alpha*direction[i].x,p.y-alpha*direction[i].y,p.z-alpha*direction[i].z);
        }
        const auto first=tree.host_scene(),second=tree.host_scene();
        const bool equal=discrete_box_equal(first,expected)&&discrete_box_equal(first,second);
        rows.push_back({{"case",name},{"passed",equal}});
        if(!equal)throw std::runtime_error(std::string("P2 scene fixture mismatch: ")+name);
    };
    bool invalid=false;
    try {f.host_scene();}catch(const std::runtime_error&){invalid=true;}
    if(!invalid)throw std::runtime_error("P2 scene fixture exposed uninitialized allocation");
    rows.push_back({{"case","uninitialized_root_rejected"},{"passed",true}});
    f.Construct();e.Construct();check(f,"face_build",0);check(e,"edge_build",0);
    f.ConstructFullCCD(dd.data(),.5);e.ConstructFullCCD(dd.data(),.5);
    check(f,"face_swept",.5);check(e,"edge_swept",.5);
    f.Construct();e.Construct();check(f,"face_ordinary_restore",0);check(e,"edge_ordinary_restore",0);
    x[3]=make_double3(12,-8,7);dx.copy_from(x);
    f.RefitDiscrete();e.RefitDiscrete();check(f,"face_refit_growth",0);check(e,"edge_refit_growth",0);
    f.RefitFullCCD(dd.data(),.25);e.RefitFullCCD(dd.data(),.25);
    check(f,"face_swept_refit",.25);check(e,"edge_swept_refit",.25);
    f.getSceneSize();check(f,"device_scene_utility",0);
    f.face_number=1;f.MALLOC_DEVICE_MEM(1);f.ConstructRebuild();check(f,"singleton",0,false);
    e.edge_number=0;e.MALLOC_DEVICE_MEM(0);e.Construct();e.RefitFullCCD(dd.data(),1);
    invalid=false;try {e.host_scene();}catch(const std::runtime_error&){invalid=true;}
    if(!invalid)throw std::runtime_error("P2 scene fixture exposed empty root");
    rows.push_back({{"case","empty_edge_rejected"},{"passed",true}});
    std::ofstream stream(output);if(!stream)throw std::runtime_error("Cannot write P2 scene fixture");
    stream<<Json{{"passed",true},{"options",p2_options().json()},{"cases",rows}}.dump(2);
    return stream?0:1;
}
