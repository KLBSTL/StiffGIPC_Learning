// Frozen benchmark manifests preserve geometry/material settings across arms.
cudatool::DeviceBuffer<int> benchmark_collision_ids;
std::vector<std::pair<int,int>> benchmark_fixed_spans;
gipc::Json benchmark_scene_record;

Eigen::Matrix4d benchmark_matrix(const gipc::Json& a)
{
    if(a.size()!=16)throw std::runtime_error("Scene transform must have 16 entries");
    Eigen::Matrix4d m;
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)m(r,c)=a[r*4+c].get<double>();
    return m;
}
void apply_solver_overrides()
{
    if(const char* v=std::getenv("GIPC_NEWTON_TOL"))ipc.Newton_solver_threshold=std::stod(v);
    if(const char* v=std::getenv("GIPC_PCG_TOL"))ipc.pcg_threshold=std::stod(v);
    if(const char* v=std::getenv("GIPC_DT"))ipc.IPC_dt=std::stod(v);
    if(const char* v=std::getenv("GIPC_FRICTION"))ipc.frictionRate=ipc.gd_frictionRate=std::stod(v);
}
void load_benchmark_scene(const std::string& case_id)
{
    if(case_id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos)
        throw std::runtime_error("Invalid benchmark case ID");
    std::ifstream input(assets_dir+"benchmark_scenes/"+case_id+".json");
    if(!input)throw std::runtime_error("Unknown benchmark scene: "+case_id);
    benchmark_scene_record=gipc::Json::parse(input);
    const auto& f=benchmark_scene_record.at("effective_scalar_fields");
    ipc.pcg_data.P_type=f.at("preconditioner_type").get<int>();
    ipc.IPC_dt=f.at("dt").get<double>();ipc.density=f.at("density").get<double>();
    ipc.PoissonRate=f.at("poisson_ratio").get<double>();
    ipc.clothDensity=f.at("cloth_density").get<double>();ipc.clothThickness=f.at("cloth_thickness").get<double>();
    ipc.clothYoungModulus=f.at("cloth_young").get<double>();ipc.bendYoungModulus=f.at("bending_young").get<double>();
    ipc.strainRate=f.at("strain_rate").get<double>();ipc.relative_dhat=f.at("relative_dhat").get<double>();
    ipc.frictionRate=f.at("friction_rate").get<double>();ipc.gd_frictionRate=f.at("ground_friction_rate").get<double>();
    ipc.pcg_threshold=f.at("pcg_requested_threshold").get<double>();
    ipc.Newton_solver_threshold=f.at("newton_direction_threshold_coefficient").get<double>();
    gipc::SimpleSceneImporter importer;
    for(const auto& object:benchmark_scene_record.at("objects"))
    {
        bool abd=object.at("body_type")=="ABD";
        int dim=object.at("dimension").get<int>();
        Eigen::Matrix4d source=Eigen::Matrix4d::Identity();
        if(object.contains("source_transform_f64x16"))source=benchmark_matrix(object.at("source_transform_f64x16"));
        else
        {
            double scale=object.value("source_scale",1.0);source.block<3,3>(0,0)*=scale;
            if(object.contains("world_translation_m"))
                for(int j=0;j<3;++j)source(j,3)=object.at("world_translation_m")[j].get<double>();
        }
        std::vector<Eigen::Matrix4d> instances;
        if(object.contains("instance_transforms_f64x16"))
            for(const auto& a:object.at("instance_transforms_f64x16"))instances.push_back(benchmark_matrix(a));
        else instances.push_back(Eigen::Matrix4d::Identity());
        std::string fixed=object.value("fixed_mode",std::string("none"));
        double young=1e4;
        if(object.contains("young_pa") && !object.at("young_pa").is_null())young=object.at("young_pa").get<double>();
        if(object.contains("tet_young_pa") && !object.at("tet_young_pa").is_null())young=object.at("tet_young_pa").get<double>();
        for(const auto& instance:instances)
        {
            int begin=tetMesh.vertexNum;
            importer.load_geometry(tetMesh,dim,abd?gipc::BodyType::ABD:gipc::BodyType::FEM,
                instance*source,young,assets_dir+object.at("stiff_mesh").get<std::string>(),
                ipc.pcg_data.P_type,fixed=="all"?BodyBoundaryType::Fixed:BodyBoundaryType::Free);
            int end=tetMesh.vertexNum;
            if(fixed=="all" && !abd && !object.value("self_collision",true))
                benchmark_fixed_spans.emplace_back(begin,end);
            if(fixed=="diagonal_corners")
            {
                double xmin=1e300,xmax=-1e300,zmin=1e300,zmax=-1e300;
                for(int i=begin;i<end;++i){auto v=tetMesh.vertexes[i];xmin=std::min(xmin,v.x);xmax=std::max(xmax,v.x);zmin=std::min(zmin,v.z);zmax=std::max(zmax,v.z);}
                for(int i=begin;i<end;++i){auto v=tetMesh.vertexes[i];double x=(v.x-xmin)/(xmax-xmin),z=(v.z-zmin)/(zmax-zmin);
                    if((x<1e-3&&z<1e-3)||(x>1-1e-3&&z>1-1e-3))tetMesh.boundaryTypies[i]=1;}
            }
        }
    }
    if(case_id.rfind("cloth_sphere7_",0)==0)
    {tetMesh.minConer=make_double3(-.8,-.87,-.8);tetMesh.maxConer=make_double3(.8,.2,.8);}
    apply_solver_overrides();
    benchmark_scene_record["effective_run"]={{"newton_tol",ipc.Newton_solver_threshold},{"pcg_tol",ipc.pcg_threshold},
        {"dt",ipc.IPC_dt},{"friction",ipc.frictionRate}};
    std::filesystem::create_directories(std::string(gipc::output_dir()));
    std::ofstream output(std::string(gipc::output_dir())+"/scene.json");
    output<<benchmark_scene_record.dump(2);
    std::cout<<"FROZEN_SCENE_READY "<<case_id<<" vertices="<<tetMesh.vertexNum<<std::endl;
}
