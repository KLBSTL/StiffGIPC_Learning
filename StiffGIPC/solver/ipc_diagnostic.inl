// Opt-in observation for the cumulative-TOI experiment. Never changes a stop rule.
#include <filesystem>
#include <numeric>
#include <climits>
#include <array>
#include <iterator>
namespace {
__global__ void ipc_diagnostic_field(int vertices,int abd_vertices,int bodies,
    const gipc::ABDJacobi* jacobian,const int* body_ids,const gipc::Vector12* abd_residual,
    const BodyBoundaryType* abd_boundary,int fem_offset,int fem_count,
    const double3* fb,const double3* shape,const double* mass,const int* boundary,
    double inverse_dt,double4* field) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=vertices)return;
    double4 value=make_double4(0,0,0,0);
    if(i<abd_vertices) {
        const int body=body_ids[i];
        if(body>=0&&body<bodies&&abd_boundary[body]!=BodyBoundaryType::Fixed) {
            const gipc::Vector3 r=(jacobian[i]*abd_residual[body])*inverse_dt;
            value=make_double4(r[0],r[1],r[2],mass[i]);
        }
    } else if(i>=fem_offset&&i<fem_offset+fem_count&&boundary[i]==0) {
        const double scale=inverse_dt/mass[i];
        value=make_double4((fb[i].x+shape[i].x)*scale,(fb[i].y+shape[i].y)*scale,
                          (fb[i].z+shape[i].z)*scale,mass[i]);
    }
    field[i]=value;
}
bool ipc_diagnostic_enabled(){const char* p=std::getenv("GIPC_DIAGNOSTIC_DIR");return p&&p[0];}
gipc::Json ipc_diagnostic_observe(int frame,int iteration,bool terminal,int vertices,
    const double* gradient,const gipc::Matrix12x12* inverse_mass,const BodyBoundaryType* abd_boundary,
    int bodies,const gipc::ABDJacobi* jacobian,const int* body_ids,int abd_vertices,
    const double3* fb,const double3* shape,const double* mass,const int* boundary,
    int fem_offset,int fem_count,double dt,const int4* pairs,int pair_count,
    const uint32_t* ground,int ground_count,const double3* positions) {
    gipc::CostScope scope("diagnostic.full_residual");
    const auto started=std::chrono::steady_clock::now();
    static thread_local cudatool::DeviceBuffer<gipc::Vector12> ar;
    static thread_local cudatool::DeviceBuffer<double4> field;
    ar.resize(bodies);field.resize(vertices);
    if(bodies)ipc_mass_residual_abd<<<(bodies+255)/256,256>>>(bodies,gradient,inverse_mass,abd_boundary,ar.data());
    ipc_diagnostic_field<<<(vertices+255)/256,256>>>(vertices,abd_vertices,bodies,jacobian,body_ids,
        ar.data(),abd_boundary,fem_offset,fem_count,fb,shape,mass,boundary,1/dt,field.data());
    CUDA_SAFE_CALL(cudaGetLastError());
    std::vector<double4> host(vertices);
    CUDA_SAFE_CALL(cudaMemcpy(host.data(),field.data(),vertices*sizeof(double4),cudaMemcpyDeviceToHost));
    std::vector<double> squared;double sum=0,weighted=0,total_mass=0,maximum=0;
    for(const auto& v:host)if(v.w!=0) {
        const double s=v.x*v.x+v.y*v.y+v.z*v.z;
        if(!std::isfinite(s)||!std::isfinite(v.w)||v.w<=0)throw std::runtime_error("Invalid diagnostic residual/mass");
        squared.push_back(s);sum+=s;weighted+=v.w*s;total_mass+=v.w;maximum=std::max(maximum,std::sqrt(s));
    }
    // Independent reduction path already used by the forcing monitor.
    const double check=ipc_physical_residual(gradient,inverse_mass,abd_boundary,bodies,jacobian,body_ids,
        abd_vertices,fb,shape,mass,boundary,fem_offset,fem_count,dt);
    if(std::abs(check-maximum)>1e-10*std::max({1.,check,maximum}))throw std::runtime_error("Residual field/max mismatch");
    std::sort(squared.begin(),squared.end(),std::greater<double>());
    auto share=[&](double fraction){const size_t count=std::min(squared.size(),static_cast<size_t>(std::ceil(fraction*squared.size())));
        return sum>0?std::accumulate(squared.begin(),squared.begin()+count,0.)/sum:0.;};
    gipc::Json row={{"frame",frame},{"iteration",iteration},{"terminal",terminal},{"active_vertices",squared.size()},
        {"residual_max",maximum},{"residual_mass_rms",total_mass>0?std::sqrt(weighted/total_mass):0.},
        {"residual_rms",squared.empty()?0.:std::sqrt(sum/squared.size())},
        {"top1_share",share(.01)},{"top5_share",share(.05)},{"top10_share",share(.1)},
        {"pair_count",pair_count},{"ground_count",ground_count},{"units","m/s"}};
    static thread_local std::vector<std::array<int,4>> previous;
    std::vector<int4> raw(pair_count);std::vector<uint32_t> gp(ground_count);
    if(pair_count)CUDA_SAFE_CALL(cudaMemcpy(raw.data(),pairs,pair_count*sizeof(int4),cudaMemcpyDeviceToHost));
    if(ground_count)CUDA_SAFE_CALL(cudaMemcpy(gp.data(),ground,ground_count*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    std::vector<std::array<int,4>> current;
    for(auto p:raw)current.push_back({p.x,p.y,p.z,p.w});
    for(auto p:gp)current.push_back({INT_MIN,static_cast<int>(p),0,0});
    std::sort(current.begin(),current.end());current.erase(std::unique(current.begin(),current.end()),current.end());
    std::vector<std::array<int,4>> common;
    std::set_intersection(previous.begin(),previous.end(),current.begin(),current.end(),std::back_inserter(common));
    const size_t united=previous.size()+current.size()-common.size();
    row["pair_record_churn"]=united?1.-double(common.size())/united:0.;
    row["unique_pair_records"]=current.size();previous=std::move(current);
    if(terminal) {
        const auto directory=std::filesystem::path(std::getenv("GIPC_DIAGNOSTIC_DIR"));
        std::filesystem::create_directories(directory);
        char name[64];std::snprintf(name,sizeof(name),"field_%04d.bin",frame);
        std::ofstream output(directory/name,std::ios::binary);
        output.write(reinterpret_cast<const char*>(host.data()),host.size()*sizeof(double4));
        if(!output)throw std::runtime_error("Cannot write residual field");
        std::snprintf(name,sizeof(name),"positions_%04d.bin",frame);
        std::vector<double3> x(vertices);CUDA_SAFE_CALL(cudaMemcpy(x.data(),positions,vertices*sizeof(double3),cudaMemcpyDeviceToHost));
        std::ofstream position_output(directory/name,std::ios::binary);
        position_output.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(double3));
        if(!position_output)throw std::runtime_error("Cannot write residual positions");
    }
    row["observation_host_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return row;
}
}
