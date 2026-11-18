#include <linear_system/preconditioner/fem_mas_preconditioner.h>
#include <linear_system/subsystem/fem_linear_subsystem.h>
#include <gipc/utils/timer.h>
#include <fstream>
#include <solver/mas_factor_action_options.h>
#include <gipc/statistics.h>
#include <gipc/cost_trace.h>
namespace gipc
{
Json MAS_Preconditioner::diagnostic_snapshot(const std::string& prefix) const
{
    Json result={{"kind","MAS_full_owned_buffers"},{"offset",get_offset()},
        {"wide_apply",MAS_Prec.wide_apply_enabled()},
        {"inverse64",MAS_Prec.inverse64_enabled()},
        {"cholesky",MAS_Prec.cholesky_enabled()},
        {"factor_action",MAS_Prec.cholesky_enabled()?mas_factor_action_mode():"inactive"},
        {"dimensions",MAS_Prec.diagnostic_dimensions()},
        {"dimension_names",{"nodes","mapped_nodes","levels","collision_offset","clusters","clevel_x","clevel_y","neighbor_list_size"}}};
    MAS_Prec.diagnostic_buffers([&](const char* name,const void* device,size_t count,size_t item_bytes)
    {
        const size_t bytes=count*item_bytes;
        std::vector<unsigned char> data(bytes);
        if(bytes)CUDA_SAFE_CALL(cudaMemcpy(data.data(),device,bytes,cudaMemcpyDeviceToHost));
        std::uint64_t hash=14695981039346656037ull;
        for(auto byte:data){hash^=byte;hash*=1099511628211ull;}
        result["buffers"][name]={{"count",count},{"item_bytes",item_bytes},{"fnv1a64",hash}};
        if(!prefix.empty())
        {
            std::ofstream file(prefix+"_"+name+".bin",std::ios::binary);
            file.write(reinterpret_cast<const char*>(data.data()),bytes);
            if(!file)throw std::runtime_error("Failed to write MAS snapshot");
        }
    });
    return result;
}
std::function<void()> MAS_Preconditioner::checkpoint_scratch() const
{
    struct Saved {void* device;std::vector<unsigned char> bytes;};
    std::vector<Saved> saved;
    MAS_Prec.diagnostic_buffers([&](const char* name,const void* device,size_t count,size_t item_bytes)
    {
        const std::string field(name);
        if(field!="d_multiLevelR" && field!="d_multiLevelZ" &&
           field!="d_multiLevelR64" && field!="d_multiLevelZ64")return;
        Saved value{const_cast<void*>(device),std::vector<unsigned char>(count*item_bytes)};
        if(!value.bytes.empty())CUDA_SAFE_CALL(cudaMemcpy(value.bytes.data(),device,value.bytes.size(),cudaMemcpyDeviceToHost));
        saved.push_back(std::move(value));
    });
    return [saved=std::move(saved)](){for(const auto& value:saved)
        if(!value.bytes.empty())CUDA_SAFE_CALL(cudaMemcpy(value.device,value.bytes.data(),value.bytes.size(),cudaMemcpyHostToDevice));};
}

std::vector<std::uintptr_t> MAS_Preconditioner::graph_signature() const
{
    auto key = MAS_Prec.graph_signature();
    key.push_back(get_offset());
    return key;
}
MAS_Preconditioner::MAS_Preconditioner(FEMLinearSubsystem& subsystem,
                                       MASPreconditioner&  mMAS,
                                       double*             mMasses,
                                       uint32_t*           mCpNum,
                                       const cudatool::DeviceBuffer<int4>& mCollisionPairs)
    : Base(subsystem)
    , MAS_Prec(mMAS)
    , masses(mMasses)
    , cpNum(mCpNum)
    , collision_pairs(mCollisionPairs)
{
    preconditioner_id = 1;
}

void MAS_Preconditioner::assemble()
{
    double      collision_num = *cpNum;
    gipc::Timer timer{"precomputing mas Preconditioner"};
    int         triplet_number = 0;
    uint32_t*   indices = calculate_subsystem_bcoo_indices(triplet_number);
    MAS_Prec.setPreconditioner_bcoo(system_bcoo_matrix(),
                                    system_bcoo_rows(),
                                    system_bcoo_cols(),
                                    indices,
                                    get_offset(),
                                    triplet_number,
                                    collision_num,
                                    collision_pairs.data());
    if(const char* dir=std::getenv("GIPC_DIAGNOSTIC_DIR");(dir&&dir[0])||std::getenv("GIPC_REUSE_OBSERVE_DIR"))
    {
        CostScope observation("diagnostic.mas_topology");
        Json row={{"dimensions",MAS_Prec.diagnostic_dimensions()},
                  {"dimension_names",{"nodes","mapped_nodes","levels","collision_offset","clusters","clevel_x","clevel_y","neighbor_list_size"}}};
        MAS_Prec.diagnostic_buffers([&](const char* name,const void* device,size_t count,size_t item_bytes)
        {
            const std::string key(name);
            if(key!="d_levelSize"&&key!="d_coarseTable"&&key!="d_partId_map_real"&&key!="d_real_map_partId")return;
            std::vector<unsigned char> host(count*item_bytes);
            if(!host.empty())CUDA_SAFE_CALL(cudaMemcpy(host.data(),device,host.size(),cudaMemcpyDeviceToHost));
            uint64_t hash=14695981039346656037ull;
            for(auto byte:host){hash^=byte;hash*=1099511628211ull;}
            row[key]={{"count",count},{"item_bytes",item_bytes},{"fnv1a64",hash}};
        });
        Statistics::instance().at_current_frame()["newton"].back()["mas_diagnostic"]=row;
    }
}

void MAS_Preconditioner::apply(cudatool::CDenseVectorView<Float> r,
                               cudatool::DenseVectorView<Float>  z)
{
    MAS_Prec.preconditioning((double3*)r.data(), (double3*)z.data());
}
}  // namespace gipc
