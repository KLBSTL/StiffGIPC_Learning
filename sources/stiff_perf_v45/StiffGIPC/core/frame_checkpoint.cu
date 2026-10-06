// Same-binary physical-frame-boundary checkpoints. Derived solver/BVH caches
// are rebuilt; no partially completed Newton/PCG call is serialized.
#include <core/GIPC.cuh>
#include <core/gipc_path.h>
#include <abd_system/abd_sim_data.h>
#include <abd_system/abd_system.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <cstring>
#include <type_traits>
#include <cstdlib>
extern int total_Frames;
extern bool isUpdateBoundary;
namespace
{
uint64_t checkpoint_hash(const std::vector<unsigned char>& bytes)
{
    uint64_t hash=14695981039346656037ull;
    for(auto b:bytes){hash^=b;hash*=1099511628211ull;}
    return hash;
}
std::vector<unsigned char> checkpoint_device_bytes(const void* pointer,size_t bytes)
{
    std::vector<unsigned char> result(bytes);
    if(bytes)CUDA_SAFE_CALL(cudaMemcpy(result.data(),pointer,bytes,cudaMemcpyDeviceToHost));
    return result;
}
std::string checkpoint_env(const char* key)
{const char* value=std::getenv(key);return value?value:"";}
}

void GIPC::frame_checkpoint(device_TetraData& mesh,const std::string& directory,bool restore)
{
    namespace fs=std::filesystem;
    if(!use_toi || isUpdateBoundary || animation)
        throw std::runtime_error("Frame checkpoint supports static-topology TOI batch scenes without animated boundaries");
    const std::string binary_id=checkpoint_env("GIPC_BINARY_ID");
    if(binary_id.empty())throw std::runtime_error("Checkpoint requires the verified binary identity from its runner");
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const fs::path folder(directory);
    gipc::Json meta;
    if(restore)
    {
        std::ifstream input(folder/"checkpoint.json");
        if(!input)throw std::runtime_error("Checkpoint has no committed metadata");
        input>>meta;
        if(meta.at("schema")!="gipc-v45-frame-boundary-1" || meta.at("binary_id")!=binary_id
           || meta.at("contact_bytes")!=sizeof(ToiContact))
            throw std::runtime_error("Checkpoint schema/binary/ABI mismatch");
    }
    else
    {
        if(fs::exists(folder))throw std::runtime_error("Refusing to overwrite an existing checkpoint");
        fs::create_directories(folder);
        meta={{"schema","gipc-v45-frame-boundary-1"},{"binary_id",binary_id},{"contact_bytes",sizeof(ToiContact)},
            {"completed_physical_frames",total_Frames},{"next_physical_frame",total_Frames+1},
            {"scope","Same-binary static-topology TOI frame boundary; no mid-Newton restart"},
            {"derived_caches","BVH rebuilt; matrices, preconditioner and CUDA graphs regenerated on next solve"}};
    }
    gipc::Json context={{"scene",checkpoint_env("GIPC_SCENE")},{"case",checkpoint_env("GIPC_CASE")},
        {"vertices",vertexNum},{"faces",surface_Num},{"tets",tetrahedraNum},{"edges",edge_Num},
        {"dt",IPC_dt},{"newton_tol",Newton_solver_threshold},{"friction",frictionRate},{"ground_friction",gd_frictionRate},
        {"stretch",stretchStiff},{"shear",shearStiff},{"bend",bendStiff},{"strain",strainRate},
        {"length_lame",lengthRateLame},{"volume_lame",volumeRateLame},{"abd_kappa",m_abd_system->parms.kappa},
        {"abd_dt",m_abd_system->parms.dt},{"abd_density",m_abd_system->parms.mass_density},
        {"abd_gravity",{m_abd_system->parms.gravity[0],m_abd_system->parms.gravity[1],m_abd_system->parms.gravity[2]}},
        {"motor_speed",m_abd_system->parms.motor_speed},
        {"motor_strength",m_abd_system->parms.motor_strength}};
    for(const char* key:{"GIPC_TOI_POLICY","GIPC_TOI_MU_SCALE","GIPC_TOI_MU_MODE","GIPC_TOI_MU_SCOPE",
        "GIPC_TOI_DELTA","GIPC_TOI_PERSIST_CONTACTS","GIPC_TOI_WARM_CONTACTS","GIPC_TOI_WARM_LAMBDA",
        "GIPC_TOI_WARM_GAMMA","GIPC_TOI_WARM_FRICTION","GIPC_TOI_INDEPENDENT_FRICTION_HISTORY",
        "GIPC_TOI_STALL_WINDOW","GIPC_TOI_STALL_HOLD_DELTA","GIPC_TOI_SAFE_INJECTIVITY",
        "GIPC_TOI_TRIAL_INJECTIVITY","GIPC_TOI_ROBUST_VELOCITY_TOL","GIPC_TOI_VOLUME_BOUND",
        "GIPC_TOI_FILTER_RATIO","GIPC_TOI_EARLY_TERMINATION","GIPC_TOI_REUSE_INITIAL_ASSEMBLY"})
        context["environment"][key]=checkpoint_env(key);
    auto immutable=[&](const char* name,const auto& buffer)
    {
        auto bytes=checkpoint_device_bytes(buffer.data(),buffer.size()*sizeof(*buffer.data()));
        context["immutable"][name]={{"bytes",bytes.size()},{"fnv1a64",checkpoint_hash(bytes)}};
    };
    immutable("rest_vertices",mesh.rest_vertexes);immutable("tetrahedra",mesh.tetrahedras);
    immutable("triangles",mesh.triangles);immutable("faces",_faces);immutable("edges",_edges);
    immutable("masses",mesh.masses);immutable("boundary",mesh.BoundaryType);
    immutable("point_body",mesh.point_id_to_body_id);
    // shape_grads is the assembled RHS workspace, cleared by computeGradientAndHessian.
    immutable("tet_inverse",mesh.DmInverses);immutable("triangle_inverse",mesh.triDmInverses);
    immutable("length_rate",mesh.lengthRate);immutable("volume_rate",mesh.volumeRate);
    immutable("ground_normals",_groundNormal);immutable("ground_offsets",_groundOffset);
    immutable("abd_boundary",m_abd_sim_data->body_id_to_boundary_type());
    immutable("abd_tet_volumes",m_abd_sim_data->tet_id_to_volume());
    // ABD local coordinates and mass reductions depend on initialization order.
    // Restore them together with q, instead of mixing q with freshly reduced J/M.
    immutable("motor_axis",m_abd_system->body_id_to_motor_rotation_axis);
    if(restore && meta.at("context")!=context)throw std::runtime_error("Checkpoint scene/physics context mismatch");
    if(!restore)meta["context"]=context;

    // Validate every member before changing device state. Paths are fixed
    // identifiers, not arbitrary paths read from checkpoint metadata.
    std::map<std::string,std::vector<unsigned char>> payloads;
    if(restore)
    {
        for(auto it=meta.at("buffers").begin();it!=meta.at("buffers").end();++it)
        {
            const auto name=it.key();
            if(name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos)
                throw std::runtime_error("Invalid checkpoint member name");
            const auto bytes=it.value().at("bytes").get<size_t>();
            if(bytes>1024ull*1024*1024 || fs::file_size(folder/(name+".bin"))!=bytes)
                throw std::runtime_error("Checkpoint member size mismatch");
            auto& data=payloads[name];data.resize(bytes);
            std::ifstream file(folder/(name+".bin"),std::ios::binary);
            if(bytes)file.read(reinterpret_cast<char*>(data.data()),bytes);
            if(!file || checkpoint_hash(data)!=it.value().at("fnv1a64").get<uint64_t>())
                throw std::runtime_error("Checkpoint member integrity mismatch: "+name);
        }
    }
    size_t restored_buffers=0;
    auto transfer=[&](const char* name,auto& buffer,bool device)
    {
        using T=std::remove_reference_t<decltype(*buffer.data())>;
        if(restore)
        {
            const auto& data=payloads.at(name);
            if(data.size()%sizeof(T))throw std::runtime_error("Checkpoint element size mismatch");
            buffer.resize(data.size()/sizeof(T));
            if(!data.empty())
            {
                if(device)CUDA_SAFE_CALL(cudaMemcpy(buffer.data(),data.data(),data.size(),cudaMemcpyHostToDevice));
                else std::memcpy(buffer.data(),data.data(),data.size());
            }
            auto actual=device?checkpoint_device_bytes(buffer.data(),data.size()):data;
            if(actual!=data)throw std::runtime_error("Checkpoint device restore mismatch");
            ++restored_buffers;
        }
        else
        {
            const size_t bytes=buffer.size()*sizeof(T);
            std::vector<unsigned char> data(bytes);
            if(bytes)
            {
                if(device)CUDA_SAFE_CALL(cudaMemcpy(data.data(),buffer.data(),bytes,cudaMemcpyDeviceToHost));
                else std::memcpy(data.data(),buffer.data(),bytes);
            }
            std::ofstream file(folder/(std::string(name)+".bin"),std::ios::binary);
            if(bytes)file.write(reinterpret_cast<const char*>(data.data()),bytes);
            file.close();if(!file)throw std::runtime_error("Checkpoint member write failed");
            meta["buffers"][name]={{"bytes",bytes},{"count",buffer.size()},{"item_bytes",sizeof(T)},{"fnv1a64",checkpoint_hash(data)}};
        }
    };
#define DEVICE(name,value) transfer(name,value,true)
    DEVICE("vertices",mesh.vertexes);DEVICE("old_vertices",mesh.o_vertexes);
    DEVICE("velocities",mesh.velocities);DEVICE("x_tilde",mesh.xTilta);DEVICE("targets",mesh.targetVert);
    DEVICE("temp_vertices",mesh.temp_double3Mem);DEVICE("external_force",mesh.fb);DEVICE("total_force",mesh.totalForce);
    DEVICE("direction",pcg_data.dx);
    auto& abd=m_abd_sim_data->device;
    DEVICE("abd_j",abd.unique_point_id_to_J);DEVICE("abd_point_mass",abd.unique_point_id_to_mass);
    DEVICE("abd_mass",abd.body_id_to_abd_mass);DEVICE("abd_mass_inv",abd.body_id_to_abd_mass_inv);
    DEVICE("abd_volume",abd.body_id_to_volume);DEVICE("abd_gravity",abd.body_id_to_abd_gravity);
    DEVICE("abd_tet_mass",abd.tet_id_to_abd_mass);DEVICE("abd_tet_gravity",abd.tet_id_to_abd_gravity_force);
    DEVICE("abd_mass_center",m_abd_system->body_mass_center);DEVICE("abd_scalar_mass",m_abd_system->body_mass);
    DEVICE("abd_q",abd.body_id_to_q);DEVICE("abd_q_prev",abd.body_id_to_q_prev);DEVICE("abd_q_tilde",abd.body_id_to_q_tilde);
    DEVICE("abd_q_v",abd.body_id_to_q_v);DEVICE("abd_q_temp",abd.body_id_to_q_temp);DEVICE("abd_dq",abd.body_id_to_dq);
    DEVICE("abd_force",abd.body_id_to_abd_force);DEVICE("abd_tet_force",abd.tet_id_to_abd_force);
    DEVICE("collision_pairs",_collisonPairs);DEVICE("cp_num",_cpNum);DEVICE("matrix_index",_MatIndex);
    DEVICE("ground_pairs",_environment_collisionPair);DEVICE("gp_num",_gpNum);
    DEVICE("friction_lambda",lambda_lastH_scalar);DEVICE("friction_coord",distCoord);DEVICE("friction_basis",tanBasis);
    DEVICE("friction_pairs",_collisonPairs_lastH);DEVICE("friction_index",_MatIndex_last);
    DEVICE("ground_friction_lambda",lambda_lastH_scalar_gd);DEVICE("ground_friction_pairs",_collisonPairs_lastH_gd);
    DEVICE("safe",toi.safe);DEVICE("trial",toi.trial);DEVICE("previous_trial",toi.previous_trial);
    DEVICE("safe_q",toi.safe_q);DEVICE("trial_q",toi.trial_q);DEVICE("contacts",toi.contacts);
    DEVICE("frame_friction_contacts",toi.frame_friction_contacts);
#undef DEVICE
    transfer("host_contacts",toi.host_contacts,false);transfer("host_vertex_mu",toi.host_vertex_mu,false);
    transfer("host_targets",mesh.host_target_vertices,false);
    auto scalar=[&](const char* name,auto& value)
    {
        using T=std::decay_t<decltype(value)>;
        if(restore)value=meta.at("scalars").at(name).get<T>();else meta["scalars"][name]=value;
    };
#define SCALAR(value) scalar(#value,value)
    SCALAR(Kappa);SCALAR(dHat);SCALAR(fDhat);SCALAR(dTol);SCALAR(Step);SCALAR(animation_subRate);SCALAR(animation_fullRate);
    SCALAR(toi.mu);SCALAR(toi.initial_mu);SCALAR(toi.delta);SCALAR(toi.self_count);SCALAR(toi.ground_count);
    SCALAR(toi.robust_port);SCALAR(toi.independent_friction_history);SCALAR(toi.frame_friction_frame_id);
    SCALAR(h_gpNum);SCALAR(h_gpNum_last);SCALAR(h_close_cpNum);SCALAR(h_close_gpNum);
#undef SCALAR
    for(int i=0;i<5;++i)
    {scalar(("cp_"+std::to_string(i)).c_str(),h_cpNum[i]);scalar(("cp_last_"+std::to_string(i)).c_str(),h_cpNum_last[i]);}
    if(restore)
    {
        if(restored_buffers!=payloads.size() || mesh.vertexes.size()!=vertexNum)
            throw std::runtime_error("Checkpoint buffer schema mismatch");
        total_Frames=meta.at("completed_physical_frames").get<int>();
        _vertexes=mesh.vertexes.data();_rest_vertexes=mesh.rest_vertexes.data();targetVert=mesh.targetVert.data();
        _moveDir=pcg_data.dx.data();h_ccd_cpNum=0;
        buildBVH();
        gipc::Json report={{"restored_buffers",restored_buffers},{"device_bytes_verified",true},
            {"context_verified",true},{"completed_physical_frames",total_Frames},{"checkpoint",directory},
            {"binary_id",binary_id},{"scope",meta.at("scope")}};
        std::ofstream out(fs::path(gipc::output_dir())/"checkpoint_restore.json");out<<report.dump(2);out.close();
        if(!out)throw std::runtime_error("Checkpoint restore report write failed");
    }
    else
    {
        std::ofstream out(folder/"checkpoint.json");out<<meta.dump(2);out.close();
        if(!out)throw std::runtime_error("Checkpoint metadata commit failed");
    }
}
