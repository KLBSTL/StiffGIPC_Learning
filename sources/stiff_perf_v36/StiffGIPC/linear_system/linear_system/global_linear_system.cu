#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <linear_system/preconditioner/diag_preconditioner.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/timer.h>
#include <cstring>
#include <cmath>
#include <fstream>
#include <gipc/statistics.h>
#include <linear_system/preconditioner/abd_preconditioner.h>

namespace gipc
{
bool GlobalLinearSystem::build_linear_system()
{
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }


    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);

    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    convert_new();

    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);

    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}


IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    CT_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

Json GlobalLinearSystem::snapshot_system(const std::string& prefix) const
{
    Json result;
    auto save=[&](const std::string& name,const void* device,size_t bytes)
    {
        std::vector<unsigned char> data(bytes);
        if(bytes)CUDA_SAFE_CALL(cudaMemcpy(data.data(),device,bytes,cudaMemcpyDeviceToHost));
        std::uint64_t hash=14695981039346656037ull;
        for(auto byte:data){hash^=byte;hash*=1099511628211ull;}
        result["buffers"][name]={{"bytes",bytes},{"fnv1a64",hash}};
        if(!prefix.empty())
        {
            std::ofstream file(prefix+"_"+name+".bin",std::ios::binary);
            file.write(reinterpret_cast<const char*>(data.data()),bytes);
            if(!file)throw std::runtime_error("Failed to write fixed-system snapshot");
        }
    };
    auto* a=gipc_global_triplet;
    const size_t n=a->h_unique_key_number;
    save("rows",a->block_row_indices(),n*sizeof(int));
    save("cols",a->block_col_indices(),n*sizeof(int));
    save("values",a->block_values(),n*sizeof(Matrix3x3));
    save("rhs",m_b.buffer_view().data(),m_b.size()*sizeof(Float));
    result["dofs"]=m_b.size();result["blocks"]=n;
    result["matrix_format"]="symmetric upper block COO, int32 indices, float64 column-major 3x3";
    bool complete=true;
    if(m_global_preconditioner)
    {
        auto* diagonal=dynamic_cast<DiagPreconditioner*>(m_global_preconditioner.get());
        if(diagonal)
        {
            auto key=diagonal->graph_signature();
            save("diag_inverse",reinterpret_cast<const void*>(key[0]),key[1]*sizeof(Matrix3x3));
        }
        else complete=false;
    }
    result["local_preconditioners"]=Json::array();
    for(size_t i=0;i<m_local_preconditioners.size();++i)
    {
        const auto& p=m_local_preconditioners[i];
        auto* abd=dynamic_cast<ABDPreconditioner*>(p.get());
        auto key=p->graph_signature();
        if(abd)
        {
            const std::string name="abd_inverse_"+std::to_string(i);
            save(name,reinterpret_cast<const void*>(key[0]),key[1]*sizeof(Matrix12x12));
            result["local_preconditioners"].push_back({{"kind","abd"},{"buffer",name},{"count",key[1]},{"offset",key[2]}});
        }
        else
        {
            complete=false;
            result["local_preconditioners"].push_back({{"kind","MAS_in_process_only"},{"signature",key}});
        }
    }
    result["preconditioner_export_complete"]=complete;
    if(!prefix.empty())std::ofstream(prefix+"_meta.json")<<result.dump(2);
    return result;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    if(!success)
        return 0;
    CT_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    SizeT iter;
    try { iter = m_solver->solve(m_x, m_b); }
    catch(...)
    {
        if(const char* prefix=std::getenv("GIPC_FAILURE_SYSTEM"))snapshot_system(prefix);
        throw;
    }
    if(const char* prefix=std::getenv("GIPC_FAILURE_SYSTEM");prefix &&
       Statistics::instance().at_current_frame()["newton"].back()["pcg"].value("iteration_limit",false))
        snapshot_system(prefix);
    distribute_solution();
    return iter;
}

Json GlobalLinearSystem::audit_reassembly(const std::function<void()>& assemble_derivatives)
{
    auto read=[](auto* device,size_t n)
    {
        using T=std::remove_cv_t<std::remove_pointer_t<decltype(device)>>;
        std::vector<T> values(n);
        if(n)CUDA_SAFE_CALL(cudaMemcpy(values.data(),device,n*sizeof(T),cudaMemcpyDeviceToHost));
        return values;
    };
    auto difference=[](const auto& first,const auto& second)
    {
        if(first.size()!=second.size())return Json{{"size_identical",false}};
        double squared=0,norm=0,maximum=0;size_t unequal=0;
        for(size_t i=0;i<first.size();++i)
        {
            const double delta=second[i]-first[i];
            squared+=delta*delta;norm+=first[i]*first[i];maximum=std::max(maximum,std::abs(delta));
            unequal+=std::memcmp(&first[i],&second[i],sizeof(first[i]))!=0;
        }
        return Json{{"size_identical",true},{"relative_difference",norm>0?std::sqrt(squared/norm):std::sqrt(squared)},
                    {"maximum_absolute_difference",maximum},{"bitwise_different_values",unequal}};
    };
    auto* matrix=gipc_global_triplet;
    const size_t count=matrix->h_unique_key_number;
    const auto rows=read(matrix->block_row_indices(),count),cols=read(matrix->block_col_indices(),count);
    const auto values=read(reinterpret_cast<const double*>(matrix->block_values()),count*9);
    const auto rhs=read(m_b.buffer_view().data(),m_b.size());
    const auto original_x=read(m_x.buffer_view().data(),m_x.size());
    assemble_derivatives();
    if(!build_linear_system())throw std::runtime_error("Diagnostic reassembly produced an empty system");
    const size_t repeated_count=matrix->h_unique_key_number;
    const auto repeated_rows=read(matrix->block_row_indices(),repeated_count);
    const auto repeated_cols=read(matrix->block_col_indices(),repeated_count);
    const bool structure=rows==repeated_rows && cols==repeated_cols;
    Json audit={{"blocks_before",count},{"blocks_after",repeated_count},{"structure_identical",structure},
                {"rhs",difference(rhs,read(m_b.buffer_view().data(),m_b.size()))},
                {"solution_buffer",difference(original_x,read(m_x.buffer_view().data(),m_x.size()))}};
    if(structure)audit["matrix"]=difference(values,read(reinterpret_cast<const double*>(matrix->block_values()),count*9));
    if(original_x.size()!=m_x.size())throw std::runtime_error("Diagnostic reassembly changed solution dimensions");
    if(!original_x.empty())CUDA_SAFE_CALL(cudaMemcpy(m_x.buffer_view().data(),original_x.data(),original_x.size()*sizeof(Float),cudaMemcpyHostToDevice));
    return audit;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

bool GlobalLinearSystem::fused_diag_update_available() const
{
    return dynamic_cast<DiagPreconditioner*>(m_global_preconditioner.get()) != nullptr;
}

void GlobalLinearSystem::fused_diag_update(
    cudatool::DenseVectorView<Float> x, cudatool::DenseVectorView<Float> r,
    cudatool::CDenseVectorView<Float> p, cudatool::CDenseVectorView<Float> ap,
    const Float* alpha, cudatool::DenseVectorView<Float> z)
{
    auto* diagonal = dynamic_cast<DiagPreconditioner*>(m_global_preconditioner.get());
    CT_ASSERT(diagonal, "Fused update requires the global diagonal preconditioner");
    diagonal->update_and_apply(x, r, p, ap, alpha, z);
    // Preserve every local override (including the ABD 12x12 blocks) and order.
    const cudatool::CDenseVectorView<Float> residual{
        r.buffer_view().data(), static_cast<int>(r.size())};
    for(auto& local : m_local_preconditioners)
        local->do_apply(residual, z);
}

void GlobalLinearSystem::apply_preconditioner(cudatool::DenseVectorView<Float> z,
                                              cudatool::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
        CUDA_SAFE_CALL(cudaMemcpyAsync(z.buffer_view().data(),r.buffer_view().data(),
            r.size()*sizeof(Float),cudaMemcpyDeviceToDevice,cudaStreamPerThread));

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}


void GlobalLinearSystem::convert_new()
{
    size_t final_triplet_count =
        static_cast<size_t>(gipc_global_triplet->global_triplet_offset);
    gipc_global_triplet->ensure_triplet_capacity(2 * final_triplet_count);
    gipc_global_triplet->resize_conversion_scratch(final_triplet_count);

    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);

    // Conversion has compacted the matrix into the prefix. Keep the
    // allocation, but expose only live unique blocks to downstream users.
    gipc_global_triplet->resize_triplets(
        static_cast<size_t>(gipc_global_triplet->h_unique_key_number));
    //#ifndef SymGH
    //    m_converter.ge2sym(*gipc_global_triplet);
    //#endif
}


std::vector<std::uintptr_t> GlobalLinearSystem::graph_signature() const
{
    auto* a = gipc_global_triplet;
    std::vector<std::uintptr_t> key = {
        reinterpret_cast<std::uintptr_t>(a->block_values()),
        reinterpret_cast<std::uintptr_t>(a->block_row_indices()),
        reinterpret_cast<std::uintptr_t>(a->block_col_indices()),
        static_cast<std::uintptr_t>(a->h_unique_key_number)};
    for(auto n : m_rhs_count_per_subsystem) key.push_back(n);
    for(auto n : m_rhs_offset_per_subsystem) key.push_back(n);
    auto append = [&](const IPreconditioner& p) {
        auto part = p.graph_signature();
        key.push_back(part.size());
        key.insert(key.end(), part.begin(), part.end());
    };
    if(m_global_preconditioner) append(*m_global_preconditioner);
    for(const auto& p : m_local_preconditioners) append(*p);
    return key;
}

void GlobalLinearSystem::spmv(Float                             a,
                              cudatool::CDenseVectorView<Float> x,
                              Float                             b,
                              cudatool::DenseVectorView<Float>  y)
{

    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                gipc_global_triplet->h_unique_key_number,
                                x,
                                b,
                                y);
}
}  // namespace gipc
