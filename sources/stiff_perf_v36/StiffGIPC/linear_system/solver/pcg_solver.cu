#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <cuda_tools/cuda_cub_wrappers.h>
#include <cub/block/block_reduce.cuh>
#include <cstdlib>
#include <cub/device/device_reduce.cuh>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

extern int total_Frames;



__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    int valid_items = min(numbers - idof, static_cast<int>(blockDim.x));
    double temp = idx < numbers ? a[idx] * b[idx] : 0.0;

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage storage;
    temp = BlockReduce(storage).Sum(temp, valid_items);
    if(threadIdx.x == 0)
        squeue[blockIdx.x] = temp;
}



__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}

__global__ void audit_residual(double* r,const double* b,const double* ax,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)r[i]=b[i]-ax[i];
}


double My_PCG_General_v_v_Reduction_Algorithm(double*       partials,
                                              const double* A,
                                              const double* B,
                                              double*       result_output,
                                              int           vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    PCG_vdv_Reduction<<<blockNum, threadNum>>>(partials, A, B, numbers);
    cudatool::DeviceReduce().Sum(partials, result_output, blockNum);

    double result = 0.0;
    CUDA_SAFE_CALL(
        cudaMemcpy(&result, result_output, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}
SizeT PCGSolver::solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    reduction_result.resize_discard(1);
    auto max_iter = static_cast<SizeT>(m_config.max_iter_ratio * b.size());
    auto iter = m_config.conditional_graph ? pcg_graph(x, b, max_iter) : pcg(x, b, max_iter);
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    info["iteration_limit"]=iter>=max_iter;
    if(m_config.conditional_graph && std::getenv("GIPC_AUDIT_GRAPH")
       && std::string(std::getenv("GIPC_AUDIT_GRAPH"))=="1")
    {
        // Solve the exact same A/b/preconditioner with the host loop. This is
        // a quality-only run; all duplicate work is included in its timing.
        std::vector<Float> graph_x(x.size()),host_x(x.size());
        CUDA_SAFE_CALL(cudaMemcpy(graph_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        x.buffer_view().fill(0);
        auto host_iter=pcg(x,b,max_iter);
        CUDA_SAFE_CALL(cudaMemcpy(host_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        double difference=0,reference=0;
        for(size_t i=0;i<host_x.size();++i){double d=host_x[i]-graph_x[i];difference+=d*d;reference+=host_x[i]*host_x[i];}
        info["same_system_host_iterations"]=host_iter;
        info["same_system_relative_solution_difference"]=reference>0?std::sqrt(difference/reference):std::sqrt(difference);
        info["same_system_solution_difference_norm"]=std::sqrt(difference);
        info["same_system_host_solution_norm"]=std::sqrt(reference);
        x.buffer_view().fill(0);
        auto repeat_iter=pcg(x,b,max_iter);
        std::vector<Float> repeat_x(x.size());
        CUDA_SAFE_CALL(cudaMemcpy(repeat_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        double repeat_difference=0;
        for(size_t i=0;i<host_x.size();++i){double d=host_x[i]-repeat_x[i];repeat_difference+=d*d;}
        info["same_system_host_repeat_iterations"]=repeat_iter;
        info["same_system_host_repeat_relative_difference"]=reference>0?std::sqrt(repeat_difference/reference):std::sqrt(repeat_difference);
        CUDA_SAFE_CALL(cudaMemcpy(x.buffer_view().data(),graph_x.data(),x.size()*sizeof(Float),cudaMemcpyHostToDevice));
    }
    if(const char* from=std::getenv("GIPC_PCG_REPLAY_FROM_FRAME"); from
       && total_Frames+1>=std::stoi(from)
       && Statistics::instance().at_current_frame()["newton"].size()==1)
    {
        // Keep the assembled system fixed and restore the primary solution.
        // Duplicate solves are diagnostics and never count as useful work.
        const auto primary_info=info;
        std::vector<Float> primary_x(x.size()),original_b(b.size());
        CUDA_SAFE_CALL(cudaMemcpy(primary_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(original_b.data(),b.buffer_view().data(),b.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        auto residual=[&]()
        {
            spmv(cudatool::CDenseVectorView<Float>{x.buffer_view().data(),static_cast<int>(x.size())},Ap.view());
            audit_residual<<<(b.size()+255)/256,256>>>(r.buffer_view().data(),b.buffer_view().data(),Ap.buffer_view().data(),b.size());
            const double rr=My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),r.buffer_view().data(),r.buffer_view().data(),reduction_result.data(),b.size());
            const double bb=My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),b.buffer_view().data(),b.buffer_view().data(),reduction_result.data(),b.size());
            return bb>0?std::sqrt(rr/bb):std::sqrt(rr);
        };
        auto difference=[](const std::vector<Float>& value,const std::vector<Float>& reference)
        {
            double squared=0,norm=0,maximum=0;
            size_t bitwise_different=0;
            for(size_t i=0;i<value.size();++i)
            {
                const double delta=value[i]-reference[i];
                squared+=delta*delta;norm+=reference[i]*reference[i];
                maximum=std::max(maximum,std::abs(delta));
                bitwise_different+=std::memcmp(&value[i],&reference[i],sizeof(Float))!=0;
            }
            return Json{{"relative_solution_difference",norm>0?std::sqrt(squared/norm):std::sqrt(squared)},
                        {"maximum_absolute_difference",maximum},{"bitwise_different_dofs",bitwise_different}};
        };
        Json audit={{"frame",total_Frames+1},{"dofs",b.size()},
                    {"primary_execution",m_config.conditional_graph?"conditional_graph":"host"},
                    {"primary_iterations",iter},{"primary_iteration_limit",iter>=max_iter},
                    {"primary_true_relative_residual",residual()},
                    {"same_execution_runs",Json::array()},{"host_reference_runs",Json::array()}};
        auto repeat_operator=[&](auto operation,cudatool::DenseVectorView<Float> output)
        {
            operation();
            std::vector<Float> first(output.size());
            CUDA_SAFE_CALL(cudaMemcpy(first.data(),output.buffer_view().data(),output.size()*sizeof(Float),cudaMemcpyDeviceToHost));
            auto records=Json::array();
            for(int repeat=0;repeat<2;++repeat)
            {
                operation();
                std::vector<Float> repeated(output.size());
                CUDA_SAFE_CALL(cudaMemcpy(repeated.data(),output.buffer_view().data(),output.size()*sizeof(Float),cudaMemcpyDeviceToHost));
                records.push_back(difference(repeated,first));
            }
            return records;
        };
        audit["operator_replays"]={
            {"spmv",repeat_operator([&](){spmv(cudatool::CDenseVectorView<Float>{x.buffer_view().data(),static_cast<int>(x.size())},Ap.view());},Ap.view())},
            {"preconditioner",repeat_operator([&](){apply_preconditioner(z.view(),b);},z.view())}};
        for(int repeat=0;repeat<2;++repeat)
        {
            x.buffer_view().fill(0);
            const auto repeated_iter=m_config.conditional_graph?pcg_graph(x,b,max_iter):pcg(x,b,max_iter);
            std::vector<Float> repeated_x(x.size());
            CUDA_SAFE_CALL(cudaMemcpy(repeated_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
            auto record=difference(repeated_x,primary_x);
            record["iterations"]=repeated_iter;record["iteration_limit"]=repeated_iter>=max_iter;
            record["true_relative_residual"]=residual();
            audit["same_execution_runs"].push_back(std::move(record));
        }
        if(m_config.conditional_graph)
        {
            std::vector<Float> first_host;
            for(int repeat=0;repeat<2;++repeat)
            {
                x.buffer_view().fill(0);
                const auto host_iter=pcg(x,b,max_iter);
                std::vector<Float> host_x(x.size());
                CUDA_SAFE_CALL(cudaMemcpy(host_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
                auto record=difference(host_x,primary_x);
                record["iterations"]=host_iter;record["iteration_limit"]=host_iter>=max_iter;
                record["true_relative_residual"]=residual();
                if(repeat==0)first_host=host_x;
                record["relative_to_first_host"]=difference(host_x,first_host)["relative_solution_difference"];
                audit["host_reference_runs"].push_back(std::move(record));
            }
        }
        std::vector<Float> current_b(b.size()),restored_x(x.size());
        CUDA_SAFE_CALL(cudaMemcpy(current_b.data(),b.buffer_view().data(),b.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(x.buffer_view().data(),primary_x.data(),x.size()*sizeof(Float),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(restored_x.data(),x.buffer_view().data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        audit["rhs_unchanged_bitwise"]=std::memcmp(original_b.data(),current_b.data(),b.size()*sizeof(Float))==0;
        audit["primary_solution_restored_bitwise"]=std::memcmp(primary_x.data(),restored_x.data(),x.size()*sizeof(Float))==0;
        if(!audit["rhs_unchanged_bitwise"].get<bool>() || !audit["primary_solution_restored_bitwise"].get<bool>())
            throw std::runtime_error("Fixed-system PCG replay changed its RHS or failed to restore the primary solution");
        info=primary_info;
        info["fixed_system_replay"]=std::move(audit);
    }
    if(const char* audit=std::getenv("GIPC_AUDIT_PCG"); audit && audit[0]=='1')
    {
        spmv(cudatool::CDenseVectorView<Float>{x.buffer_view().data(),static_cast<int>(x.size())},Ap.view());
        audit_residual<<<(b.size()+255)/256,256>>>(r.buffer_view().data(),b.buffer_view().data(),Ap.buffer_view().data(),b.size());
        double rr=My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),r.buffer_view().data(),r.buffer_view().data(),reduction_result.data(),b.size());
        double bb=My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),b.buffer_view().data(),b.buffer_view().data(),reduction_result.data(),b.size());
        info["true_relative_residual"]=bb>0?std::sqrt(rr/bb):std::sqrt(rr);
    }

    if(std::getenv("GIPC_TOI_DIAGNOSTIC_LOG"))
    {
        double bx=My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),b.buffer_view().data(),x.buffer_view().data(),reduction_result.data(),b.size());
        info["gradient_dot_newton_step"]=-bx;
        if(std::getenv("GIPC_TOI_CURVATURE_FROM_FRAME"))
        {
            spmv(cudatool::CDenseVectorView<Float>{x.buffer_view().data(),static_cast<int>(x.size())},Ap.view());
            double xax=My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),x.buffer_view().data(),Ap.buffer_view().data(),reduction_result.data(),b.size());
            info["assembled_direction_curvature"]=xax;
        }
    }
    if(std::getenv("GIPC_FIXED_STUDY_DIR"))fixed_system_study(x,b,max_iter,iter);
    return iter;
}


SizeT PCGSolver::pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    {
        //Timer timer{"dot"};
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    reduction_result.data(),
                                                    z.size());
    }

    p.copy_from(z);
    rz0 = rz;

    for(k = 1; k < max_iter; ++k)
    {
        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            Float dot_res =
                My_PCG_General_v_v_Reduction_Algorithm(z.buffer_view().data(),
                                                       p.buffer_view().data(),
                                                       Ap.buffer_view().data(),
                                                       reduction_result.data(),
                                                       z.size());

            alpha = rz / dot_res;
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
        }

        if(diagnostic_fixed_iterations>0 ? k>=diagnostic_fixed_iterations :
           std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            //Timer timer{"dot"};
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            reduction_result.data(),
                                                            z.size());
        }

        beta = rz_new / rz;

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
        }

        rz = rz_new;
    }

    return k;
}

}  // namespace gipc

#include "pcg_graph_impl.inl"
#include "pcg_fixed_study.inl"
