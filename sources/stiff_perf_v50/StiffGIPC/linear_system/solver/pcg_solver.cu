#include <linear_system/solver/pcg_solver.h>
#include <linear_system/solver/pcg_guard.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <gipc/linear_stage.h>
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
    linear_stage("pcg_workspace_begin");

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    reduction_result.resize_discard(1);
    auto max_iter = static_cast<SizeT>(m_config.max_iter_ratio * b.size());
    SizeT iter=0;
    linear_stage("pcg_workspace_end");
    linear_stage("pcg_loop_begin");
    try { iter=m_config.conditional_graph ? pcg_graph(x,b,max_iter) : pcg(x,b,max_iter); }
    catch(const std::exception&)
    {
        Statistics::instance().at_current_frame()["newton"].back()["pcg"]["iteration_limit"]=false;
        if(const char* prefix=std::getenv("GIPC_MAS_AUDIT"))operator_audit(x,b,std::string(prefix)+"_failure");
        throw;
    }
    linear_stage("pcg_loop_end");
    linear_stage("pcg_audit_begin");
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
    linear_stage("pcg_audit_end");
    linear_stage("fixed_study_begin");
    if(std::getenv("GIPC_FIXED_STUDY_DIR"))fixed_system_study(x,b,max_iter,iter);
    linear_stage("fixed_study_end");
    if(const char* dir=std::getenv("GIPC_STATE_WINDOW_DIR");dir && total_Frames+1>=21 && total_Frames+1<=25)
    {
        std::vector<Float> values(x.size());
        CUDA_SAFE_CALL(cudaMemcpy(values.data(),x.data(),values.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        const std::string prefix=std::string(dir)+"/f"+std::to_string(total_Frames+1)+"_n"
            +std::to_string(Statistics::instance().at_current_frame()["newton"].size());
        std::ofstream file(prefix+"_x.bin",std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Float));
    }
    if(const char* prefix=std::getenv("GIPC_MAS_AUDIT");prefix &&
       (iter>=max_iter || (total_Frames==0 && Statistics::instance().at_current_frame()["newton"].size()==1)))
        operator_audit(x,b,std::string(prefix)+(iter>=max_iter?"_failure":"_initial"));
    return iter;
}


void PCGSolver::operator_audit(cudatool::DenseVectorView<Float> x,
                              cudatool::CDenseVectorView<Float> b,const std::string& prefix)
{
    Json audit={{"frame",total_Frames+1},{"direction",Statistics::instance().at_current_frame()["newton"].size()},
                {"system",snapshot_system(prefix)},{"probes",Json::array()},
                {"positive_samples_do_not_prove_spd",true}};
    const size_t n=b.size();
    std::vector<Float> rhs(n),solution(n),ax(n);
    CUDA_SAFE_CALL(cudaMemcpy(rhs.data(),b.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(solution.data(),x.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
    spmv(cudatool::CDenseVectorView<Float>{x.data(),static_cast<int>(n)},Ap.view());
    CUDA_SAFE_CALL(cudaMemcpy(ax.data(),Ap.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
    auto save=[&](const std::string& name,const std::vector<Float>& data)
    {
        std::ofstream file(prefix+"_"+name+".bin",std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(Float));
        if(!file)throw std::runtime_error("Failed to write operator probe");
    };
    save("x",solution);
    std::vector<std::vector<Float>> inputs(10,std::vector<Float>(n)),outputs;
    inputs[0]=rhs;
    for(size_t i=0;i<n;++i)inputs[1][i]=rhs[i]-ax[i];
    std::uint64_t random=0x93d72ae45ull;
    for(size_t k=2;k<inputs.size();++k)
        for(size_t i=0;i<n;++i)
        {
            random^=random<<13;random^=random>>7;random^=random<<17;
            inputs[k][i]=static_cast<double>(random>>11)*0x1.0p-53*2.-1.;
        }
    auto dot=[](const auto& a,const auto& b)
    {long double sum=0;for(size_t i=0;i<a.size();++i)sum+=static_cast<long double>(a[i])*b[i];return static_cast<double>(sum);};
    for(size_t k=0;k<inputs.size();++k)
    {
        CUDA_SAFE_CALL(cudaMemcpy(r.data(),inputs[k].data(),n*sizeof(Float),cudaMemcpyHostToDevice));
        apply_preconditioner(z.view(),cudatool::CDenseVectorView<Float>{r.data(),static_cast<int>(r.size())});
        std::vector<Float> first(n),again(n),av(n);
        CUDA_SAFE_CALL(cudaMemcpy(first.data(),z.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
        spmv(cudatool::CDenseVectorView<Float>{r.data(),static_cast<int>(r.size())},Ap.view());
        CUDA_SAFE_CALL(cudaMemcpy(av.data(),Ap.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
        Json record={{"index",k},{"input",k==0?"rhs":k==1?"true_residual":"seeded_random"},
            {"v_Mv",dot(inputs[k],first)},{"v_Av",dot(inputs[k],av)},
            {"norm_v",std::sqrt(dot(inputs[k],inputs[k]))},{"norm_Mv",std::sqrt(dot(first,first))},
            {"repeat_relative",Json::array()}};
        for(int repeat=0;repeat<3;++repeat)
        {
            apply_preconditioner(z.view(),cudatool::CDenseVectorView<Float>{r.data(),static_cast<int>(r.size())});
            CUDA_SAFE_CALL(cudaMemcpy(again.data(),z.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
            long double d=0;for(size_t i=0;i<n;++i)d+=static_cast<long double>(again[i]-first[i])*(again[i]-first[i]);
            record["repeat_relative"].push_back(std::sqrt(static_cast<double>(d)/std::max(dot(first,first),1e-300)));
        }
        save("v"+std::to_string(k),inputs[k]);save("Mv"+std::to_string(k),first);save("Av"+std::to_string(k),av);
        audit["probes"].push_back(record);outputs.push_back(std::move(first));
    }
    audit["bilinear_pairs"]=Json::array();
    for(size_t k=0;k+1<inputs.size();k+=2)
    {
        double a=dot(inputs[k],outputs[k+1]),b=dot(inputs[k+1],outputs[k]);
        double scale=std::sqrt(dot(inputs[k],inputs[k])*dot(outputs[k+1],outputs[k+1]))
            +std::sqrt(dot(inputs[k+1],inputs[k+1])*dot(outputs[k],outputs[k]));
        audit["bilinear_pairs"].push_back({{"u",k},{"v",k+1},{"u_Mv",a},{"v_Mu",b},
            {"scaled_asymmetry",std::abs(a-b)/std::max(scale,1e-300)}});
    }
    std::vector<Float> after(n);
    CUDA_SAFE_CALL(cudaMemcpy(after.data(),x.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
    audit["solution_unchanged_bitwise"]=std::memcmp(after.data(),solution.data(),n*sizeof(Float))==0;
    std::ofstream(prefix+"_audit.json")<<audit.dump(2);
    if(!audit["solution_unchanged_bitwise"].get<bool>())throw std::runtime_error("Operator audit modified solution");
}

void PCGSolver::fail_pcg(int code,Float rho,Float curvature)
{
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    info["breakdown"]=pcg_error_name(code);info["breakdown_rho"]=rho;info["breakdown_curvature"]=curvature;
    info["iteration_limit"]=false;
    throw std::runtime_error(std::string("PCG numerical breakdown: ")+pcg_error_name(code));
}
bool PCGSolver::check_host_rho(Float rho)
{
    if(int code=pcg_rho_error(rho))fail_pcg(code,rho,0);
    if(rho!=0)return false;
    std::vector<Float> values(r.size());
    CUDA_SAFE_CALL(cudaMemcpy(values.data(),r.buffer_view().data(),values.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    for(Float v:values)if(v!=0)fail_pcg(3,rho,0);
    return true;
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
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    info["execution"]="host";info["rho_initial"]=rz0;info["iterations"]=0;
    if(check_host_rho(rz)){info["zero_residual"]=true;return 0;}

    for(k = 1; k < max_iter; ++k)
    {
        info["iterations"]=k;
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

            if(int code=pcg_curvature_error(dot_res))fail_pcg(code,rz,dot_res);
            alpha = rz / dot_res;
            if(!std::isfinite(alpha))fail_pcg(6,rz,dot_res);
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

        if(check_host_rho(rz_new)){info["zero_residual"]=true;return k;}
        beta = rz_new / rz;
        if(!std::isfinite(beta))fail_pcg(7,rz_new,0);

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

    info["rho_stop"]=rz;
    return k;
}

}  // namespace gipc

#include "pcg_graph_impl.inl"
#include "pcg_guard_fixture.inl"
#include "pcg_fixed_study.inl"
