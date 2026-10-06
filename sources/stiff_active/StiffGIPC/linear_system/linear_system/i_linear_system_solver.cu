#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/global_linear_system.h>

namespace gipc
{
IterativeSolver::~IterativeSolver() {}

std::string IterativeSolver::spmv_quadratic_unavailable_reason(SizeT count) const
{return m_system->spmv_quadratic_unavailable_reason(count);}
SizeT IterativeSolver::spmv_quadratic_partial_count() const
{return m_system->spmv_quadratic_partial_count();}
void IterativeSolver::spmv_quadratic(cudatool::CDenseVectorView<Float> x,
    cudatool::DenseVectorView<Float> y,Float* partials)
{m_system->spmv_quadratic(x,y,partials);}

void IterativeSolver::spmv(Float                         a,
                           cudatool::CDenseVectorView<Float> x,
                           Float                         b,
                           cudatool::DenseVectorView<Float>  y)
{
    m_system->spmv(a, x, b, y);
}

void IterativeSolver::apply_preconditioner(cudatool::DenseVectorView<Float> z,
                                           cudatool::CDenseVectorView<Float> r) const
{
    m_system->apply_preconditioner(z, r);
}


std::string IterativeSolver::mas_fused_dot_unavailable_reason(SizeT count) const
{return m_system->mas_fused_dot_unavailable_reason(count);}
SizeT IterativeSolver::mas_fused_dot_partial_count(SizeT count) const
{return m_system->mas_fused_dot_partial_count(count);}
void IterativeSolver::apply_preconditioner_fused_dot(cudatool::DenseVectorView<Float> z,
    cudatool::CDenseVectorView<Float> r,Float* partials,bool prepared_only) const
{m_system->apply_preconditioner_fused_dot(z,r,partials,prepared_only);}
void IterativeSolver::mas_dot_scratch(const std::function<void(void*,size_t)>& visitor) const
{m_system->mas_dot_scratch(visitor);}

bool IterativeSolver::fused_diag_update_available() const
{
    return m_system->fused_diag_update_available();
}

void IterativeSolver::fused_diag_update(
    cudatool::DenseVectorView<Float> x, cudatool::DenseVectorView<Float> r,
    cudatool::CDenseVectorView<Float> p, cudatool::CDenseVectorView<Float> ap,
    const Float* alpha, cudatool::DenseVectorView<Float> z) const
{
    m_system->fused_diag_update(x, r, p, ap, alpha, z);
}

cudatool::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_context;
}
std::vector<std::uintptr_t> IterativeSolver::graph_signature() const
{
    return m_system->graph_signature();
}
Json IterativeSolver::snapshot_system(const std::string& prefix) const
{
    return m_system->snapshot_system(prefix);
}
}  // namespace gipc
