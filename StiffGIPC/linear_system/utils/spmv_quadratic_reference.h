#pragma once
// Host-only independent expansion of the *stored* SRBK operator. No Eigen,
// CUDA or reuse of the fused per-block scalar arithmetic is needed here.
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace gipc
{
inline long double spmv_quadratic_gamma(std::uint64_t operations)
{
    const long double ne=operations*static_cast<long double>(std::numeric_limits<double>::epsilon());
    if(ne>=0.5L)throw std::runtime_error("SpMV study roundoff budget too large");
    return ne/(1-ne);
}
struct SpmvQuadraticReference
{
    std::vector<long double> ap,absolute_terms,ap_bound;
    std::vector<std::uint64_t> term_count;
    long double quadratic=0,quadratic_absolute_terms=0;
    long double fused_scalar_bound=0,old_scalar_bound=0;
    std::uint64_t expanded_terms=0,lower_blocks=0,upper_blocks=0,diagonal_blocks=0;
};
inline SpmvQuadraticReference spmv_quadratic_reference(
    const std::vector<int>& rows,const std::vector<int>& cols,
    const std::vector<double>& blocks,const std::vector<double>& input)
{
    if(input.size()%3 || rows.size()!=cols.size() || blocks.size()!=rows.size()*9)
        throw std::runtime_error("SpMV study matrix/vector shape mismatch");
    SpmvQuadraticReference ref;
    ref.ap.resize(input.size());ref.absolute_terms.resize(input.size());
    ref.ap_bound.resize(input.size());ref.term_count.resize(input.size());
    const long double tiny=std::numeric_limits<double>::denorm_min();
    long double max_input=0;
    for(double value:input)if(!std::isfinite(value))
        throw std::runtime_error("SpMV study input is nonfinite");
    for(double value:input)max_input=std::max(max_input,std::abs(static_cast<long double>(value)));
    auto add=[&](size_t i,size_t j,double coefficient){
        if(!std::isfinite(coefficient))throw std::runtime_error("SpMV study matrix is nonfinite");
        const long double term=static_cast<long double>(coefficient)*input[j];
        ref.ap[i]+=term;ref.absolute_terms[i]+=std::abs(term);++ref.term_count[i];
        // Expand both orientations separately; do not use the candidate's
        // factor-of-two identity as the CPU reference.
        const long double q=static_cast<long double>(input[i])*term;
        if(!std::isfinite(term) || !std::isfinite(q))
            throw std::runtime_error("SpMV study reference product overflow");
        ref.quadratic+=q;ref.quadratic_absolute_terms+=std::abs(q);++ref.expanded_terms;
    };
    for(size_t k=0;k<rows.size();++k)
    {
        const int i=rows[k],j=cols[k];
        if(i<0 || j<0 || static_cast<size_t>(i)>=input.size()/3 || static_cast<size_t>(j)>=input.size()/3)
            throw std::runtime_error("SpMV study block index out of bounds");
        if(i<j)++ref.upper_blocks;else if(i>j)++ref.lower_blocks;else ++ref.diagonal_blocks;
        for(int row=0;row<3;++row)for(int col=0;col<3;++col)
        {
            const double a=blocks[k*9+col*3+row]; // exported column-major block
            add(static_cast<size_t>(i)*3+row,static_cast<size_t>(j)*3+col,a);
            if(i!=j)add(static_cast<size_t>(j)*3+col,static_cast<size_t>(i)*3+row,a);
        }
    }
    long double propagated_ap_error=0,dot_absolute_terms=0;
    for(size_t i=0;i<input.size();++i)
    {
        // Product + three-term local sum + row/atomic sum has at most c+6
        // rounding operations on any path. Include an equally conservative
        // CPU-reference budget even on MSVC where long double is FP64.
        const auto c=ref.term_count[i];
        if(!std::isfinite(ref.ap[i]) || !std::isfinite(ref.absolute_terms[i]))
            throw std::runtime_error("SpMV study reference sum overflow");
        ref.ap_bound[i]=2*spmv_quadratic_gamma(c+6)*ref.absolute_terms[i]+2*(c+6)*tiny;
        propagated_ap_error+=std::abs(static_cast<long double>(input[i]))*ref.ap_bound[i];
        dot_absolute_terms+=std::abs(static_cast<long double>(input[i])*ref.ap[i]);
    }
    const auto scalar_operations=ref.expanded_terms+input.size()+32;
    // Count every expanded scalar term, not just CUDA reduction depth. This
    // covers any legal atomic/reduction order, local factor-two evaluation,
    // the independent CPU expansion, and underflow in FP64 products.
    ref.fused_scalar_bound=2*spmv_quadratic_gamma(scalar_operations)*ref.quadratic_absolute_terms
        +2*scalar_operations*tiny*(1+max_input);
    ref.old_scalar_bound=ref.fused_scalar_bound+propagated_ap_error
        +2*spmv_quadratic_gamma(input.size()+2)*dot_absolute_terms;
    if(!std::isfinite(ref.quadratic) || !std::isfinite(ref.fused_scalar_bound) || !std::isfinite(ref.old_scalar_bound))
        throw std::runtime_error("SpMV study reference/bound overflow");
    return ref;
}
}
