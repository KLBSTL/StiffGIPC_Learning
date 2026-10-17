#pragma once
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace gipc {
struct NativeBalancedRejected : std::runtime_error { using std::runtime_error::runtime_error; };
inline void balanced_require(bool value,const char* message) {
    if(!value)throw NativeBalancedRejected(message);
}
// Only screens whether to run the authoritative original-A/MAS quality check.
// Never accepts a solution using a recursive residual.
inline bool native_quality_check_due(int iteration,int last_check,double rr,double bb,int limit) {
    balanced_require(iteration>0&&last_check>=0&&last_check<=iteration&&limit>=iteration,
                     "Invalid native quality schedule");
    balanced_require(std::isfinite(rr)&&rr>=0&&std::isfinite(bb)&&bb>0,
                     "Invalid recursive residual screen");
    return last_check==0||iteration==limit||iteration-last_check>=8||rr<=.012*.012*bb;
}
inline bool native_balanced_study_enabled() {
    const char* value=std::getenv("GIPC_FIXED_NATIVE_BALANCED_STUDY");
    if(!value||std::string(value)=="0")return false;
    if(std::string(value)=="1")return true;
    throw std::runtime_error("GIPC_FIXED_NATIVE_BALANCED_STUDY must be 0 or 1");
}
// Sorted retained update indices; scalar rows refer to every original update.
inline Eigen::MatrixXd native_stat_coefficients(const std::vector<int>& ids,
    const std::vector<std::array<double,3>>& rows,int rank=8) {
    balanced_require(rank==8&&ids.size()>=8&&!rows.empty(),"Source rank eight unavailable");
    std::vector<double> suffix(rows.size()+1,0.);
    for(size_t i=rows.size();i-->0;) {
        const auto& a=rows[i];
        balanced_require(std::isfinite(a[0])&&std::isfinite(a[1])&&std::isfinite(a[2])
            &&a[0]>0&&a[1]>0&&a[2]>0&&std::abs(a[0]*a[2]-a[1])<1e-10*a[1],"Invalid source CG context");
        suffix[i]=suffix[i+1]+a[0]*a[1];
        balanced_require(std::isfinite(suffix[i]),"Nonfinite source CG energy");
    }
    for(size_t i=0;i<ids.size();++i)
        balanced_require(ids[i]>=0&&size_t(ids[i])<rows.size()&&(i==0||ids[i]>ids[i-1]),"Invalid retained source indices");
    const int w=static_cast<int>(ids.size());Eigen::MatrixXd H(w,w);
    for(int j=0;j<w;++j)for(int i=0;i<w;++i)H(i,j)=suffix[std::max(ids[i],ids[j])];
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(H);
    balanced_require(eigen.info()==Eigen::Success&&eigen.eigenvalues().allFinite(),"Source eigensolve failed");
    const auto& ev=eigen.eigenvalues();
    balanced_require(ev[w-1]>0&&ev[w-8]>1e-10*ev[w-1]&&ev[0]>=-1e-10*ev[w-1],"Source rank eight unavailable");
    Eigen::MatrixXd U(w,8);
    for(int j=0;j<8;++j)U.col(j)=eigen.eigenvectors().col(w-1-j)/std::sqrt(ev[w-1-j]);
    return U;
}
inline Eigen::MatrixXd native_coarse_factor(const Eigen::MatrixXd& raw,double& condition) {
    balanced_require(raw.rows()==8&&raw.cols()==8&&raw.allFinite(),"Invalid coarse matrix");
    balanced_require((raw-raw.transpose()).norm()<1e-8*std::max(raw.norm(),1e-300),"Coarse matrix asymmetry rejected");
    const Eigen::MatrixXd E=.5*(raw+raw.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(E,Eigen::EigenvaluesOnly);
    balanced_require(eigen.info()==Eigen::Success&&eigen.eigenvalues().allFinite(),"Coarse eigensolve failed");
    const auto& ev=eigen.eigenvalues();
    balanced_require(ev[7]>0&&ev[0]>1e-12*ev[7],"Coarse rank/conditioning rejected");
    condition=ev[7]/ev[0];Eigen::LLT<Eigen::MatrixXd> llt(E);
    balanced_require(llt.info()==Eigen::Success,"Coarse Cholesky rejected");
    return llt.matrixL().toDenseMatrix();
}
} // namespace gipc
