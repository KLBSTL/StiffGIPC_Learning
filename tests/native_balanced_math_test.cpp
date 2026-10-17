#include <linear_system/solver/native_balanced_math.h>
#include <iostream>
#include <limits>
using namespace gipc;
int main() {
    try {
        std::vector<std::array<double,3>> rows;
        for(int i=0;i<20;++i){double rho=std::pow(.8,i),alpha=1./(i+1);rows.push_back({alpha,rho,rho/alpha});}
        std::vector<int> ids{0,1,2,3,4,5,6,7,8,9,10,11};
        const auto U=native_stat_coefficients(ids,rows);Eigen::MatrixXd H(12,12);
        for(int i=0;i<12;++i)for(int j=0;j<12;++j){H(i,j)=0;for(int k=std::max(i,j);k<20;++k)H(i,j)+=rows[k][0]*rows[k][1];}
        if((U.transpose()*H*U-Eigen::MatrixXd::Identity(8,8)).norm()>1e-10)return 1;
        int rejected=0;
        try{native_stat_coefficients({0,1,2,3,4,5},rows);}catch(const NativeBalancedRejected&){++rejected;}
        try{auto bad=rows;bad[3][2]*=2;native_stat_coefficients(ids,bad);}catch(const NativeBalancedRejected&){++rejected;}
        try{auto bad=ids;bad[4]=bad[3];native_stat_coefficients(bad,rows);}catch(const NativeBalancedRejected&){++rejected;}
        double condition=0;Eigen::MatrixXd E=Eigen::MatrixXd::Identity(8,8);E(0,0)=2;
        const auto L=native_coarse_factor(E,condition);
        if((L*L.transpose()-E).norm()>1e-12||std::abs(condition-2)>1e-12)return 2;
        try{auto bad=E;bad(2,2)=0;native_coarse_factor(bad,condition);}catch(const NativeBalancedRejected&){++rejected;}
        try{auto bad=E;bad(1,2)=.01;native_coarse_factor(bad,condition);}catch(const NativeBalancedRejected&){++rejected;}
        if(rejected!=5)return 3;
        if(!native_quality_check_due(59,0,1.,1.,2000)
           ||native_quality_check_due(60,59,1.,1.,2000)
           ||!native_quality_check_due(67,59,1.,1.,2000)
           ||!native_quality_check_due(60,59,.012*.012,1.,2000)
           ||!native_quality_check_due(2000,1999,1.,1.,2000)
           ||!native_quality_check_due(60,59,0.,1.,2000))return 5;
        try{native_quality_check_due(60,61,1.,1.,2000);return 6;}catch(const NativeBalancedRejected&){}
        try{native_quality_check_due(60,59,std::numeric_limits<double>::quiet_NaN(),1.,2000);return 7;}
            catch(const NativeBalancedRejected&){}
        try{native_quality_check_due(60,59,1.,0.,2000);return 8;}catch(const NativeBalancedRejected&){}
        std::cout<<"native balanced coefficient, factor and rejection contracts passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}
}
