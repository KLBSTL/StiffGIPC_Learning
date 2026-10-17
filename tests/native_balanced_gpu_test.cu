#include <linear_system/solver/native_balanced_gpu.cuh>
#include <iostream>
#include <cstring>
using namespace gipc;
__global__ void fixture_diagonal(int n,double shift,const double* x,double* y) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=(shift+.03*i)*x[i];
}
void projection_contract() {
    for(int n:{1,31,32,33,255,256,257,1025}) {
        Eigen::VectorXd r(n),a(8),c(8);Eigen::MatrixXd W(n,8),Z(n,8);
        for(int j=0;j<8;++j){a[j]=.13*(j-3);c[j]=.07*(4-j);}
        for(int i=0;i<n;++i){r[i]=std::sin(.17*i)+.3;
            for(int j=0;j<8;++j){W(i,j)=std::cos(.013*(i+1)*(j+1));Z(i,j)=std::sin(.021*(i+1)*(j+2));}}
        BalancedBuffer<double> dr(n),dw(size_t(n)*8),dz(size_t(n)*8),da(8),dc(8),out(n);
        auto upload=[](double* d,const double* h,size_t count){balanced_cuda(cudaMemcpy(d,h,count*sizeof(double),cudaMemcpyHostToDevice));};
        auto unchanged=[](const double* d,const double* h,size_t count){std::vector<double> v(count);
            balanced_cuda(cudaMemcpy(v.data(),d,count*sizeof(double),cudaMemcpyDeviceToHost));
            if(std::memcmp(v.data(),h,count*sizeof(double))!=0)throw std::runtime_error("Projection input changed");};
        upload(dr.value,r.data(),n);upload(dw.value,W.data(),size_t(n)*8);upload(dz.value,Z.data(),size_t(n)*8);
        upload(da.value,a.data(),8);upload(dc.value,c.data(),8);
        balanced_input_projection<<<(n+255)/256,256,0,cudaStreamPerThread>>>(n,dr.value,dw.value,da.value,out.value);
        balanced_cuda(cudaGetLastError());Eigen::VectorXd value(n);
        balanced_cuda(cudaMemcpy(value.data(),out.value,n*sizeof(double),cudaMemcpyDeviceToHost));
        const Eigen::VectorXd input_expected=r-W*a;
        if(!value.allFinite()||(value-input_expected).norm()/std::max(input_expected.norm(),1e-300)>2e-12)
            throw std::runtime_error("Fused input projection differs from CPU formula");
        upload(out.value,r.data(),n);
        balanced_output_projection<<<(n+255)/256,256,0,cudaStreamPerThread>>>(n,dz.value,da.value,dc.value,out.value);
        balanced_cuda(cudaGetLastError());balanced_cuda(cudaMemcpy(value.data(),out.value,n*sizeof(double),cudaMemcpyDeviceToHost));
        const Eigen::VectorXd output_expected=r+Z*(a-c);
        if(!value.allFinite()||(value-output_expected).norm()/std::max(output_expected.norm(),1e-300)>2e-12)
            throw std::runtime_error("Fused output projection differs from CPU formula");
        unchanged(dr.value,r.data(),n);unchanged(dw.value,W.data(),size_t(n)*8);unchanged(dz.value,Z.data(),size_t(n)*8);
        unchanged(da.value,a.data(),8);unchanged(dc.value,c.data(),8);
    }
}
int main() {
    try {
        projection_contract();
        constexpr int n=32,w=12;NativeBalancedSpace space(n);
        std::vector<int> ids,slots;std::vector<std::array<double,3>> scalar;
        Eigen::MatrixXd history(n,w);Eigen::VectorXd final(n),rhs(n);
        std::vector<double> bank(n*w);
        for(int i=0;i<n;++i){final[i]=std::cos(.17*i);rhs[i]=std::sin(.27*i)+.3;}
        for(int j=0;j<w;++j) {
            ids.push_back(j);slots.push_back((j*5)%w);const double rho=std::pow(.8,j),alpha=1./(j+1);
            scalar.push_back({alpha,rho,rho/alpha});
            for(int i=0;i<n;++i){history(i,j)=std::sin(.11*(i+1)*(j+1));bank[slots.back()*n+i]=history(i,j);}
        }
        BalancedBuffer<double> dbank(n*w),dfinal(n),drhs(n),out(n);
        balanced_cuda(cudaMemcpy(dbank.value,bank.data(),bank.size()*sizeof(double),cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(dfinal.value,final.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(drhs.value,rhs.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        const auto U=native_stat_coefficients(ids,scalar);space.source(dbank.value,slots,dfinal.value,U);
        Eigen::MatrixXd Z=(final.replicate(1,w)-history)*U,actual(n,8);
        balanced_cuda(cudaMemcpy(actual.data(),space.Z.value,n*8*sizeof(double),cudaMemcpyDeviceToHost));
        if((actual-Z).norm()/Z.norm()>1e-10)return 1;
        auto multiply=[&](const double* x,double* y){fixture_diagonal<<<1,64,0,cudaStreamPerThread>>>(n,2.,x,y);balanced_cuda(cudaGetLastError());};
        auto base=[&](const double* x,double* y){fixture_diagonal<<<1,64,0,cudaStreamPerThread>>>(n,.2,x,y);balanced_cuda(cudaGetLastError());};
        space.target(multiply);Eigen::VectorXd value(n);
        Eigen::MatrixXd A=Eigen::MatrixXd::Zero(n,n),M=A;
        for(int i=0;i<n;++i){A(i,i)=2.+.03*i;M(i,i)=.2+.03*i;}
        const Eigen::MatrixXd E=Z.transpose()*A*Z;
        const Eigen::MatrixXd Q=Z*E.llt().solve(Z.transpose()),I=Eigen::MatrixXd::Identity(n,n);
        double relative=0;
        const Eigen::MatrixXd full_operator=Q+(I-Q*A.transpose())*M*(I-A*Q);
        for(int probe=0;probe<3;++probe) {
            Eigen::VectorXd input=rhs;
            if(probe==1)input.setZero();else if(probe==2)input=Z.col(3);
            balanced_cuda(cudaMemcpy(drhs.value,input.data(),n*sizeof(double),cudaMemcpyHostToDevice));
            const Eigen::VectorXd expected=full_operator*input;
            for(auto mode:{NativeBalancedSpace::ApplyMode::Reference,NativeBalancedSpace::ApplyMode::FusedProjection}) {
                space.apply(drhs.value,out.value,base,nullptr,mode);
                balanced_cuda(cudaMemcpy(value.data(),out.value,n*sizeof(double),cudaMemcpyDeviceToHost));
                const double error=(value-expected).norm()/std::max(expected.norm(),1e-300);
                if(!std::isfinite(error)||error>1e-9)return 2;relative=std::max(relative,error);
                Eigen::VectorXd after(n);balanced_cuda(cudaMemcpy(after.data(),drhs.value,n*sizeof(double),cudaMemcpyDeviceToHost));
                if(std::memcmp(after.data(),input.data(),n*sizeof(double))!=0)return 4;
            }
        }
        balanced_cuda(cudaMemset(space.Z.value,0,n*8*sizeof(double)));bool rejected=false;
        try{space.target(multiply);}catch(const NativeBalancedRejected&){rejected=true;}
        if(!rejected)return 3;
        std::cout<<"native GPU tail/input/CPU-formula/reference/fused/zero-rank contracts passed; action_relative="<<relative<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}
}
