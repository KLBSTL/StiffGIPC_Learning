#include <fem/femEnergy.cuh>
#include <fem/direct_pullback.cuh>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void checked(cudaError_t error)
{
    if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct DeviceMemory
{
    std::vector<void*> allocations;
    ~DeviceMemory() { for(void* pointer:allocations) cudaFree(pointer); }
    template<class T> T* copy(const std::vector<T>& values)
    {
        T* pointer; checked(cudaMalloc(&pointer,values.size()*sizeof(T)));
        allocations.push_back(pointer);
        checked(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
        return pointer;
    }
};
template<class T> std::vector<T> download(const T* pointer,int count)
{
    std::vector<T> values(count);
    checked(cudaMemcpy(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost));
    return values;
}
double discrepancy(const double* reference,const double* candidate,int count)
{
    double worst=0;
    for(int i=0;i<count;++i)
    {
        if(!std::isfinite(reference[i]) || !std::isfinite(candidate[i]))
            throw std::runtime_error("Nonfinite production FEM output");
        worst=std::max(worst,std::abs(reference[i]-candidate[i])/(1+std::abs(reference[i])));
    }
    if(worst>5e-10) throw std::runtime_error("Production FEM pullback differs: "+std::to_string(worst));
    return worst;
}

// An independent Jacobian oracle: perturb positions, build Ds, and multiply
// by Dm^-1 with Eigen. The reference never uses PositionMap coefficients.
template<int D> void mapping_oracle()
{
    constexpr int V=D+1;
    struct Inverse { double m[D][D]; } inverse{};
    Eigen::Matrix<double,D,D> inv;
    for(int i=0;i<D;++i) for(int j=0;j<D;++j)
        inverse.m[i][j]=inv(i,j)=i==j?1.2+.3*i:.13*(i+1)-.07*(j+1);
    Eigen::Matrix<double,3,V> x=Eigen::Matrix<double,3,V>::Random();
    auto deformation=[&](const Eigen::Matrix<double,3,V>& positions) {
        Eigen::Matrix<double,3,D> ds;
        for(int i=0;i<D;++i) ds.col(i)=positions.col(i+1)-positions.col(0);
        return Eigen::Matrix<double,3,D>(ds*inv);
    };
    Eigen::Matrix<double,3*D,3*V> jacobian;
    const double step=1e-5;
    for(int i=0;i<3*V;++i)
    {
        auto plus=x,minus=x;
        plus(i%3,i/3)+=step; minus(i%3,i/3)-=step;
        const auto difference=(deformation(plus)-deformation(minus)).eval();
        jacobian.col(i)=Eigen::Map<const Eigen::Matrix<double,3*D,1>>(difference.data())/(2*step);
    }
    const Eigen::Matrix<double,3*D,3*D> random=Eigen::Matrix<double,3*D,3*D>::Random();
    const Eigen::Matrix<double,3*D,3*D> q=random.transpose()*random+Eigen::Matrix<double,3*D,3*D>::Identity();
    const auto dense=(jacobian.transpose()*q*jacobian).eval();
    const gipc::fem::PositionMap<D,V> map(inverse);
    double worst=0;
    for(int i=0;i<V;++i) for(int j=0;j<V;++j)
        for(int r=0;r<3;++r) for(int c=0;c<3;++c)
            worst=std::max(worst,std::abs(map.entry(q,i,j,r,c)-dense(3*i+r,3*j+c)));
    if(worst>1e-8) throw std::runtime_error("Independent finite-difference Jacobian oracle failed");
}

template<bool Direct> void launch_triangle(int count,__GEIGEN__::Matrix2x2d* inverse,
    const double3* x,const uint3* elements,const double* area,double3* gradient,
    Eigen::Matrix3d* blocks,int* rows,int* cols,double stiffness)
{
    _calculate_triangle_fem_gradient_hessian<Direct><<<(count+127)/128,128>>>(inverse,x,
        elements,area,gradient,count,stiffness,.7*stiffness,.013,7,blocks,rows,cols,1.08,23);
}
template<bool Direct> void launch_tet(int count,__GEIGEN__::Matrix3x3d* inverse,
    const double3* x,const uint4* elements,const double* volume,double3* gradient,
    const double* mu,const double* lambda,Eigen::Matrix3d* blocks,int* rows,int* cols)
{
    _calculate_fem_gradient_hessian<Direct><<<(count+127)/128,128>>>(inverse,x,elements,
        volume,gradient,count,mu,lambda,7,blocks,rows,cols,.013,23);
}

template<int D> double production_equivalence(int count,double stiffness)
{
    constexpr int V=D+1, B=V*(V+1)/2;
    using Inverse=std::conditional_t<D==2,__GEIGEN__::Matrix2x2d,__GEIGEN__::Matrix3x3d>;
    using Element=std::conditional_t<D==2,uint3,uint4>;
    std::vector<Inverse> inverse(count);
    std::vector<Element> elements(count);
    std::vector<double3> x(count*V);
    std::vector<double> weight(count),mu(count,stiffness),lambda(count,3.7*stiffness);
    for(int e=0;e<count;++e)
    {
        for(int i=0;i<D;++i) for(int j=0;j<D;++j)
            inverse[e].m[i][j]=i==j?1.0+.02*(e%7):.11*(i+1)-.04*(j+1);
        // Reversed / permuted global indices exercise the upper-block transpose.
        for(int i=0;i<V;++i)
            reinterpret_cast<unsigned*>(&elements[e])[i]=V*e+(i+e)%V;
        Eigen::Matrix<double,3,D> ds=Eigen::Matrix<double,3,D>::Zero();
        for(int i=0;i<D;++i) ds(i,i)=.7+.045*(e%17);
        ds(0,1)=.08; ds(2,0)=.09;
        if constexpr(D==3) { ds(1,2)=.1; if(e%11==0) ds(2,2)=-.03; }
        const Eigen::Vector3d origin(.07*e,-.02*e,.01*e);
        for(int i=0;i<V;++i)
        {
            Eigen::Vector3d point=origin;
            if(i) point+=ds.col(i-1);
            x[reinterpret_cast<unsigned*>(&elements[e])[i]]=make_double3(point[0],point[1],point[2]);
        }
        weight[e]=.2+.01*(e%19);
    }
    DeviceMemory memory;
    auto* di=memory.copy(inverse);auto* de=memory.copy(elements);auto* dx=memory.copy(x);
    auto* dw=memory.copy(weight);auto* dm=memory.copy(mu);auto* dl=memory.copy(lambda);
    const std::vector<double3> zero(count*V,make_double3(0,0,0));
    auto* gr=memory.copy(zero);auto* gc=memory.copy(zero);
    const std::vector<Eigen::Matrix3d> sentinel(count*B+7,Eigen::Matrix3d::Constant(-33.0));
    auto* hr=memory.copy(sentinel);auto* hc=memory.copy(sentinel);
    const std::vector<int> ids(count*B+7,-1);
    auto* rr=memory.copy(ids);auto* rc=memory.copy(ids);auto* cr=memory.copy(ids);auto* cc=memory.copy(ids);
    if constexpr(D==2)
    {
        launch_triangle<false>(count,di,dx,de,dw,gr,hr,rr,rc,stiffness);
        launch_triangle<true>(count,di,dx,de,dw,gc,hc,cr,cc,stiffness);
    }
    else
    {
        launch_tet<false>(count,di,dx,de,dw,gr,dm,dl,hr,rr,rc);
        launch_tet<true>(count,di,dx,de,dw,gc,dm,dl,hc,cr,cc);
    }
    checked(cudaGetLastError());checked(cudaDeviceSynchronize());
    const auto a=download(gr,count*V),b=download(gc,count*V);
    const auto h=download(hr,count*B+7),k=download(hc,count*B+7);
    if(download(rr,count*B+7)!=download(cr,count*B+7) || download(rc,count*B+7)!=download(cc,count*B+7))
        throw std::runtime_error("Triplet index/offset mismatch");
    const double gradient_error=discrepancy(reinterpret_cast<const double*>(a.data()),reinterpret_cast<const double*>(b.data()),count*V*3);
    const double hessian_error=discrepancy(h[0].data(),k[0].data(),(count*B+7)*9);
    return std::max(gradient_error,hessian_error);
}

int main() try
{
    mapping_oracle<2>();mapping_oracle<3>();
    double worst=0;
    for(int count:{1,127,128,129,257}) for(double stiffness:{1e2,1e5,1e8})
    {
        worst=std::max(worst,production_equivalence<2>(count,stiffness));
        worst=std::max(worst,production_equivalence<3>(count,stiffness));
    }
    std::cout<<"Passed independent Jacobian oracle and 30 production GPU comparisons; max scaled error "<<worst<<"\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr<<error.what()<<"\n";return 1;
}
