#undef NDEBUG
#include <cuda_tools/cuda_all.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>
#include <solver/ipc_physical_residual.inl>

template<class T> void upload(cudatool::DeviceBuffer<T>& target,const std::vector<T>& values) {
    target.resize(values.size());
    if(!values.empty())CUDA_SAFE_CALL(cudaMemcpy(target.data(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void close(double value,double expected){require(std::abs(value-expected)<=1e-10*std::max(1.,expected),"Physical residual differs from FP64 reference");}
int main() {
    using namespace gipc;
    cudatool::DeviceBuffer<double> gradient,masses;
    cudatool::DeviceBuffer<Matrix12x12> inverse;
    cudatool::DeviceBuffer<ABDJacobi> jacobians;
    cudatool::DeviceBuffer<BodyBoundaryType> types;
    cudatool::DeviceBuffer<int> body,boundary;
    cudatool::DeviceBuffer<double3> barrier,shape;
    Matrix12x12 factor=Matrix12x12::Zero();
    for(int i=0;i<12;++i)for(int j=0;j<12;++j)factor(i,j)=std::sin(1.+i+3*j);
    Matrix12x12 mass=factor.transpose()*factor+Matrix12x12::Identity();
    Vector12 g;for(int i=0;i<12;++i)g[i]=std::cos(2.*i);
    const Vector3 points[]={Vector3{1,2,3},Vector3{-.5,.2,1},Vector3{10,0,0}};
    std::vector<ABDJacobi> js;for(const auto& p:points)js.emplace_back(p);
    std::vector<double> all_gradient(24,1e100);std::copy(g.data(),g.data()+12,all_gradient.begin());
    upload(gradient,all_gradient);upload(inverse,std::vector<Matrix12x12>{mass.inverse(),mass.inverse()});
    upload(jacobians,js);upload(body,std::vector<int>{0,0,1});
    upload(types,std::vector<BodyBoundaryType>{BodyBoundaryType::Free,BodyBoundaryType::Fixed});
    upload(masses,std::vector<double>{.5,2,0});upload(boundary,std::vector<int>{0,0,1});
    upload(barrier,std::vector<double3>{make_double3(3,4,0),make_double3(0,2,0),make_double3(1e100,0,0)});
    upload(shape,std::vector<double3>{make_double3(1,0,0),make_double3(0,2,0),make_double3(1e100,0,0)});
    auto measure=[&](int abd_count,int fem_count,double dt) {
        return ipc_physical_residual(gradient.data(),inverse.data(),types.data(),2,
            jacobians.data(),body.data(),abd_count,barrier.data(),shape.data(),masses.data(),boundary.data(),0,fem_count,dt);
    };
    Vector12 physical=mass.inverse()*g;double expected=0;
    for(int i=0;i<2;++i)expected=std::max(expected,(js[i]*physical).norm());
    close(measure(3,0,.02),expected/.02);
    close(measure(0,3,.02),std::sqrt(32.)/.5/.02);
    close(measure(3,3,.02),std::max(expected,std::sqrt(32.)/.5)/.02);
    close(measure(0,0,.02),0);
    // Affine rest-coordinate reparameterization: J'=J S, M'=S^T M S, g'=S^T g.
    Matrix12x12 change=Matrix12x12::Zero();change.block<3,3>(0,0).setIdentity();
    const Vector3 shift{.4,-.7,1.2};
    for(int axis=0;axis<3;++axis){change.block<1,3>(axis,3+3*axis)=shift.transpose();
        change.block<3,3>(3+3*axis,3+3*axis)=2*Eigen::Matrix3d::Identity();}
    const Matrix12x12 transformed_mass=change.transpose()*mass*change;
    const Vector12 transformed_g=change.transpose()*g;
    std::copy(transformed_g.data(),transformed_g.data()+12,all_gradient.begin());upload(gradient,all_gradient);
    upload(inverse,std::vector<Matrix12x12>{transformed_mass.inverse(),mass.inverse()});
    for(int i=0;i<3;++i)js[i]=ABDJacobi{2*points[i]+shift};upload(jacobians,js);
    close(measure(3,0,.02),expected/.02);
    all_gradient[0]=std::numeric_limits<double>::quiet_NaN();upload(gradient,all_gradient);
    require(std::isinf(measure(3,0,.02)),"Nonfinite free ABD gradient hidden");
    // Capacity growth, many blocks, and one-element tail; no ABD reads.
    std::vector<double3> zero(513,make_double3(0,0,0));auto tail=zero;tail.back()=make_double3(3,4,0);
    upload(barrier,tail);upload(shape,zero);upload(masses,std::vector<double>(513,1));upload(boundary,std::vector<int>(513,0));
    close(measure(0,513,.01),500);
    upload(barrier,zero);close(measure(0,513,.01),0);
    std::cout<<"physical residual: ABD/FEM reference, fixed masks, affine invariance, finite guard, empty, growth/tail passed\n";
    return 0;
}
