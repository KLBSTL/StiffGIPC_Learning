#include <cuda_runtime.h>
#include <linear_system/preconditioner/diag_fused_update.cuh>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>

using gipc::Float;
using gipc::Matrix3x3;
using gipc::Vector3;

void check(cudaError_t error)
{
    if(error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> T* upload(const std::vector<T>& host)
{
    T* device;
    check(cudaMalloc(&device, host.size() * sizeof(T)));
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}
// The original pipeline: one scalar thread updates x/r; a later kernel
// multiplies the stored residual by each dense 3x3 inverse block.
__global__ void reference_update(int n, Float* x, Float* r,
                                 const Float* p, const Float* ap,
                                 const Float* alpha)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < n)
    {
        x[j] += *alpha * p[j];
        r[j] -= *alpha * ap[j];
    }
}
__global__ void reference_apply(int n, const Matrix3x3* inverse,
                                const Float* r, Float* z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        Eigen::Map<Vector3>(z + 3 * i) = inverse[i] * Eigen::Map<const Vector3>(r + 3 * i);
}

int main()
try
{
    int fixtures = 0;
    size_t compared = 0;
    double max_cpu_relative = 0;
    for(int blocks : {1, 85, 86, 255, 256, 257, 2049})
    for(double alpha : {0.0, -0.125, 0.03125, 1.0})
    {
        const int n = blocks * 3, guard = 6;
        std::vector<Matrix3x3> inverse(blocks);
        std::vector<double> x(n + guard, 999), r(x), p(x), ap(x), z(x);
        for(int i = 0; i < blocks; ++i)
        {
            Vector3 v(std::sin(i + .17), std::cos(i + .31), std::sin(i + .79));
            inverse[i] = v * v.transpose() + Matrix3x3::Identity() * (.1 + (i % 13));
        }
        for(int j = 0; j < n; ++j)
        {
            x[j] = std::sin(j + .3);
            r[j] = std::cos(j + .7) * std::pow(10., j % 9 - 4);
            p[j] = std::sin(j + 1.1);
            ap[j] = std::cos(j + 1.7) * std::pow(10., j % 7 - 3);
        }
        auto* d = upload(inverse); auto* dp = upload(p); auto* da = upload(ap);
        auto* ds = upload(std::vector<double>{alpha});
        auto* xr = upload(x); auto* rr = upload(r); auto* zr = upload(z);
        auto* xf = upload(x); auto* rf = upload(r); auto* zf = upload(z);
        reference_update<<<(n + 255) / 256, 256>>>(n, xr, rr, dp, da, ds);
        reference_apply<<<(blocks + 255) / 256, 256>>>(blocks, d, rr, zr);
        gipc::details::diag_fused_update_kernel<<<(blocks + 255) / 256, 256>>>(blocks, d, xf, rf, dp, da, ds, zf);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        std::vector<double> rx(x.size()), rrhost(x.size()), rz(x.size());
        std::vector<double> fx(x.size()), fr(x.size()), fz(x.size());
        const std::array<std::pair<double*, double*>, 6> copies{{
            {rx.data(), xr}, {rrhost.data(), rr}, {rz.data(), zr},
            {fx.data(), xf}, {fr.data(), rf}, {fz.data(), zf}}};
        for(const auto& pair : copies)
            check(cudaMemcpy(pair.first, pair.second, x.size() * sizeof(double), cudaMemcpyDeviceToHost));
        if(std::memcmp(rx.data(), fx.data(), x.size() * sizeof(double))
           || std::memcmp(rrhost.data(), fr.data(), x.size() * sizeof(double))
           || std::memcmp(rz.data(), fz.data(), x.size() * sizeof(double)))
            throw std::runtime_error("Fused result differs bitwise from original pipeline");
        for(int i = 0; i < blocks; ++i)
        {
            Vector3 cpu_r;
            for(int k = 0; k < 3; ++k)
                cpu_r[k] = std::fma(-alpha, ap[3 * i + k], r[3 * i + k]);
            Vector3 cpu_z = inverse[i] * cpu_r;
            for(int k = 0; k < 3; ++k)
            {
                const double error = std::abs(cpu_z[k] - fz[3 * i + k]) / std::max(1., std::abs(cpu_z[k]));
                max_cpu_relative = std::max(max_cpu_relative, error);
                if(!std::isfinite(error) || error > 1e-12)
                    throw std::runtime_error("CPU reference mismatch");
            }
        }
        for(int j = n; j < n + guard; ++j)
            if(fx[j] != 999 || fr[j] != 999 || fz[j] != 999)
                throw std::runtime_error("Guard overwritten");
        for(void* ptr : std::array<void*, 10>{d, dp, da, ds, xr, rr, zr, xf, rf, zf})
            check(cudaFree(ptr));
        ++fixtures; compared += 3 * x.size();
    }
    std::cout << "{\"passed\":true,\"fixtures\":" << fixtures
              << ",\"bitwise_compared_values\":" << compared
              << ",\"max_cpu_relative_error\":" << max_cpu_relative << "}\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
