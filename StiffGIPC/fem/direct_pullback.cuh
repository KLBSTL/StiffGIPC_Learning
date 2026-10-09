#pragma once
#include <cuda_runtime.h>
#include <cstdlib>

namespace gipc::fem
{
// vec(F) is column-major. Each position component influences only the same
// row of F. Keep the deformation Hessian (and its projection) unchanged.
template<int Dimension, int Vertices>
struct PositionMap
{
    double coefficient[Vertices][Dimension];

    template<class Inverse>
    __host__ __device__ explicit PositionMap(const Inverse& inverse)
    {
#pragma unroll
        for(int column = 0; column < Dimension; ++column)
        {
            coefficient[0][column] = 0;
#pragma unroll
            for(int vertex = 1; vertex < Vertices; ++vertex)
            {
                coefficient[vertex][column] = inverse.m[vertex - 1][column];
                coefficient[0][column] -= coefficient[vertex][column];
            }
        }
    }

    template<class Hessian>
    __host__ __device__ __forceinline__ double entry(
        const Hessian& q, int i, int j, int row, int column) const
    {
        double value = 0;
#pragma unroll
        for(int b = 0; b < Dimension; ++b)
        {
            double left = 0;
#pragma unroll
            for(int a = 0; a < Dimension; ++a)
                left += coefficient[i][a] * q(3 * a + row, 3 * b + column);
            value += left * coefficient[j][b];
        }
        return value;
    }
};

struct ClothHessian
{
    const double (*values)[6];
    __host__ __device__ double operator()(int row, int column) const
    {
        return values[row][column];
    }
};

inline bool direct_pullback_requested()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GIPC_FEM_DIRECT_PULLBACK");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
}
