#pragma once

// This file is included after mas_cholesky.inl, which defines CHOL_N from the
// actual BANKSIZE. Preserve L, and store B = L^-1 in FP64 row-major layout.
// Each thread solves one fixed identity column. It reads only entries of B
// that the same thread has already written; columns have no dependencies.
__global__ void invert_cholesky_factor(const double* factors,
                                      double* inverse,
                                      int* status)
{
    const int block = blockIdx.x;
    const int column = threadIdx.x;
    const double* l = factors + block * CHOL_N * CHOL_N;
    double* b = inverse + block * CHOL_N * CHOL_N;
    __shared__ int invalid;
    if(column == 0)
        invalid = 0;
    __syncthreads();
    if(column < CHOL_N)
    {
        for(int row = 0; row < CHOL_N; ++row)
        {
            double value = 0;
            if(row >= column)
            {
                value = row == column ? 1.0 : 0.0;
                for(int k = column; k < row; ++k)
                    value -= l[row * CHOL_N + k] * b[k * CHOL_N + column];
                value /= l[row * CHOL_N + row];
            }
            b[row * CHOL_N + column] = value;
            if(!isfinite(value))
                atomicMax(&invalid, row * CHOL_N + column + 1);
        }
    }
    __syncthreads();
    if(column == 0)
        status[block] = invalid;
}

// Apply B^T(B r), keeping the factorized SPD form. Independent scalar rows
// execute fixed-order FP64 dot products. Padding avoids the stride-CHOL_N
// shared-memory bank pattern when one product reads the transpose.
__global__ void factor_inverse_action(const double* inverse,
                                      const Eigen::Vector3d* rhs,
                                      double3* output)
{
    const int block = blockIdx.x;
    const int row = threadIdx.x;
    const double* b = inverse + block * CHOL_N * CHOL_N;
    __shared__ double cached[CHOL_N][CHOL_N + 1];
    __shared__ double r[CHOL_N];
    __shared__ double intermediate[CHOL_N];
    for(int index = row; index < CHOL_N * CHOL_N; index += blockDim.x)
        cached[index / CHOL_N][index % CHOL_N] = b[index];
    if(row < CHOL_N)
        r[row] = reinterpret_cast<const double*>(rhs)[block * CHOL_N + row];
    __syncthreads();
    if(row < CHOL_N)
    {
        double value = 0;
        for(int column = 0; column <= row; ++column)
            value += cached[row][column] * r[column];
        intermediate[row] = value;
    }
    __syncthreads();
    if(row < CHOL_N)
    {
        double value = 0;
        for(int column = row; column < CHOL_N; ++column)
            value += cached[column][row] * intermediate[column];
        reinterpret_cast<double*>(output)[block * CHOL_N + row] = value;
    }
}
