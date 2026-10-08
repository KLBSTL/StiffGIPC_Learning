#include <linear_system/utils/spmv.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using gipc::Float;
using gipc::Matrix3x3;

namespace
{
void check(cudaError_t error)
{
    if(error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> struct DeviceArray
{
    T* data = nullptr;
    explicit DeviceArray(const std::vector<T>& values)
    {
        check(cudaMalloc(&data, values.size() * sizeof(T)));
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if(data) cudaFree(data); }
};

struct CapturedGraph
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~CapturedGraph()
    {
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
    }
};

struct Entry
{
    int row, col;
    Matrix3x3 matrix;
};

// This is a host FP64 reference for every stored block, including both
// orientations when present. It does not duplicate the production CUDA kernel.
std::vector<Float> cpu_reference(const std::vector<Entry>& entries,
                                 const std::vector<Float>& x,
                                 const std::vector<Float>& y,
                                 int dofs, Float a, Float b)
{
    auto expected = y;
    for(int i = 0; i < dofs; ++i)
        expected[i] = b == 0 ? Float(0) : b * y[i];
    for(const auto& entry : entries)
        for(int k = 0; k < 3; ++k)
        {
            Float upper = 0, lower = 0;
            for(int c = 0; c < 3; ++c)
            {
                upper += entry.matrix(k, c) * x[3 * entry.col + c];
                lower += entry.matrix(c, k) * x[3 * entry.row + c];
            }
            expected[3 * entry.row + k] += a * upper;
            if(entry.row != entry.col)
                expected[3 * entry.col + k] += a * lower;
        }
    return expected;
}

void verify(const std::vector<Float>& expected, const Float* device,
            int dofs, const std::string& label, double& maximum_error)
{
    std::vector<Float> actual(expected.size());
    check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(Float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < dofs; ++i)
    {
        const double error = std::abs(actual[i] - expected[i]) / std::max(1., std::abs(expected[i]));
        maximum_error = std::max(maximum_error, error);
        if(!std::isfinite(actual[i]) || !std::isfinite(error) || error > 1e-12)
            throw std::runtime_error(label + ": CPU FP64 reference mismatch at " + std::to_string(i));
    }
    for(size_t i = dofs; i < actual.size(); ++i)
        if(actual[i] != expected[i])
            throw std::runtime_error(label + ": output tail guard overwritten");
}

void fixture(int count, bool long_row, bool empty_vector,
             int& cases, int& graph_replays, double& maximum_error)
{
    const int rows = empty_vector ? 0 : 23;
    const int dofs = 3 * rows, guard = 7;
    std::vector<Entry> entries;
    for(int i = 0; i < count; ++i)
    {
        Entry entry;
        // Rows 21 and 22 are empty. Long rows cross warp and block boundaries.
        entry.row = long_row ? 0 : i % 21;
        entry.col = long_row ? i % 21 : (i * 5 + 3) % 21;
        if(i % 7 == 0) entry.col = entry.row;
        if(!long_row && i < 2 && count >= 2)
        {
            entry.row = i;
            entry.col = 1 - i;  // Both stored orientations remain contributions.
        }
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                entry.matrix(r, c) = ((i + 3 * r + 7 * c) % 17 - 8) * .015625
                                     + (r == c ? .75 : 0.);
        entries.push_back(entry);
    }
    std::stable_sort(entries.begin(), entries.end(), [](const Entry& first, const Entry& second)
    {
        return first.row < second.row || (first.row == second.row && first.col < second.col);
    });
    // Tail matrix/index sentinels must never be consumed by inactive lanes.
    std::vector<Matrix3x3> matrices(count + guard, Matrix3x3::Constant(999));
    std::vector<int> row_ids(count + guard, 1000000), col_ids(row_ids);
    for(int i = 0; i < count; ++i)
    {
        matrices[i] = entries[i].matrix;
        row_ids[i] = entries[i].row;
        col_ids[i] = entries[i].col;
    }
    std::vector<Float> x(dofs + guard, 999), y(x);
    for(int i = 0; i < dofs; ++i)
    {
        x[i] = (i % 23 - 11) * .03125;
        y[i] = (i % 19 - 9) * .0625;
    }
    DeviceArray<Matrix3x3> device_matrices(matrices);
    DeviceArray<int> device_rows(row_ids), device_cols(col_ids);
    DeviceArray<Float> device_x(x), device_y(y), initial_y(y);
    gipc::Spmv spmv;
    const cudatool::CDenseVectorView<Float> x_view(device_x.data, dofs);
    const cudatool::DenseVectorView<Float> y_view(device_y.data, dofs);
    for(Float a : {0., .75, -1.25})
        for(Float b : {0., .375, -.5})
        {
            auto case_y = y;
            if(b == 0)
                std::fill(case_y.begin(), case_y.begin() + dofs,
                          std::numeric_limits<Float>::quiet_NaN());
            check(cudaMemcpy(initial_y.data, case_y.data(), case_y.size() * sizeof(Float),
                             cudaMemcpyHostToDevice));
            const auto expected = cpu_reference(entries, x, case_y, dofs, a, b);
            const std::string label = "count=" + std::to_string(count)
                + (long_row ? " long" : " mixed") + (empty_vector ? " empty" : "")
                + " a=" + std::to_string(a) + " b=" + std::to_string(b);
            auto launch = [&]()
            {
                check(cudaMemcpyAsync(device_y.data, initial_y.data, y.size() * sizeof(Float),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                spmv.warp_reduce_sym_spmv(a, device_matrices.data, device_rows.data,
                                         device_cols.data, count, x_view, b, y_view);
            };
            launch();
            check(cudaGetLastError());
            check(cudaStreamSynchronize(cudaStreamPerThread));
            verify(expected, device_y.data, dofs, label + " host", maximum_error);

            CapturedGraph captured;
            check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch();
            check(cudaStreamEndCapture(cudaStreamPerThread, &captured.graph));
            check(cudaGraphInstantiate(&captured.exec, captured.graph, nullptr, nullptr, 0));
            for(int replay = 0; replay < 3; ++replay)
            {
                check(cudaGraphLaunch(captured.exec, cudaStreamPerThread));
                check(cudaStreamSynchronize(cudaStreamPerThread));
                verify(expected, device_y.data, dofs, label + " graph", maximum_error);
                ++graph_replays;
            }
            ++cases;
        }
    std::vector<Float> x_after(x.size());
    check(cudaMemcpy(x_after.data(), device_x.data, x_after.size() * sizeof(Float), cudaMemcpyDeviceToHost));
    if(x_after != x)
        throw std::runtime_error("SpMV changed its input or input tail guard");
}
}  // namespace

int main()
try
{
    int cases = 0, graph_replays = 0;
    double maximum_error = 0;
    for(int count : {0, 1, 31, 32, 33, 255, 256, 257, 4097})
        for(bool long_row : {false, true})
            fixture(count, long_row, false, cases, graph_replays, maximum_error);
    fixture(0, false, true, cases, graph_replays, maximum_error);
    std::cout << "{\"passed\":true,\"cases\":" << cases
              << ",\"graph_replays\":" << graph_replays
              << ",\"max_cpu_relative_error\":" << maximum_error << "}\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
