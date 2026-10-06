#pragma once
#include <cuda_tools/cuda_all.h>
#include <gipc/type_define.h>
#include <vector>
int validate_toi_components(const char* report_path);

// PT=0, EE=1, ground=2. Primitive identity is independent of the closest
// distance feature, which can change between AL iterations.
struct ToiContact
{
    int kind = 0;
    int ids[4] = {-1,-1,-1,-1};
    double3 grad[4]{};
    double3 anchor[4]{};
    double offset = 0, lambda = 0, gamma = 1, slack = 0, penalty_mu = 0;
    // Optional diagnostic friction history, independent of AL warm starting.
    // The default friction path still reads lambda * gamma.
    double friction_lambda = 0, friction_gamma = 1;
    double2 coordinates{};
};
struct ToiState
{
    cudatool::DeviceBuffer<double3> safe, trial, previous_trial;
    cudatool::DeviceBuffer<gipc::Vector12> safe_q, trial_q;
    cudatool::DeviceBuffer<ToiContact> contacts;
    cudatool::DeviceBuffer<double> pair_times, ground_times, energy, scalar;
    std::vector<ToiContact> host_contacts;
    std::vector<double> host_vertex_mu;
    double mu = 0, initial_mu = 0, delta = 0;
    int self_count = 0, ground_count = 0;
    bool independent_friction_history = false;
};
