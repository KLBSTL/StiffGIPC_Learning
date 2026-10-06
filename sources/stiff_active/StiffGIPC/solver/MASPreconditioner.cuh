//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <fem/device_fem_data.cuh>
#include <gipc/utils/json.h>
#include <math/eigen_data.h>
#include <cuda_tools/cuda_all.h>
#include "linear_system/linear_system/global_matrix.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cstdint>
#include <solver/mas_factor_action_options.h>
#include <solver/legacy_restrict_options.h>

class MASPreconditioner
{

    int totalNodes            = 0;
    int totalMapNodes         = 0;
    int levelnum              = 0;
    int collision_node_Offset = 0;
    int totalNumberClusters   = 0;
    //int bankSize;
    int2 h_clevelSize = {};

    cudatool::DeviceBuffer<int2>               d_levelSize;
    cudatool::DeviceBuffer<int>                d_coarseSpaceTables;
    cudatool::DeviceBuffer<int>                d_prefixOriginal;
    cudatool::DeviceBuffer<int>                d_prefixSumOriginal;
    cudatool::DeviceBuffer<int>                d_goingNext;
    cudatool::DeviceBuffer<int>                d_denseLevel;
    cudatool::DeviceBuffer<__GEIGEN__::itable> d_coarseTable;
    cudatool::DeviceBuffer<unsigned int>       d_fineConnectMask;
    cudatool::DeviceBuffer<unsigned int>       d_nextConnectMask;
    cudatool::DeviceBuffer<unsigned int>       d_nextPrefix;
    cudatool::DeviceBuffer<unsigned int>       d_nextPrefixSum;


    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixT>    d_MatMas;
    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymT> d_inverseMatMas;
    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymf> d_precondMatMas;
    cudatool::DeviceBuffer<Eigen::Vector3f>           d_multiLevelR;
    cudatool::DeviceBuffer<Precision_T3>              d_multiLevelZ;
    cudatool::DeviceBuffer<Eigen::Vector3d> d_multiLevelR64;
    cudatool::DeviceBuffer<double3> d_multiLevelZ64;
    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymT> d_precondMatMas64;
    bool wide_apply = false;
    bool inverse64 = false;
    bool cholesky = false;
    cudatool::DeviceBuffer<double> d_cholesky;
    cudatool::DeviceBuffer<int> d_cholesky_status;
    cudatool::DeviceBuffer<double> d_factor_inverse;
    cudatool::DeviceBuffer<int> d_factor_inverse_status;
    cudatool::DeviceBuffer<int> d_restriction_starts, d_restriction_nodes;
    void prepare_cholesky();
    void prepare_restriction_map(bool measure=false);
    double restriction_map_prepare_ms=0;
    gipc::Json diagnostic_stage_study(const double3* R,double3* Z,const std::string& prefix);
    bool precision_initialized = false;

  public:
    static int replay_fixture(const char* prefix,const char* output);
    static int cholesky_fixture(const char* prefix,const char* output);
    bool wide_apply_enabled() const { return wide_apply; }
    bool inverse64_enabled() const { return inverse64; }
    bool cholesky_enabled() const { return cholesky; }
    bool fused_dot_collect_supported() const;
    double restriction_map_prepare_time_ms() const { return restriction_map_prepare_ms; }
    // Diagnostic serialization of every owned buffer, including scratch.
    template<class Save> void diagnostic_buffers(Save save) const
    {
#define MAS_SAVE(name) save(#name, name.data(), name.size(), sizeof(*name.data()))
        MAS_SAVE(d_levelSize); MAS_SAVE(d_coarseSpaceTables);
        MAS_SAVE(d_prefixOriginal); MAS_SAVE(d_prefixSumOriginal);
        MAS_SAVE(d_goingNext); MAS_SAVE(d_denseLevel); MAS_SAVE(d_coarseTable);
        MAS_SAVE(d_fineConnectMask); MAS_SAVE(d_nextConnectMask);
        MAS_SAVE(d_nextPrefix); MAS_SAVE(d_nextPrefixSum);
        MAS_SAVE(d_MatMas); MAS_SAVE(d_inverseMatMas); MAS_SAVE(d_precondMatMas);
        MAS_SAVE(d_multiLevelR); MAS_SAVE(d_multiLevelZ);
        MAS_SAVE(d_multiLevelR64); MAS_SAVE(d_multiLevelZ64);
        MAS_SAVE(d_precondMatMas64);
        MAS_SAVE(d_cholesky); MAS_SAVE(d_cholesky_status);
        MAS_SAVE(d_factor_inverse); MAS_SAVE(d_factor_inverse_status);
        MAS_SAVE(d_restriction_starts); MAS_SAVE(d_restriction_nodes);
        MAS_SAVE(d_neighborList); MAS_SAVE(d_neighborStart); MAS_SAVE(d_neighborStartTemp);
        MAS_SAVE(d_neighborNum); MAS_SAVE(d_neighborListInit); MAS_SAVE(d_neighborNumInit);
        MAS_SAVE(d_partId_map_real); MAS_SAVE(d_real_map_partId);
#undef MAS_SAVE
    }
    std::vector<int> diagnostic_dimensions() const
    { return {totalNodes,totalMapNodes,levelnum,collision_node_Offset,
              totalNumberClusters,h_clevelSize.x,h_clevelSize.y,neighborListSize}; }
    int                                  neighborListSize = 0;
    cudatool::DeviceBuffer<unsigned int> d_neighborList;
    cudatool::DeviceBuffer<unsigned int> d_neighborStart;
    cudatool::DeviceBuffer<unsigned int> d_neighborStartTemp;
    cudatool::DeviceBuffer<unsigned int> d_neighborNum;
    cudatool::DeviceBuffer<unsigned int> d_neighborListInit;
    cudatool::DeviceBuffer<unsigned int> d_neighborNumInit;
    cudatool::DeviceBuffer<int>          d_partId_map_real;
    cudatool::DeviceBuffer<int>          d_real_map_partId;

  public:
    static std::size_t requiredGoingNextCapacity(std::size_t vertex_count,
                                                 std::size_t mapped_node_count,
                                                 std::size_t level_count)
    {
        const std::size_t base_count = std::max(vertex_count, mapped_node_count);
        if(base_count == 0 || level_count == 0)
            return 0;
        constexpr std::size_t bank_size = BANKSIZE;
        if(base_count
           > std::numeric_limits<std::size_t>::max() - (bank_size - 1))
            std::abort();
        const std::size_t padded_count =
            (base_count + bank_size - 1) / bank_size * bank_size;
        if(padded_count > std::numeric_limits<std::size_t>::max() / level_count)
            std::abort();
        return padded_count * level_count;
    }

    void initPreconditioner_Neighbor(int vertNum,
                                     int mCollision_node_offset,
                                     int totalNeighborNum,
                                     int partMapSize);
    void computeNumLevels(int vertNum);  // called in initPreconditioner_Neighbor

    void initPreconditioner_Matrix();


    int  ReorderRealtime(int cpNum, const int4* collisionPairs = nullptr);
    void BuildConnectMaskL0();           // called in ReorderRealtime
    void PreparePrefixSumL0();           // called in ReorderRealtime
    void BuildLevel1();                  // called in ReorderRealtime
    void BuildConnectMaskLx(int level);  // called in ReorderRealtime
    void NextLevelCluster(int level);    // called in ReorderRealtime
    void PrefixSumLx(int level);         // called in ReorderRealtime
    void ComputeNextLevel(int level);    // called in ReorderRealtime
    void AggregationKernel();            // called in ReorderRealtime
    void BuildCollisionConnection(unsigned int* connectionMsk,
                                  int*          coarseTableSpace,
                                  int           level,
                                  int           cpNum,
                                  const int4* collisionPairs);  // called in ReorderRealtime

    void setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                int*             row_ids,
                                int*             col_ids,
                                uint32_t*        indices,
                                int              offset,
                                int              triplet_num,
                                int              cpNum,
                                const int4*      collisionPairs);
    void PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                             int*             row_ids,
                             int*             col_ids,
                             uint32_t*        indices,
                             int              offset,
                             int              triplet_number);

    void preconditioning(const double3* R, double3* Z, double* dot_partials=nullptr);
    void collect_fused_dot(const double3* R,double3* Z,double* dot_partials);
    void BuildMultiLevelR(const double3* R);  // called in preconditioning
    void SchwarzLocalXSym();                  // called in preconditioning
    void SchwarzLocalXSym_block3();           // called in preconditioning
    void SchwarzLocalXSym_sym();              // called in preconditioning
    void CollectFinalZ(double3* Z,const double3* R=nullptr,double* dot_partials=nullptr);

    void FreeMAS();
    std::vector<std::uintptr_t> graph_signature() const
    {
        return {static_cast<std::uintptr_t>(totalNodes),
                static_cast<std::uintptr_t>(totalMapNodes),
                static_cast<std::uintptr_t>(totalNumberClusters),
                static_cast<std::uintptr_t>(levelnum),
                reinterpret_cast<std::uintptr_t>(d_multiLevelR.data()),
                reinterpret_cast<std::uintptr_t>(d_multiLevelZ.data()),
                reinterpret_cast<std::uintptr_t>(d_goingNext.data()),
                reinterpret_cast<std::uintptr_t>(d_prefixOriginal.data()),
                reinterpret_cast<std::uintptr_t>(d_fineConnectMask.data()),
                reinterpret_cast<std::uintptr_t>(d_partId_map_real.data()),
                reinterpret_cast<std::uintptr_t>(d_precondMatMas.data()),
                reinterpret_cast<std::uintptr_t>(d_coarseTable.data()),
                reinterpret_cast<std::uintptr_t>(d_real_map_partId.data()),
                static_cast<std::uintptr_t>(wide_apply),
                reinterpret_cast<std::uintptr_t>(d_multiLevelR64.data()),
                reinterpret_cast<std::uintptr_t>(d_multiLevelZ64.data()),
                static_cast<std::uintptr_t>(inverse64),
                reinterpret_cast<std::uintptr_t>(d_precondMatMas64.data()),
                static_cast<std::uintptr_t>(cholesky),
                reinterpret_cast<std::uintptr_t>(d_cholesky.data()),
                reinterpret_cast<std::uintptr_t>(d_restriction_starts.data()),
                reinterpret_cast<std::uintptr_t>(d_restriction_nodes.data()),
                static_cast<std::uintptr_t>(cholesky && gipc::mas_factor_inverse_action()),
                reinterpret_cast<std::uintptr_t>(d_factor_inverse.data()),
                static_cast<std::uintptr_t>(!wide_apply && gipc::legacy_ordered_restrict())};
    }
};
