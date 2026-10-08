#pragma once
#include <gipc/type_define.h>
#include <cuda_tools/cuda_all.h>
#include "linear_system/linear_system/global_matrix.h"
#include <linear_system/utils/linear_structure_probe.h>
#include <memory>
namespace gipc
{
class LinearStructureProbe;
class Converter
{
    std::unique_ptr<LinearStructureProbe> structure_probe_;
  public:
    Converter();
    ~Converter();
    Converter(const Converter&) = delete;
    Converter& operator=(const Converter&) = delete;
    Converter(Converter&&) noexcept;
    Converter& operator=(Converter&&) noexcept;
    void configure_structure_probe(const LinearStructureProbeConfig& config);
    void reset_structure_probe() noexcept;
    // Borrowed until the next observation/configuration/reset. Disabled or
    // not-yet-observed diagnostics return nullptr; copy before serializing.
    const LinearStructureProbeResult* last_structure_probe() const noexcept;

    // Triplet -> BCOO
    void convert(GIPCTripletMatrix& global_triplets,
                 const int&         start,
                 const int&         length,
                 const int&         out_start_id);


    void _radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                        const int&         start,
                                        const int&         length,
                                        const int&         out_start_id);


    void _make_unique_indices(GIPCTripletMatrix& global_triplets,
                              const int&         start,
                              const int&         length,
                              const int&         out_start_id);


    void _make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                           const int&         start,
                                           const int&         length,
                                           const int&         out_start_id);


    void ge2sym(GIPCTripletMatrix& global_triplets);
};
}  // namespace gipc
