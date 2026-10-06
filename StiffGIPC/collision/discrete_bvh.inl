// Included once by the active mlbvh.cu, after the unchanged build/query helpers.
// The legacy kernels and original-element-ID EE filter remain the sole query
// implementation. Diagnostic scratch is local to one validation call and never
// aliases GIPC's pair, CCD-pair, MatIndex, or count buffers.
#include <algorithm>
#include <chrono>
#include <vector>
#include <gipc/cost_trace.h>

namespace
{
inline uint32_t discrete_count(const lbvh_f& t) { return t.face_number; }
inline uint32_t discrete_count(const lbvh_e& t) { return t.edge_number; }
inline int discrete_kind(const lbvh_f&) { return 0; }
inline int discrete_kind(const lbvh_e&) { return 1; }
inline const void* discrete_elements(const lbvh_f& t) { return t._faces; }
inline const void* discrete_elements(const lbvh_e& t) { return t._edges; }
inline const void* discrete_mapping(const lbvh_f& t) { return t._surfVerts; }
inline const void* discrete_mapping(const lbvh_e& t) { return t._rest_vertexes; }

template <typename Tree>
std::array<uintptr_t, 40> discrete_signature(const Tree& tree)
{
    std::array<uintptr_t, 40> key{};
    size_t slot = 0;
    auto pointer = [&](const void* p) { key[slot++] = reinterpret_cast<uintptr_t>(p); };
    key[slot++] = discrete_count(tree);
    key[slot++] = tree.vert_number;
    pointer(tree._vertexes); pointer(discrete_elements(tree));
    pointer(discrete_mapping(tree)); pointer(tree._bodyId); pointer(tree._btype);
    auto buffer = [&](const auto& b) {
        pointer(b.data()); key[slot++] = b.size(); key[slot++] = b.capacity();
    };
    buffer(tree._nodes); buffer(tree._bvs); buffer(tree._indices);
    buffer(tree._MChash); buffer(tree._flags); buffer(tree._tempLeafBox);
    buffer(tree._sort_indices); buffer(tree._sort_morton_codes);
    return key;
}

template <typename Tree>
void discrete_note_rebuild(Tree& tree)
{
    tree.discrete_state.mode = gipc::DiscreteBVHTreeMode::discrete;
    tree.discrete_state.builds_since_rebuild = 1;
    tree.discrete_state.signature = discrete_signature(tree);
}

template <typename Tree>
bool discrete_storage_matches(const Tree& tree)
{
    const size_t n = discrete_count(tree);
    return n && tree._indices.size() == n && tree._MChash.size() == n
           && tree._nodes.size() == 2 * n - 1 && tree._bvs.size() == 2 * n - 1
           && tree._tempLeafBox.size() == n && tree._flags.size() == n - 1;
}

template <typename Tree>
void discrete_record_cache_bytes(const Tree& tree)
{
    auto& stats = gipc::discrete_bvh_stats()[discrete_kind(tree)];
    stats.swept_cache_capacity_bytes_peak = std::max<uint64_t>(
        stats.swept_cache_capacity_bytes_peak, tree.swept_cache_capacity_bytes());
}

template <typename Tree>
void discrete_select_storage(Tree& tree, bool swept)
{
    if(!gipc::discrete_bvh_config().enabled) return;
    if(tree.select_storage(swept))
    {
        auto& stats = gipc::discrete_bvh_stats()[discrete_kind(tree)];
        ++stats.storage_switches;
        if(!swept && tree.discrete_state.mode == gipc::DiscreteBVHTreeMode::discrete)
            ++stats.ordinary_cache_restores;
    }
    discrete_record_cache_bytes(tree);
}

// Only the selected swept bundle may be allocated here. Calling the public
// MALLOC_DEVICE_MEM would reset/release BOTH caches, including the ordinary
// topology that this feature is intended to preserve.
template <typename Tree>
void discrete_prepare_swept_storage(Tree& tree)
{
    discrete_select_storage(tree, true);
    const uint32_t n = discrete_count(tree);
    if(n > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Swept BVH exceeds the legacy signed element-count range");
    if(n && !discrete_storage_matches(tree))
    {
        tree.swept_state = {};
        tree._indices.resize(n);
        tree._MChash.resize(n);
        tree._sort_indices.release();
        tree._sort_morton_codes.release();
        tree._nodes.resize(2 * static_cast<size_t>(n) - 1);
        tree._bvs.resize(2 * static_cast<size_t>(n) - 1);
        tree._tempLeafBox.resize(n);
        tree._flags.resize(n - 1);
    }
    discrete_record_cache_bytes(tree);
}

template <typename Tree>
bool discrete_swept_valid(const Tree& tree)
{
    return tree.swept_storage_active
        && tree.swept_state.mode == gipc::DiscreteBVHTreeMode::swept
        && discrete_storage_matches(tree)
        && tree.swept_state.signature == discrete_signature(tree);
}

template <typename Tree>
void discrete_note_swept_build(Tree& tree)
{
    if(!gipc::discrete_bvh_config().enabled) return;
    tree.swept_state.mode = gipc::DiscreteBVHTreeMode::swept;
    tree.swept_state.signature = discrete_signature(tree);
    ++gipc::discrete_bvh_stats()[discrete_kind(tree)].swept_full_builds;
    discrete_record_cache_bytes(tree);
}

template <typename Tree>
void discrete_select_query_storage(Tree& tree, bool swept)
{
    if(!gipc::discrete_bvh_config().enabled) return;
    discrete_select_storage(tree, swept);
    const bool valid = swept ? discrete_swept_valid(tree)
        : tree.discrete_state.mode == gipc::DiscreteBVHTreeMode::discrete
          && discrete_storage_matches(tree)
          && tree.discrete_state.signature == discrete_signature(tree);
    if(!valid)
        throw std::runtime_error("BVH query requires current topology/mapping and the matching constructed tree");
}

inline void discrete_leaf_bounds(lbvh_f& tree)
{
    calcLeafBvs(tree._vertexes, tree._faces, tree._bvs, tree.face_number, 0);
}
inline void discrete_leaf_bounds(lbvh_e& tree)
{
    calcLeafBvs(tree._vertexes, tree._edges, tree._bvs, tree.edge_number, 1);
}

template <typename Tree>
void discrete_refit(Tree& tree)
{
    discrete_select_storage(tree, false);
    if(!discrete_count(tree)) return;
    if(tree.discrete_state.mode != gipc::DiscreteBVHTreeMode::discrete
       || !discrete_storage_matches(tree)
       || tree.discrete_state.signature != discrete_signature(tree))
        throw std::runtime_error("Ordinary BVH refit requires a valid ordinary tree and unchanged storage");
    // Recompute ALL leaves from CURRENT positions, reorder by the existing
    // original-ID permutation, then refresh ALL internal bounds. No AABB or
    // contact candidate set from the previous geometry is reused.
    discrete_leaf_bounds(tree);
    sortBvs(tree._indices, tree._bvs, tree._tempLeafBox, discrete_count(tree));
    calcInternalAABB(tree._nodes, tree._bvs, tree._flags, discrete_count(tree));
    // Keep the public host scene box current, matching Construct's contract.
    CUDA_SAFE_CALL(cudaMemcpy(&tree.scene, tree._bvs.data(), sizeof(AABB), cudaMemcpyDeviceToHost));
}

template <typename Tree>
double discrete_construct(Tree& tree)
{
    const auto& config = gipc::discrete_bvh_config();
    auto& stats = gipc::discrete_bvh_stats()[discrete_kind(tree)];
    ++stats.construct_calls;
    if(!config.enabled)
    {
        ++stats.disabled_rebuilds;
        return tree.ConstructRebuild();
    }
    discrete_select_storage(tree, false);
    const uint32_t n = discrete_count(tree);
    if(n > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Ordinary BVH exceeds the legacy signed element-count range");
    const auto mode = tree.discrete_state.mode;
    const bool changed = tree.discrete_state.signature != discrete_signature(tree)
                         || !discrete_storage_matches(tree);
    const bool periodic = tree.discrete_state.builds_since_rebuild >= config.rebuild_interval;
    if(mode != gipc::DiscreteBVHTreeMode::discrete || changed || periodic || n == 0)
    {
        ++stats.production_rebuilds;
        if(mode == gipc::DiscreteBVHTreeMode::swept) ++stats.swept_rebuilds;
        else if(mode == gipc::DiscreteBVHTreeMode::invalid) ++stats.invalid_rebuilds;
        else if(changed) ++stats.signature_rebuilds;
        else ++stats.interval_rebuilds;
        if(n && !discrete_storage_matches(tree)) tree.MALLOC_DEVICE_MEM(static_cast<int>(n));
        gipc::CostScope scope("collision.discrete_bvh_policy_rebuild");
        return tree.ConstructRebuild();
    }
    gipc::CostScope scope("collision.discrete_bvh_policy_refit");
    tree.RefitDiscrete();
    ++tree.discrete_state.builds_since_rebuild;
    ++stats.production_refits;
    return 0;
}

inline bool discrete_box_equal(const AABB& a, const AABB& b)
{
    return a.lower.x == b.lower.x && a.lower.y == b.lower.y && a.lower.z == b.lower.z
           && a.upper.x == b.upper.x && a.upper.y == b.upper.y && a.upper.z == b.upper.z;
}

template <typename Tree>
std::vector<AABB> discrete_check_tree(const Tree& tree, bool check_bounds)
{
    const size_t n = discrete_count(tree);
    if(!n) return {};
    std::vector<Node> nodes;
    std::vector<uint32_t> indices;
    tree._nodes.copy_to(nodes); tree._indices.copy_to(indices);
    if(nodes.size() != 2 * n - 1 || indices.size() != n)
        throw std::runtime_error("Discrete BVH validation: unexpected tree storage size");
    std::vector<unsigned char> seen(nodes.size(), 0), elements(n, 0);
    std::vector<std::pair<uint32_t, uint32_t>> pending{{0, 0}};
    uint32_t max_depth = 0;
    while(!pending.empty())
    {
        const auto item = pending.back(); pending.pop_back();
        const uint32_t index = item.first, depth = item.second;
        if(index >= nodes.size() || seen[index]++)
            throw std::runtime_error("Discrete BVH validation: invalid/cyclic/duplicate node");
        max_depth = std::max(max_depth, depth);
        // The existing DFS stack has 65 entries. LBVH uses unique 64-bit keys;
        // refit does not alter this topology or increase its maximum depth.
        if(depth > 64)
            throw std::runtime_error("Discrete BVH validation: tree exceeds the 65-entry query stack");
        const auto& node = nodes[index];
        if(index == 0 && node.parent_idx != 0xFFFFFFFFu)
            throw std::runtime_error("Discrete BVH validation: root has a parent");
        if(node.element_idx != 0xFFFFFFFFu)
        {
            if(index < n - 1 || node.element_idx >= n || elements[node.element_idx]++
               || indices[index - (n - 1)] != node.element_idx
               || node.left_idx != 0xFFFFFFFFu || node.right_idx != 0xFFFFFFFFu)
                throw std::runtime_error("Discrete BVH validation: original element-ID mapping is invalid");
        }
        else
        {
            if(index >= n - 1 || node.left_idx >= nodes.size() || node.right_idx >= nodes.size()
               || node.left_idx == node.right_idx || nodes[node.left_idx].parent_idx != index
               || nodes[node.right_idx].parent_idx != index)
                throw std::runtime_error("Discrete BVH validation: child/parent links are invalid");
            pending.emplace_back(node.left_idx, depth + 1);
            pending.emplace_back(node.right_idx, depth + 1);
        }
    }
    if(std::find(seen.begin(), seen.end(), 0) != seen.end()
       || std::find(elements.begin(), elements.end(), 0) != elements.end())
        throw std::runtime_error("Discrete BVH validation: disconnected tree or missing original ID");
    auto& stats = gipc::discrete_bvh_stats()[discrete_kind(tree)];
    stats.diagnostic_max_tree_depth = std::max<uint64_t>(stats.diagnostic_max_tree_depth, max_depth);
    if(!check_bounds) return {};
    std::vector<AABB> boxes, original(n);
    tree._bvs.copy_to(boxes);
    for(size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& b = boxes[i];
        if(!std::isfinite(b.lower.x) || !std::isfinite(b.lower.y) || !std::isfinite(b.lower.z)
           || !std::isfinite(b.upper.x) || !std::isfinite(b.upper.y) || !std::isfinite(b.upper.z)
           || b.lower.x > b.upper.x || b.lower.y > b.upper.y || b.lower.z > b.upper.z)
            throw std::runtime_error("Discrete BVH validation: invalid bounds");
        if(i < n - 1)
        {
            if(!discrete_box_equal(b, merge(boxes[nodes[i].left_idx], boxes[nodes[i].right_idx])))
                throw std::runtime_error("Discrete BVH validation: internal bounds are stale");
        }
        else original[nodes[i].element_idx] = b;
    }
    return original;
}

inline void discrete_raw_query(lbvh_f& t, double gap, int4* pairs, int4* ccd,
                               uint32_t* counts, int* indices, uint32_t capacity)
{
    selfQuery_vf(t._bodyId, t._btype, t._vertexes, t._faces, t._surfVerts,
                t._bvs, t._nodes, pairs, ccd, counts, indices, gap, capacity, t.vert_number);
}
inline void discrete_raw_query(lbvh_e& t, double gap, int4* pairs, int4* ccd,
                               uint32_t* counts, int* indices, uint32_t capacity)
{
    if(t.edge_number <= 1) return;
    selfQuery_ee(t._bodyId, t._btype, t._vertexes, t._rest_vertexes, t._edges,
                t._bvs, t._nodes, pairs, ccd, counts, indices, gap, capacity, t.edge_number);
}

struct DiscretePairSnapshot
{
    std::array<uint32_t, 5> counts{};
    std::vector<std::array<int, 9>> pairs; // type + exact DCD int4 + exact CCD int4
};

template <typename Tree>
DiscretePairSnapshot discrete_query_snapshot(Tree& tree, double dHat)
{
    cudatool::DeviceBuffer<int4> pairs, ccd;
    cudatool::DeviceBuffer<int> indices;
    cudatool::DeviceBuffer<uint32_t> counts(5);
    DiscretePairSnapshot result;
    auto& stats = gipc::discrete_bvh_stats()[discrete_kind(tree)];
    uint32_t capacity = 0;
    for(;;)
    {
        CUDA_SAFE_CALL(cudaMemset(counts.data(), 0, 5 * sizeof(uint32_t)));
        discrete_raw_query(tree, dHat, pairs.data(), ccd.data(), counts.data(), indices.data(), capacity);
        ++stats.diagnostic_query_passes;
        CUDA_SAFE_CALL(cudaMemcpy(result.counts.data(), counts.data(), 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if(result.counts[0] <= capacity) break;
        if(result.counts[0] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Discrete BVH validation exceeds MatIndex signed range");
        // The first pass intentionally starts at zero to exercise the unchanged
        // count/grow/rerun path. Never allocate a configured worst-case pair limit.
        capacity = result.counts[0];
        pairs.resize_discard(capacity); ccd.resize_discard(capacity); indices.resize_discard(capacity);
        ++stats.diagnostic_overflow_retries;
        stats.diagnostic_peak_pair_capacity = std::max<uint64_t>(stats.diagnostic_peak_pair_capacity, capacity);
    }
    const auto& c = result.counts;
    if(uint64_t(c[2]) + c[3] + c[4] != c[0] || c[1] != 0)
        throw std::runtime_error("Discrete BVH validation: incomplete typed collision counts");
    pairs.resize(c[0]); ccd.resize(c[0]); indices.resize(c[0]);
    std::vector<int4> hp, hc;
    std::vector<int> hi;
    pairs.copy_to(hp); ccd.copy_to(hc); indices.copy_to(hi);
    std::array<std::vector<int>, 5> typed_indices;
    result.pairs.reserve(c[0]);
    for(size_t i = 0; i < hp.size(); ++i)
    {
        const auto p = hp[i], q = hc[i];
        const int type = (p.x >= 0 || p.y < 0) ? 4 : (p.z < 0 ? 2 : (p.w < 0 ? 3 : 4));
        typed_indices[type].push_back(hi[i]);
        result.pairs.push_back({type, p.x, p.y, p.z, p.w, q.x, q.y, q.z, q.w});
    }
    for(int type = 2; type <= 4; ++type)
    {
        auto& ranks = typed_indices[type];
        std::sort(ranks.begin(), ranks.end());
        if(ranks.size() != c[type])
            throw std::runtime_error("Discrete BVH validation: pair encoding/type count mismatch");
        for(size_t j = 0; j < ranks.size(); ++j)
            if(ranks[j] != static_cast<int>(j))
                throw std::runtime_error("Discrete BVH validation: MatIndex is not a complete type-local permutation");
    }
    // Atomic output order and the corresponding MatIndex rank may differ across
    // trees. Compare the exact typed multiset (retain duplicates), and require
    // every type's MatIndex to be a bijection onto its live matrix block range.
    std::sort(result.pairs.begin(), result.pairs.end());
    return result;
}

template <typename Tree>
void discrete_validate_query(Tree& tree, double dHat)
{
    if(!gipc::discrete_bvh_config().validate || !discrete_count(tree)) return;
    auto& stats = gipc::discrete_bvh_stats()[discrete_kind(tree)];
    ++stats.validation_calls;
    const auto start = std::chrono::steady_clock::now();
    gipc::CostSampleGuard sample("diagnostic_discrete_bvh_validation");
    gipc::CostScope scope("diagnostic.discrete_bvh_validation");
    try
    {
        discrete_check_tree(tree, false); // validate depth/links BEFORE refit/query
        tree.RefitDiscrete();            // force this path even on initial frames
        ++stats.diagnostic_refits;
        const auto refit_boxes = discrete_check_tree(tree, true);
        const auto refit = discrete_query_snapshot(tree, dHat);
        tree.ConstructRebuild();         // leaves live tree in a known rebuilt state
        ++stats.diagnostic_rebuilds;
        const auto rebuilt_boxes = discrete_check_tree(tree, true);
        const auto rebuilt = discrete_query_snapshot(tree, dHat);
        if(refit_boxes.size() != rebuilt_boxes.size()
           || !std::equal(refit_boxes.begin(), refit_boxes.end(), rebuilt_boxes.begin(), discrete_box_equal))
            throw std::runtime_error("Discrete BVH validation: refit/rebuild leaf bounds differ by original ID");
        if(refit.counts != rebuilt.counts || refit.pairs != rebuilt.pairs)
            throw std::runtime_error("Discrete BVH validation: refit/rebuild typed pair multisets differ");
        stats.diagnostic_pairs_compared += refit.pairs.size();
        ++stats.validation_passed;
    }
    catch(...)
    {
        ++stats.validation_failed;
        stats.diagnostic_host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        throw; // Never silently turn a failed diagnostic into a production fallback.
    }
    stats.diagnostic_host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
} // namespace
