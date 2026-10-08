#include <linear_system/utils/converter.h>
#include <linear_system/utils/linear_structure_probe.cuh>
#include <gipc/utils/timer.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gipc;

namespace
{
int checks = 0, cases = 0;
void require(bool condition, const char* label)
{
    ++checks;
    if(!condition) throw std::runtime_error(label);
}
void cuda_check(cudaError_t error)
{
    if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class Operation> void require_rejected(Operation operation, const char* label)
{
    bool rejected = false;
    try { operation(); }
    catch(const std::runtime_error&) { rejected = true; }
    require(rejected, label);
}
template<class T> struct DeviceArray
{
    T* data = nullptr;
    std::size_t size;
    explicit DeviceArray(const std::vector<T>& source) : size(source.size())
    {
        cuda_check(cudaMalloc(&data, size * sizeof(T)));
        upload(source);
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if(data) cudaFree(data); }
    void upload(const std::vector<T>& source)
    {
        require(source.size() == size, "fixture upload size changed");
        cuda_check(cudaMemcpyAsync(data, source.data(), size * sizeof(T),
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    }
    std::vector<T> read() const
    {
        std::vector<T> result(size);
        cuda_check(cudaMemcpyAsync(result.data(), data, size * sizeof(T),
                                   cudaMemcpyDeviceToHost, cudaStreamPerThread));
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
        return result;
    }
};
struct Inputs
{
    static constexpr std::size_t front = 3, tail = 7;
    std::size_t count, unique_count;
    std::vector<int> rows, cols;
    std::vector<std::uint32_t> mapping, partition;
    DeviceArray<int> d_rows, d_cols;
    DeviceArray<std::uint32_t> d_mapping, d_partition;
    explicit Inputs(std::size_t n)
        : count(n), unique_count((n + 2) / 3),
          rows(n + front + tail, -1000001), cols(n + front + tail, -1000002),
          mapping(n + front + tail, 0xfefefefeu), partition(n + front + tail, 0xfdfdfdfdu),
          d_rows(rows), d_cols(cols), d_mapping(mapping), d_partition(partition)
    {
        for(std::size_t i = 0; i < count; ++i)
        {
            rows[front + i] = static_cast<int>(i / 43);
            cols[front + i] = static_cast<int>(i % 43);
            mapping[front + i] = static_cast<std::uint32_t>(count - 1 - i);
            partition[front + i] = static_cast<std::uint32_t>(i / 3);
        }
        upload();
    }
    void upload()
    {
        d_rows.upload(rows); d_cols.upload(cols);
        d_mapping.upload(mapping); d_partition.upload(partition);
    }
    void verify_read_only() const
    {
        require(d_rows.read() == rows, "probe changed borrowed rows or a guard");
        require(d_cols.read() == cols, "probe changed borrowed cols or a guard");
        require(d_mapping.read() == mapping, "probe changed borrowed mapping or a guard");
        require(d_partition.read() == partition, "probe changed borrowed partition or a guard");
    }
};
LinearStructureProbeResult result(const LinearStructureProbe& probe)
{
    require(probe.last_result() != nullptr, "missing enabled probe result");
    return *probe.last_result();
}
LinearStructureProbeResult observe(LinearStructureProbe& probe, const Inputs& data,
                                   std::uintptr_t owner = 17, int extra_rows = 0)
{
    probe.observe_raw(owner, 7, data.count, data.count + 19,
                      data.count ? data.d_rows.data + Inputs::front : nullptr,
                      data.count ? data.d_cols.data + Inputs::front : nullptr,
                      static_cast<int>(data.count) + 64 + extra_rows, 64);
    probe.observe_converted(data.count ? data.d_mapping.data + Inputs::front : nullptr,
                            data.count ? data.d_partition.data + Inputs::front : nullptr,
                            data.unique_count, nullptr, nullptr, false);
    return result(probe);
}
void require_cost(const LinearStructureStageCost& cost, bool ran, bool events, bool comparison)
{
    require(std::isfinite(cost.cpu_ms) && cost.cpu_ms >= 0, "invalid CPU stage cost");
    require(std::isfinite(cost.gpu_ms) && cost.gpu_ms >= 0, "invalid GPU stage cost");
    require(cost.gpu_valid == (ran && events), "GPU event validity contract");
    require(cost.completion_waited == (ran && (events || comparison)), "stage wait contract");
}
void disabled_contract()
{
    // These invalid spans must be ignored before device initialization. This
    // uses the production helper; there is no replacement comparison kernel.
    LinearStructureProbe probe;
    probe.observe_raw(17, SIZE_MAX, SIZE_MAX, SIZE_MAX, nullptr, nullptr, -1, -1);
    probe.observe_converted(nullptr, nullptr, SIZE_MAX);
    probe.reset();
    probe.configure({false, true, true});
    require(probe.last_result() == nullptr, "disabled probe produced a result");
    probe.configure({true, false, true});
    require_rejected([&] { probe.observe_raw(17, 0, 1, 2, nullptr, nullptr); }, "null raw span accepted");
    require_rejected([&] { probe.observe_converted(nullptr, nullptr, 0); }, "converted span without raw accepted");
    require_rejected([&] { probe.observe_raw(17, 0, SIZE_MAX, 2, nullptr, nullptr); }, "overflow raw count accepted");
    probe.configure({false, false, false});
    ++cases;
}
void direct_fixture(std::size_t count, bool events)
{
    LinearStructureProbe probe;
    probe.configure({true, events, true});
    Inputs data(count);
    auto first = observe(probe, data);
    require(first.raw_state == LinearStructureState::first && first.converted_state == LinearStructureState::first,
            "initial observation is not first");
    require(!first.gap_before && first.observation_index == 1, "initial observation metadata");
    require(!first.raw_comparison_performed && !first.converted_comparison_performed, "compared first snapshot");
    require(first.raw_snapshot_updated == (count != 0) && first.converted_snapshot_updated == (count != 0),
            "first snapshot copy contract");
    require(first.raw_snapshot_bytes == count * 8 && first.converted_snapshot_bytes == count * 8, "snapshot byte count");
    require_cost(first.raw_snapshot, count != 0, events, false);
    require_cost(first.converted_snapshot, count != 0, events, false);
    auto same = observe(probe, data);
    require(same.raw_state == LinearStructureState::hit && same.converted_state == LinearStructureState::hit,
            "equal ordered spans missed");
    require(same.raw_equal && same.mapping_equal && same.partition_equal, "equal flags incorrect");
    require(same.raw_comparison_performed == (count != 0) && same.converted_comparison_performed == (count != 0),
            "full exact comparison not performed");
    require_cost(same.raw_compare, count != 0, events, true);
    require_cost(same.converted_compare, count != 0, events, true);
    require_rejected([&] { probe.observe_converted(nullptr, nullptr, 0); }, "duplicate converted observation accepted");
    if(count)
    {
        ++data.cols[Inputs::front + count - 1]; data.d_cols.upload(data.cols);
        auto key_changed = observe(probe, data);
        require(key_changed.raw_state == LinearStructureState::content_changed && !key_changed.raw_equal,
                "one raw tail key change missed");
        require(key_changed.raw_snapshot_updated, "changed raw snapshot not replaced");
        require(observe(probe, data).raw_equal, "replacement raw snapshot did not hit");
        DeviceArray<int> relocated_rows(data.rows), relocated_cols(data.cols);
        probe.observe_raw(17, 7, count, count + 19,
                          relocated_rows.data + Inputs::front, relocated_cols.data + Inputs::front,
                          static_cast<int>(count) + 64, 64);
        probe.observe_converted(data.d_mapping.data + Inputs::front, data.d_partition.data + Inputs::front,
                                data.unique_count, nullptr, nullptr, false);
        require(result(probe).raw_equal, "same keys in new borrowed storage missed");
        require(result(probe).row_storage != first.row_storage, "relocation fixture reused the same storage");
        if(count > 1)
        {
            std::swap(data.rows[Inputs::front], data.rows[Inputs::front + count - 1]);
            std::swap(data.cols[Inputs::front], data.cols[Inputs::front + count - 1]);
            data.d_rows.upload(data.rows); data.d_cols.upload(data.cols);
            require(observe(probe, data).raw_state == LinearStructureState::content_changed,
                    "same-count permutation of raw keys was treated as equal");
        }
        // Standalone diagnostic identity injections deliberately change just
        // one index value; they never feed a production matrix or solver.
        ++data.mapping[Inputs::front + count - 1]; data.d_mapping.upload(data.mapping);
        auto map_changed = observe(probe, data);
        require(map_changed.converted_state == LinearStructureState::content_changed
                && !map_changed.mapping_equal && map_changed.partition_equal, "one mapping value change missed");
        ++data.partition[Inputs::front + count - 1]; data.d_partition.upload(data.partition);
        auto partition_changed = observe(probe, data);
        require(partition_changed.mapping_equal && !partition_changed.partition_equal, "one partition value change missed");
        require(observe(probe, data).converted_state == LinearStructureState::hit, "converted snapshots alias borrowed scratch");
        const auto previous_unique = data.unique_count;
        data.unique_count = previous_unique == count ? 0 : previous_unique + 1;
        require(observe(probe, data).converted_state == LinearStructureState::shape_changed,
                "converted unique count change missed");
        require(observe(probe, data).converted_state == LinearStructureState::hit,
                "changed converted shape did not become the next reference");
        data.unique_count = previous_unique;
        require(observe(probe, data).converted_state == LinearStructureState::shape_changed,
                "restored converted unique count change missed");
    }
    auto owner_changed = observe(probe, data, 18);
    require(owner_changed.raw_state == LinearStructureState::owner_changed
            && owner_changed.converted_state == LinearStructureState::owner_changed, "owner change missed");
    require(!owner_changed.raw_comparison_performed && !owner_changed.converted_comparison_performed, "compared different owners");
    auto shape_changed = observe(probe, data, 18, 1);
    require(shape_changed.raw_state == LinearStructureState::shape_changed
            && shape_changed.converted_state == LinearStructureState::shape_changed, "same-count matrix dimension change missed");
    probe.configure({false, events, true});
    probe.observe_raw(99, SIZE_MAX, SIZE_MAX, SIZE_MAX, nullptr, nullptr, -1, -1);
    probe.observe_converted(nullptr, nullptr, SIZE_MAX);
    require(probe.last_result() == nullptr, "disabled probe retained a readable result");
    probe.configure({true, events, true});
    auto resumed = observe(probe, data, 18, 1);
    require(resumed.gap_before && resumed.raw_state == LinearStructureState::first
            && resumed.converted_state == LinearStructureState::first, "hit claimed across an unobserved gap");
    require(!observe(probe, data, 18, 1).gap_before, "gap remained after first resumed observation");
    probe.configure({true, events, false});
    probe.observe_raw(18, 7, count, count + 19,
                      count ? data.d_rows.data + Inputs::front : nullptr,
                      count ? data.d_cols.data + Inputs::front : nullptr, static_cast<int>(count) + 65, 64);
    probe.observe_converted(nullptr, nullptr, SIZE_MAX);
    require(!result(probe).converted_observed, "disabled converted stage ran");
    probe.configure({true, events, true});
    auto converted_resumed = observe(probe, data, 18, 1);
    require(converted_resumed.raw_equal && converted_resumed.converted_state == LinearStructureState::first,
            "converted hit claimed across an unobserved converted system");
    probe.reset();
    require(observe(probe, data, 18, 1).gap_before, "explicit reset omitted the gap");
    data.verify_read_only();
    ++cases;
}
void capacity_fixture(bool events)
{
    LinearStructureProbe probe;
    probe.configure({true, events, true});
    std::size_t raw_capacity = 0, converted_capacity = 0;
    int pass = 0;
    for(std::size_t count : {1u, 257u, 33u, 257u})
    {
        Inputs data(count);
        auto current = observe(probe, data);
        require(current.raw_state == (pass++ == 0 ? LinearStructureState::first : LinearStructureState::shape_changed),
                "capacity/shape transition state");
        require(current.raw_capacity_bytes >= count * 8 && current.converted_capacity_bytes >= count * 8,
                "snapshot capacity below full span size");
        require(current.raw_capacity_bytes >= raw_capacity && current.converted_capacity_bytes >= converted_capacity,
                "snapshot capacity unexpectedly shrank");
        raw_capacity = current.raw_capacity_bytes; converted_capacity = current.converted_capacity_bytes;
        require(observe(probe, data).raw_equal, "capacity transition snapshot was not complete");
        data.verify_read_only();
    }
    ++cases;
}
void canonical_pair_fixture(bool events)
{
    LinearStructureProbe probe;
    probe.configure({true, events, true});
    auto observe_pairs = [&](std::size_t count, const std::vector<int>& pair_rows,
                             const std::vector<int>& pair_cols, bool reorder = false,
                             bool requested = true, std::uintptr_t owner = 17, int extra_rows = 0)
    {
        Inputs data(count);
        std::vector<int> compact_rows(pair_rows.size() + Inputs::front + Inputs::tail, -2000001);
        std::vector<int> compact_cols(pair_cols.size() + Inputs::front + Inputs::tail, -2000002);
        std::copy(pair_rows.begin(), pair_rows.end(), compact_rows.begin() + Inputs::front);
        std::copy(pair_cols.begin(), pair_cols.end(), compact_cols.begin() + Inputs::front);
        for(std::size_t i = 0; i < count; ++i)
        {
            data.rows[Inputs::front + i] = pair_rows[i % pair_rows.size()];
            data.cols[Inputs::front + i] = pair_cols[i % pair_cols.size()];
        }
        if(reorder && count > 1)
        {
            std::swap(data.rows[Inputs::front], data.rows[Inputs::front + count - 1]);
            std::swap(data.cols[Inputs::front], data.cols[Inputs::front + count - 1]);
        }
        std::vector<std::uint32_t> order(count);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b)
        {
            return linear_structure_key(data.rows[Inputs::front + a], data.cols[Inputs::front + a])
                < linear_structure_key(data.rows[Inputs::front + b], data.cols[Inputs::front + b]);
        });
        std::uint32_t partition = 0;
        for(std::size_t i = 0; i < count; ++i)
        {
            data.mapping[Inputs::front + i] = order[i];
            if(i && linear_structure_key(data.rows[Inputs::front + order[i]], data.cols[Inputs::front + order[i]])
                != linear_structure_key(data.rows[Inputs::front + order[i - 1]], data.cols[Inputs::front + order[i - 1]]))
                ++partition;
            data.partition[Inputs::front + i] = partition;
        }
        data.unique_count = pair_rows.size(); data.upload();
        DeviceArray<int> d_rows(compact_rows), d_cols(compact_cols);
        probe.observe_raw(owner, 7, count, count + 19,
                          count ? data.d_rows.data + Inputs::front : nullptr,
                          count ? data.d_cols.data + Inputs::front : nullptr, 64 + extra_rows, 64);
        probe.observe_converted(count ? data.d_mapping.data + Inputs::front : nullptr,
                                count ? data.d_partition.data + Inputs::front : nullptr, pair_rows.size(),
                                pair_rows.empty() ? nullptr : d_rows.data + Inputs::front,
                                pair_cols.empty() ? nullptr : d_cols.data + Inputs::front, requested);
        const auto current = result(probe);
        data.verify_read_only();
        require(d_rows.read() == compact_rows && d_cols.read() == compact_cols, "canonical pair input or guard changed");
        require_cost(current.pair_compare, current.pair_comparison_performed, events, true);
        require_cost(current.pair_snapshot, current.pair_snapshot_updated, events, false);
        return current;
    };
    std::vector<int> rows{0, 1, 2}, cols{0, 2, 4};
    const auto first = observe_pairs(5, rows, cols);
    require(first.pair_observed && first.pair_state == LinearStructureState::first && !first.pair_gap_before,
            "initial canonical pair observation contract");
    const auto reordered = observe_pairs(5, rows, cols, true);
    require(reordered.raw_state == LinearStructureState::content_changed && reordered.pair_equal
            && reordered.pair_comparison_performed, "canonical identity did not survive raw reordering");
    const auto multiplicity_changed = observe_pairs(17, rows, cols);
    require(multiplicity_changed.raw_state == LinearStructureState::shape_changed && multiplicity_changed.pair_equal,
            "canonical pair shape incorrectly depends on raw count or staging offset");
    ++cols[1];
    const auto key_changed = observe_pairs(17, rows, cols);
    require(key_changed.pair_state == LinearStructureState::content_changed && !key_changed.pair_equal
            && key_changed.pair_snapshot_updated, "single canonical key change missed");
    require(observe_pairs(17, rows, cols).pair_equal, "canonical snapshots alias borrowed indices");
    require(!observe_pairs(17, rows, cols, false, false).pair_observed, "explicitly omitted pair observation ran");
    const auto resumed = observe_pairs(17, rows, cols);
    require(resumed.pair_state == LinearStructureState::first && resumed.pair_gap_before,
            "canonical pair hit claimed across omitted pair observation");
    require(observe_pairs(17, rows, cols, false, true, 18).pair_state == LinearStructureState::owner_changed,
            "canonical owner change missed");
    require(observe_pairs(17, rows, cols, false, true, 18, 1).pair_state == LinearStructureState::shape_changed,
            "canonical matrix dimension change missed");
    const auto empty = observe_pairs(0, {}, {}, false, true, 18, 1);
    require(empty.pair_observed && empty.pair_state == LinearStructureState::shape_changed
            && !empty.pair_comparison_performed && !empty.pair_snapshot_updated, "zero unique pair contract");
    const auto empty_hit = observe_pairs(0, {}, {}, false, true, 18, 1);
    require(empty_hit.pair_equal && !empty_hit.pair_comparison_performed, "empty pair identity contract");
    rows = {0, 1, 2, 3, 4, 5}; cols = {0, 3, 4, 5, 6, 7};
    const auto grown = observe_pairs(33, rows, cols, false, true, 18, 1);
    require(grown.pair_state == LinearStructureState::shape_changed && grown.pair_snapshot_bytes == 48
            && grown.pair_capacity_bytes > first.pair_capacity_bytes, "canonical snapshot capacity growth contract");
    require(observe_pairs(33, rows, cols, false, true, 18, 1).pair_equal, "grown canonical snapshot incomplete");
    ++cases;
}
template<class T> std::vector<T> read_span(const T* source, std::size_t count)
{
    std::vector<T> result(count);
    if(count)
    {
        cuda_check(cudaMemcpyAsync(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    }
    return result;
}
void converter_fixture(int count, bool events)
{
    // Exercise the actual Converter hooks. Reassembly uploads the original raw
    // keys before every conversion; conversion itself overwrites the source
    // prefix with unique keys, which must not corrupt the independently owned
    // raw snapshot. Numerical blocks change on the second observation.
    Converter converter;
    converter.configure_structure_probe({true, events, true});
    GIPCTripletMatrix matrix;
    matrix.reshape(64, 64);
    const int start = 3, staging = start + count + 11, size = staging + count + 7;
    matrix.resize_triplets(size);
    matrix.resize_conversion_scratch(count);
    // The production exclusive scan ignores the final flag logically. Make
    // every scratch input initialized in this fixture, including that flag.
    matrix.m_block_temp_buffer.reset_zero();
    for(int round = 0; round < 3; ++round)
    {
        std::vector<int> rows(size, -1000001), cols(size, -1000002);
        std::vector<Eigen::Matrix3d> values(size, Eigen::Matrix3d::Constant(-999));
        std::vector<std::uint32_t> order(count);
        std::iota(order.begin(), order.end(), 0u);
        for(int i = 0; i < count; ++i)
        {
            rows[start + i] = (i * 13) % 17; cols[start + i] = (i * 7) % 19;
            values[start + i] = Eigen::Matrix3d::Constant((i % 9 + 1) * .03125 * (round + 1));
            values[start + i].diagonal().array() += .5 * (round + 1);
        }
        if(round == 2 && count > 1)
        {
            std::swap(rows[start], rows[start + count - 1]);
            std::swap(cols[start], cols[start + count - 1]);
            std::swap(values[start], values[start + count - 1]);
        }
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b)
        {
            return linear_structure_key(rows[start + a], cols[start + a])
                < linear_structure_key(rows[start + b], cols[start + b]);
        });
        std::vector<std::uint32_t> expected_partition(count);
        std::vector<std::uint64_t> unique_keys;
        std::vector<Eigen::Matrix3d> sums;
        for(int i = 0; i < count; ++i)
        {
            const auto source = order[i];
            const auto key = linear_structure_key(rows[start + source], cols[start + source]);
            if(unique_keys.empty() || unique_keys.back() != key)
            { unique_keys.push_back(key); sums.emplace_back(Eigen::Matrix3d::Zero()); }
            expected_partition[i] = static_cast<std::uint32_t>(unique_keys.size() - 1);
            sums.back() += values[start + source];
        }
        matrix.m_block_row_indices.copy_from(rows); matrix.m_block_col_indices.copy_from(cols);
        matrix.m_block_values.copy_from(values);
        converter.convert(matrix, start, count, staging);
        const auto* observed = converter.last_structure_probe();
        require(observed != nullptr, "Converter did not expose the enabled probe");
        const auto repeated_state = round == 2 && count > 1 ? LinearStructureState::content_changed : LinearStructureState::hit;
        require(observed->raw_state == (round == 0 ? LinearStructureState::first : repeated_state),
                "Converter probe observed sorted keys instead of complete ordered raw keys");
        require(observed->converted_state == (round == 0 ? LinearStructureState::first : repeated_state),
                "Converter mapping/partition identity missed");
        require(observed->pair_observed && observed->pair_state
                == (round == 0 ? LinearStructureState::first : LinearStructureState::hit),
                "canonical pair identity should survive raw reordering");
        require(observed->pair_comparison_performed == (round != 0 && count != 0), "canonical pair exact comparison contract");
        require(observed->shape.block_rows == 64 && observed->shape.block_cols == 64,
                "Converter did not forward matrix dimensions");
        require(observed->unique_count == unique_keys.size(), "Converter diagnostic unique count incorrect");
        if(count)
        {
            require(matrix.h_unique_key_number == static_cast<int>(unique_keys.size()), "production unique count incorrect");
            require(read_span(matrix.block_sort_index(), count) == order, "production sort mapping differs from CPU stable sort");
            require(read_span(matrix.block_index(), count) == expected_partition, "production partition differs from CPU reference");
        }
        const auto actual_rows = read_span(matrix.block_row_indices(), size);
        const auto actual_cols = read_span(matrix.block_col_indices(), size);
        const auto actual_values = read_span(matrix.block_values(), size);
        for(std::size_t i = 0; i < unique_keys.size(); ++i)
        {
            require(linear_structure_key(actual_rows[start + i], actual_cols[start + i]) == unique_keys[i], "production unique key mismatch");
            require((actual_values[start + i].array() == sums[i].array()).all(), "production numeric blocks were stale or changed");
        }
        for(int i = 0; i < size; ++i)
            if(i < start || i >= start + static_cast<int>(unique_keys.size()))
                require(actual_rows[i] == rows[i] && actual_cols[i] == cols[i], "Converter index guard overwritten");
        for(int i = 0; i < size; ++i)
            if(i < start || i >= staging + count)
                require((actual_values[i].array() == values[i].array()).all(), "Converter numeric outer guard overwritten");
    }
    converter.configure_structure_probe({false, events, true});
    require(converter.last_structure_probe() == nullptr, "disabled Converter exposed a stale result");
    ++cases;
}
} // namespace

int main()
try
{
    disabled_contract();
    int devices = 0;
    const auto available = cudaGetDeviceCount(&devices);
    if(available == cudaErrorNoDevice || (available == cudaSuccess && devices == 0))
    { std::cout << "{\"skipped\":true,\"reason\":\"no CUDA device\"}\n"; return 77; }
    cuda_check(available); cuda_check(cudaSetDevice(0));
    gipc::Timer::disable_all();
    for(bool events : {false, true})
    {
        for(std::size_t count : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 4097u})
            direct_fixture(count, events);
        capacity_fixture(events);
        canonical_pair_fixture(events);
        for(int count : {0, 1, 31, 32, 33, 255, 256, 257, 4097})
            converter_fixture(count, events);
    }
    cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    std::cout << "{\"passed\":true,\"cases\":" << cases << ",\"checks\":" << checks << "}\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
