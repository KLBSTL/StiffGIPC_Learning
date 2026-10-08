#include <linear_system/utils/linear_structure_probe.h>
#include <iostream>
#include <stdexcept>

using namespace gipc;
static_assert(!LinearStructureProbeConfig{}.enabled);
static_assert(LinearStructureProbeConfig{}.include_converted_structure);
static_assert(linear_structure_key(1, 2) != linear_structure_key(2, 1));
static_assert(linear_structure_key(-1, 0) != linear_structure_key(0, -1));
static_assert(linear_structure_key(-1, -1) == UINT64_MAX);
static_assert(linear_structure_key(std::numeric_limits<std::int32_t>::min(),
                                  std::numeric_limits<std::int32_t>::max()) == 0x800000007fffffffull);

void require(bool condition)
{
    if(!condition) throw std::runtime_error("Linear structure CPU contract failed");
}
int main()
try
{
    int checks = 0;
    for(std::size_t count : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u})
    {
        LinearStructureShape shape{1, 3, count, 300, 0, 17, 19};
        validate_linear_structure_shape(shape);
        require(linear_structure_plan(false, false, {}, shape) == LinearStructureState::disabled);
        require(linear_structure_plan(false, true, shape, shape) == LinearStructureState::disabled);
        require(linear_structure_plan(true, false, {}, shape) == LinearStructureState::first);
        require(linear_structure_plan(true, true, shape, shape)
                == (count ? LinearStructureState::compare : LinearStructureState::hit));
        auto changed = shape;
        changed.owner = 2;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::owner_changed);
        changed = shape; ++changed.item_count;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::shape_changed);
        changed = shape; ++changed.input_offset;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::shape_changed);
        changed = shape; ++changed.output_offset;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::shape_changed);
        changed = shape; ++changed.device;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::shape_changed);
        changed = shape; ++changed.block_rows;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::shape_changed);
        changed = shape; ++changed.block_cols;
        require(linear_structure_plan(true, true, shape, changed) == LinearStructureState::shape_changed);
        checks += 11;
    }
    for(int field = 0; field < 5; ++field)
    {
        auto invalid = LinearStructureShape{};
        const auto outside_int = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
        if(field == 0) invalid.item_count = outside_int;
        if(field == 1) invalid.input_offset = outside_int;
        if(field == 2) invalid.output_offset = outside_int;
        if(field == 3) invalid.block_rows = -1;
        if(field == 4) invalid.block_cols = -1;
        bool rejected = false;
        try { validate_linear_structure_shape(invalid); }
        catch(const std::runtime_error&) { rejected = true; }
        require(rejected);
        require(linear_structure_plan(false, true, {}, invalid) == LinearStructureState::disabled);
        checks += 2;
    }
    std::cout << "{\"passed\":true,\"checks\":" << checks << "}\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
