#pragma once
#include <string_view>
#include <cstdlib>
#include <string>

namespace gipc
{
constexpr auto assets_dir()
{
    return std::string_view{GIPC_ASSETS_DIR};
}

inline std::string output_dir()
{
    if(const char* path = std::getenv("GIPC_OUTPUT_PATH")) return path;
    return GIPC_OUTPUT_DIR;
}
}  // namespace gipc
