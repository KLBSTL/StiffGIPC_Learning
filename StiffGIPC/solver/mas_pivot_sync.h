#pragma once
#include <cstdlib>
namespace gipc
{
inline bool mas_pivot_only_sync()
{
    static const bool requested=[] {
        const char* value=std::getenv("GIPC_MAS_PIVOT_ONLY_SYNC");
        return value && value[0]=='1' && value[1]=='\0';
    }();
    return requested;
}
int run_mas_pivot_sync_fixture(const char* output);
}
