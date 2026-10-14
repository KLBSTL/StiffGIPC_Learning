#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <gipc/utils/json.h>
class GIPC;

namespace gipc
{
// Independent execution-only opt-ins. Never inherit GIPC_ACCEL_SUITE.
struct P2Options
{
    bool lazy_scene = false;
    bool merge_abd_energy = false;
    bool audit = false;
    static P2Options read()
    {
        auto flag=[](const char* name) {
            const char* raw=std::getenv(name);
            if(!raw || std::string(raw)=="0") return false;
            if(std::string(raw)=="1") return true;
            throw std::runtime_error(std::string(name)+" must be exactly 0 or 1");
        };
        return {flag("GIPC_P2_LAZY_SCENE"),flag("GIPC_P2_MERGE_ABD_ENERGY"),flag("GIPC_P2_AUDIT")};
    }
    Json json() const
    {
        return {{"lazy_scene",lazy_scene},{"merge_abd_energy",merge_abd_energy},{"audit",audit},
                {"line_search_device_sync_retained",true},{"abd_host_addition_order_retained",true}};
    }
};
inline const P2Options& p2_options()
{
    static const auto options=P2Options::read();
    return options;
}
int p2_scene_fixture(const char* output);
int p2_abd_fixture(GIPC& owner,const char* output);
} // namespace gipc
