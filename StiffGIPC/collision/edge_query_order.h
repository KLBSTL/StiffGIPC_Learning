#pragma once
#include <gipc/utils/json.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace gipc
{
// Independent IPC-only option. Never inherits GIPC_ACCEL_SUITE.
struct EdgeQueryOrderOptions
{
    bool leaf = false;
    std::vector<int> probe_frames;
    std::string probe_file;
    static EdgeQueryOrderOptions read()
    {
        EdgeQueryOrderOptions o;
        if(const char* p=std::getenv("GIPC_EDGE_QUERY_ORDER"))
        {
            const std::string value=p;
            if(value!="raw" && value!="leaf")
                throw std::runtime_error("GIPC_EDGE_QUERY_ORDER must be raw or leaf");
            o.leaf=value=="leaf";
        }
        if(const char* p=std::getenv("GIPC_EDGE_ORDER_PROBE_FRAMES"))
        {
            const std::string value=p;
            if(value.empty())throw std::runtime_error("Empty edge probe frame list");
            size_t begin=0;
            while(begin<value.size())
            {
                const size_t comma=value.find(',',begin);
                const std::string token=value.substr(begin,comma==std::string::npos?comma:comma-begin);
                int frame=token=="1"?1:token=="41"?41:token=="57"?57:-1;
                if(frame<0 || std::find(o.probe_frames.begin(),o.probe_frames.end(),frame)!=o.probe_frames.end())
                    throw std::runtime_error("Edge probe frames must be a unique subset of 1,41,57");
                o.probe_frames.push_back(frame);
                if(comma==std::string::npos)break;
                begin=comma+1;
                if(begin==value.size())throw std::runtime_error("Trailing comma in edge probe frames");
            }
        }
        if(const char* p=std::getenv("GIPC_EDGE_ORDER_PROBE_FILE"))o.probe_file=p;
        if(o.probe_frames.empty()!=o.probe_file.empty())
            throw std::runtime_error("Edge probe frames and output file must be supplied together");
        return o;
    }
    void require_ipc(bool ipc) const
    {
        if(!ipc && (leaf || !probe_frames.empty()))
            throw std::runtime_error("Edge query order/probe is IPC-only");
    }
    bool selected(int frame) const
    {
        return std::find(probe_frames.begin(),probe_frames.end(),frame)!=probe_frames.end();
    }
    Json json(bool ipc) const
    {
        require_ipc(ipc);
        return {{"requested",leaf?"leaf":"raw"},{"effective",leaf?"leaf":"raw"},
            {"default","raw"},{"scope","ipc_discrete_edge_triangle_query_face_order_only"},
            {"probe_frames",probe_frames},{"probe_file",probe_file},
            {"probe_pairs",7},{"probe_first_query_only",true},
            {"probe_changes_production_selection",false},
            {"zero_or_one_edge","explicit_common_boundary_not_speed_evidence"}};
    }
};
inline const EdgeQueryOrderOptions& edge_query_order_options()
{
    static const EdgeQueryOrderOptions options=EdgeQueryOrderOptions::read();
    return options;
}
// Early standalone GPU fixture, before scene/OpenGL initialization.
int edge_query_order_fixture(const char* output);
} // namespace gipc
