#pragma once
#include <gipc/utils/json.h>
#include <stdexcept>
#include <string>

namespace gipc
{
// Armed only by the protected fixed-system study, never by production PCG.
struct MasStageProbeRequest
{
    std::string prefix;
    Json result;
    bool dispatched=false;
    static MasStageProbeRequest*& active()
    {
        thread_local static MasStageProbeRequest* request=nullptr;
        return request;
    }
    explicit MasStageProbeRequest(const std::string& path):prefix(path)
    {
        if(active())throw std::runtime_error("Nested MAS stage probe");
        active()=this;
    }
    ~MasStageProbeRequest(){active()=nullptr;}
    MasStageProbeRequest(const MasStageProbeRequest&)=delete;
    MasStageProbeRequest& operator=(const MasStageProbeRequest&)=delete;
};
}
