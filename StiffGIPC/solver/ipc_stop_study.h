#pragma once
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gipc {
// Default-off research control. It owns stopping state, never physical state.
// Carrier changes no direction tolerance, objective, or accepted update.
struct IpcStopStudy {
    std::string mode="off",directory,load,save,phase="post_accept";
    int frame=0,after_movement=16,limit=512,stop_at=0,save_at=0;
    int first_movement=-1,first_toi=-1,initial_updates=0;
    static std::string env(const char* key,const char* fallback="") {
        const char* value=std::getenv(key);return value?value:fallback;
    }
    static int integer(const char* key,int fallback) {
        const auto value=env(key);if(value.empty())return fallback;
        size_t used=0;int result=std::stoi(value,&used);
        if(used!=value.size()||result<0)throw std::runtime_error(std::string("Invalid ")+key);
        return result;
    }
    static IpcStopStudy read(int physical_frame) {
        IpcStopStudy o;o.mode=env("GIPC_STOP_STUDY_MODE","off");
        if(o.mode!="off"&&o.mode!="observe"&&o.mode!="carrier"&&o.mode!="prescribed")
            throw std::runtime_error("Unknown stopping study mode");
        o.frame=integer("GIPC_STOP_STUDY_FRAME",0);
        if(o.mode!="off"&&o.frame<1)throw std::runtime_error("Stopping study needs one explicit physical frame");
        if(o.frame!=physical_frame)o.mode="off";
        if(!o.enabled())return o;
        o.directory=env("GIPC_STOP_STUDY_DIR");o.load=env("GIPC_STOP_STUDY_LOAD");o.save=env("GIPC_STOP_STUDY_SAVE");
        o.phase=env("GIPC_STOP_STUDY_PHASE","post_accept");
        o.after_movement=integer("GIPC_STOP_STUDY_TAIL",16);
        o.limit=integer("GIPC_STOP_STUDY_LIMIT",512);
        o.stop_at=integer("GIPC_STOP_STUDY_AT",0);o.save_at=integer("GIPC_STOP_STUDY_SAVE_AT",0);
        if(o.limit<1||o.limit>10000||o.after_movement>32||
           (o.phase!="post_accept"&&o.phase!="post_assembly")||
           (o.mode=="prescribed"&&(o.stop_at<1||o.stop_at>o.limit))||
           (!o.save.empty()&&(o.save_at<1||o.save_at>o.limit)))
            throw std::runtime_error("Invalid stopping study bounds");
        return o;
    }
    bool enabled()const{return mode!="off";}
    bool suppress()const{return mode=="carrier"||mode=="prescribed";}
    void observe_movement(bool candidate,int accepted) {
        if(candidate&&first_movement<0)first_movement=accepted;
    }
    void observe_toi(bool candidate,int accepted) {
        if(candidate&&first_toi<0)first_toi=accepted;
    }
    const char* exit(const char* event,int accepted)const {
        if(mode=="prescribed"&&phase==event&&accepted>=stop_at)return "study_prescribed";
        if(mode=="carrier"&&std::string(event)=="post_accept") {
            if(first_movement>=0&&accepted>=first_movement+after_movement)return "study_carrier_tail";
            if(accepted>=limit)return "study_carrier_limit";
        }
        return nullptr;
    }
    static double beta_after(double beta,double alpha,int iteration,int minimum) {
        if(!std::isfinite(alpha)||alpha<0||alpha>1||!std::isfinite(beta)||beta<0||beta>1)
            throw std::runtime_error("Invalid accepted alpha/beta");
        return iteration+1>=minimum?(1-alpha)*beta:beta;
    }
};
}
