#include <collision/p3_ee.h>
#include <iostream>
namespace
{
void set(const char* key,const char* value)
{
#ifdef _WIN32
    _putenv_s(key,value?value:"");
#else
    if(value)setenv(key,value,1);else unsetenv(key);
#endif
}
void need(bool ok){if(!ok)throw std::runtime_error("P3 options contract failed");}
template<class F> void reject(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}need(failed);}
}
int main()
{
    try
    {
        for(const char* name:{"GIPC_P3_EE_MODE","GIPC_P3_EE_QUEUE_CAPACITY","GIPC_P3_EE_STUDY_FRAMES","GIPC_P3_EE_STUDY_FILE"})set(name,nullptr);
        const auto defaults=gipc::P3EEOptions::read();need(defaults.mode=="fused"&&!defaults.active()&&defaults.queue_capacity==(1u<<20));
        defaults.require_context(false,true,true); // inactive module does not restrict other paths
        unsigned positions[4][2]{};unsigned first_f=0,first_w=0;
        for(unsigned i=0;i<gipc::p3_ee_timing_samples;++i)
        {
            ++positions[i%4][gipc::p3_ee_timing_split(i)];
            if(i%2==0)
            {
                need(gipc::p3_ee_timing_split(i)!=gipc::p3_ee_timing_split(i+1));
                if(gipc::p3_ee_timing_split(i))++first_w;else ++first_f;
            }
        }
        need(first_f==8&&first_w==8);
        for(const auto& p:positions)need(p[0]==4&&p[1]==4);
        set("GIPC_ACCEL_SUITE","1");need(!gipc::P3EEOptions::read().active());
        set("GIPC_P3_EE_MODE","adaptive");reject([]{gipc::P3EEOptions::read();});
        set("GIPC_P3_EE_MODE","two_stage");const auto split=gipc::P3EEOptions::read();need(split.active());
        split.require_context(true,false,false);
        reject([&]{split.require_context(false,false,false);});
        reject([&]{split.require_context(true,true,false);});
        reject([&]{split.require_context(true,false,true);});
        gipc::p3_ee_require_edge_count(0);gipc::p3_ee_require_edge_count(1u<<30);
        reject([]{gipc::p3_ee_require_edge_count((1u<<30)+1);});
        reject([]{gipc::p3_ee_require_edge_count(0xffffffffu);});
        for(const char* value:{"0","-1","1.5","8388609","18446744073709551616"})
        {set("GIPC_P3_EE_QUEUE_CAPACITY",value);reject([]{gipc::P3EEOptions::read();});}
        set("GIPC_P3_EE_QUEUE_CAPACITY","8388608");need(gipc::P3EEOptions::read().queue_capacity==(1u<<23));
        set("GIPC_P3_EE_STUDY_FRAMES","17");reject([]{gipc::P3EEOptions::read();});
        set("GIPC_P3_EE_STUDY_FILE","private.jsonl");set("GIPC_P3_EE_STUDY_FRAMES","17,15,14");
        auto selected=gipc::P3EEOptions::read();need(selected.selected(17)&&!selected.selected(1));
        // Windows _putenv_s(key, "") REMOVES a variable; it cannot represent
        // getenv returning a present empty string. Test that platform contract
        // explicitly, and test the empty-string rejection on POSIX only.
        set("GIPC_P3_EE_STUDY_FRAMES","");
#ifdef _WIN32
        need(gipc::P3EEOptions::read().frames.empty());
#else
        reject([]{gipc::P3EEOptions::read();});
#endif
        for(const char* value:{"0","1,1","1,","1,,2","1-3","1000001"})
        {set("GIPC_P3_EE_STUDY_FRAMES",value);reject([]{gipc::P3EEOptions::read();});}
        std::cout<<"P3 option contracts passed\n";return 0;
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
