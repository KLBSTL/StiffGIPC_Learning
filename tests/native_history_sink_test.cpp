#include <linear_system/solver/native_history_sink.h>
#include <algorithm>
#include <limits>
#include <set>
#include <iostream>

static void require(bool condition){if(!condition)throw std::runtime_error("Native history contract failed");}
template<class F>static void rejects(F f){bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}require(rejected);}
static void env(const char* value){
#ifdef _WIN32
    _putenv_s("GIPC_FIXED_NATIVE_HISTORY_STUDY",value);
#else
    setenv("GIPC_FIXED_NATIVE_HISTORY_STUDY",value,1);
#endif
}
int main(){try{
    using gipc::NativePcgHistory;
    std::vector<int> copied(128,-1);int current=0,calls=0;
    NativePcgHistory sink([&](int slot){require(slot>=0&&slot<128);copied[slot]=current;++calls;});
    for(current=0;current<220;++current)sink.observe(current,.5,1.,2.);
    std::set<int> expected,observed(sink.slot_iterations.begin(),sink.slot_iterations.end());
    for(int i=0;i<16;++i)expected.insert(i);for(int i=108;i<220;++i)expected.insert(i);
    require(expected==observed&&copied==sink.slot_iterations&&calls==220&&sink.rows.size()==220);
    rejects([&]{sink.observe(221,.5,1.,2.);});require(calls==220&&sink.rows.size()==220);
    for(double bad:{0.,-1.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
        rejects([&]{sink.observe(220,.5,bad,2.);});
    NativePcgHistory failed([](int){throw std::runtime_error("copy failure");});
    rejects([&]{failed.observe(0,.5,1.,2.);});require(failed.rows.empty()&&failed.slot_iterations[0]==-1);
    NativePcgHistory budget([](int){});for(int i=0;i<2000;++i)budget.observe(i,.5,1.,2.);
    rejects([&]{budget.observe(2000,.5,1.,2.);});require(budget.rows.size()==2000);
    rejects([]{NativePcgHistory no_copy({});});
    env("0");require(!gipc::native_history_study_enabled());env("1");require(gipc::native_history_study_enabled());
    env("true");rejects([]{gipc::native_history_study_enabled();});env("0");
    std::cout<<"Native history slot/scalar/exception/budget/parser contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
