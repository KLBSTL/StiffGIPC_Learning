#include "weighted_check_schedule.h"
#include <linear_system/solver/native_balanced_math.h>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

using gipc::research::WeightedCheckSchedule;
using gipc::research::RecursiveNorm;
static void require(bool value,const char* label){if(!value)throw std::runtime_error(label);}
template<class F>static void rejects(F f,const char* label) {
    try{f();}catch(const std::invalid_argument&){return;}
    throw std::runtime_error(label);
}
int main() {
    try {
        static_assert(std::is_trivially_copyable_v<WeightedCheckSchedule>);
        static_assert(std::is_same_v<decltype(std::declval<WeightedCheckSchedule&>().observe(1,8,.01,.02)),void>);
        int cases=0;
        WeightedCheckSchedule legacy;
        const double edge=.012*.012*3.;
        for(int i=1;i<=40;++i) {
            const double rr=i%3==0?std::nextafter(edge,0.):i%3==1?std::nextafter(edge,1.):.5;
            require(legacy.needs_recursive_norm(i,40),"Normal phase needs raw norm");
            const bool expected=gipc::native_quality_check_due(i,legacy.last_check(),rr,3.,40);
            require(legacy.due(i,40,RecursiveNorm{rr,3.})==expected,"Legacy boundary mismatch");
            if(expected)legacy.observe(i,40,.02,.02);
        }
        ++cases;
        WeightedCheckSchedule s;s.observe(27,2000,.005,.02);
        require(s.cooling_down()&&!s.needs_recursive_norm(28,2000),"Enter conditional cooldown");
        require(!s.due(28,2000,std::nullopt)&&s.due(29,2000,std::nullopt),"Two-step cadence");++cases;
        require(s.due(28,28,std::nullopt),"Actual final budget must check");++cases;
        rejects([&]{s.due(28,2000,RecursiveNorm{0,1});},"Do not count hidden norm work");++cases;
        s.observe(29,2000,.02,.02);
        require(!s.cooling_down()&&s.needs_recursive_norm(30,2000),"L2 regression exits cooldown");
        require(s.due(30,2000,RecursiveNorm{0,1}),"Resume legacy screen");++cases;
        rejects([&]{s.observe(29,2000,.005,.02);},"Ordinary duplicate must fail");
        s.observe(29,2000,.005,.02,true);require(s.cooling_down(),"Explicit zero-rho repeat");++cases;
        const auto before=s;
        rejects([&]{s.observe(30,2000,std::numeric_limits<double>::quiet_NaN(),.02);},"NaN quality");
        rejects([&]{s.observe(30,2000,.005,std::numeric_limits<double>::infinity());},"Infinite quality");
        rejects([&]{s.observe(30,2000,-1,.02);},"Negative quality");
        require(s.last_check()==before.last_check()&&s.cooling_down()==before.cooling_down(),"Invalid observation committed state");++cases;
        WeightedCheckSchedule plain;
        rejects([&]{plain.due(1,20,std::nullopt);},"Missing norm");
        rejects([&]{plain.due(1,20,RecursiveNorm{-1,1});},"Negative rr");
        rejects([&]{plain.due(1,20,RecursiveNorm{0,0});},"Zero bb");
        rejects([&]{plain.due(1,20,RecursiveNorm{std::numeric_limits<double>::quiet_NaN(),1});},"NaN rr");++cases;
        rejects([&]{s.due(0,2000,std::nullopt);},"Zero index");
        rejects([&]{s.due(28,2000,std::nullopt);},"Backward index");
        rejects([&]{s.due(30,29,std::nullopt);},"Past budget");++cases;
        WeightedCheckSchedule toy;int checks=0,norms=0;
        for(int i=57;i<=79;++i) {
            std::optional<RecursiveNorm> norm;
            if(toy.needs_recursive_norm(i,2000)){norm=RecursiveNorm{.005*.005,1};++norms;}
            if(toy.due(i,2000,norm)){toy.observe(i,2000,.005,i==79?.009:.02);++checks;}
        }
        require(checks==12&&norms==1&&toy.last_check()==79&&!toy.cooling_down(),"Aligned synthetic mixed path");++cases;
        s=WeightedCheckSchedule{};require(s.last_check()==0&&!s.cooling_down(),"Reset between solves");++cases;
        std::cout<<"weighted_check_schedule: "<<cases<<" contract groups passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
