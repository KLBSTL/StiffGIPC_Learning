#include <cmath>
#include <iostream>
#if __has_include("toi_port_policy.cuh")
#include "toi_port_policy.cuh"
#else
// RED baseline: v31 updates weight only and cannot preserve release age.
namespace toi_port {
void update_release(double& lambda,double& gamma,int&,double value,double mu,bool active)
{if(active){lambda-=mu*value;gamma=1;}else{lambda=0;gamma*=.9;}}
bool retain_contact(int){return true;}
bool selected_toi(double t,double maximum){return t==maximum;}
}
#endif
int main()
{
    int failures=0;
    auto check=[&](bool ok,const char* name){if(!ok){std::cerr<<"FAIL "<<name<<'\n';++failures;}};
    double lambda=3,gamma=1;int age=0;
    for(int i=0;i<26;++i)toi_port::update_release(lambda,gamma,age,.2,4,false);
    check(age==26&&!toi_port::retain_contact(age),"26 releases remove contact independently of weight reset");
    gamma=1;
    check(!toi_port::retain_contact(age),"reset weight cannot resurrect expired contact");
    toi_port::update_release(lambda,gamma,age,-.5,4,true);
    check(age==0&&lambda==2&&gamma==1&&toi_port::retain_contact(age),"reactivation resets age and updates multiplier");
    check(toi_port::selected_toi(.45,.45-.5e-6),"near-equal vertex TOI tie retained");
    check(!toi_port::selected_toi(1.,1.),"stationary TOI excluded");
    check(!toi_port::selected_toi(NAN,1.),"invalid TOI excluded");
    if(!failures)std::cout<<"PASS 6 release/filter lifecycle checks\n";
    return failures?1:0;
}
