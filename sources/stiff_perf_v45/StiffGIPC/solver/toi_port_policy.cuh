#pragma once
#include <cmath>
#ifdef __CUDACC__
#define TOI_PORT_HD __host__ __device__
#else
#define TOI_PORT_HD
#endif
namespace toi_port
{
TOI_PORT_HD inline void update_release(double& lambda,double& gamma,int& age,
                                       double value,double mu,bool active)
{
    if(active){lambda-=mu*value;age=0;gamma=1;}
    else {lambda=0;if(age<26)++age;gamma=pow(.9,age);}
}
TOI_PORT_HD inline bool retain_contact(int age){return age<=25;}
TOI_PORT_HD inline bool selected_toi(double toi,double maximum_vertex_minimum)
{
    return toi>=0 && toi<1-1e-6 && toi<maximum_vertex_minimum+1e-6;
}
}
#undef TOI_PORT_HD
