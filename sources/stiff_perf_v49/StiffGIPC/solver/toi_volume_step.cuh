#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace toi_volume
{
__device__ inline double3 subtract(double3 a,double3 b)
{return make_double3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ inline double triple(double3 a,double3 b,double3 c)
{return a.x*(b.y*c.z-b.z*c.y)+a.y*(b.z*c.x-b.x*c.z)+a.z*(b.x*c.y-b.y*c.x);}
__device__ inline double evaluate(double a,double b,double c,double d,double t)
{return ((a*t+b)*t+c)*t+d;}

// Relative determinant coefficients avoid a world-unit cutoff for polynomial
// degree. Partition at derivative roots, then bracket the FIRST threshold
// crossing; positive endpoints alone do not establish a positive whole path.
__device__ inline double first_threshold(double a,double b,double c,double d)
{
    // A double root may evaluate slightly above zero after coefficient
    // rounding. Shift the threshold by its arithmetic uncertainty so that
    // tangential volume collapse is conservatively bracketed too.
    double uncertainty=4*2.2204460492503131e-16*(fabs(a)+fabs(b)+fabs(c)+fabs(d));
    d-=fmin(.5*d,uncertainty);
    double cuts[4]={0,1,0,0};int count=2;
    double scale=fmax(fabs(3*a),fmax(fabs(2*b),fabs(c)));
    if(scale>0)
    {
        double A=3*a/scale,B=2*b/scale,C=c/scale;
        if(fabs(A)>1e-14)
        {
            double discriminant=B*B-4*A*C;
            if(discriminant>=0)
            {
                double q=-.5*(B+copysign(sqrt(discriminant),B));
                double roots[2]={q/A,q!=0?C/q:-B/(2*A)};
                for(double root:roots)if(root>0&&root<1)cuts[count++]=root;
            }
        }
        else if(fabs(B)>1e-14)
        {double root=-C/B;if(root>0&&root<1)cuts[count++]=root;}
    }
    for(int i=1;i<count;++i)for(int j=i;j>0&&cuts[j]<cuts[j-1];--j)
    {double v=cuts[j];cuts[j]=cuts[j-1];cuts[j-1]=v;}
    for(int i=1;i<count;++i)
    {
        double high=cuts[i],value=evaluate(a,b,c,d,high);
        if(value<=0)
        {
            double low=cuts[i-1];
            for(int k=0;k<80;++k)
            {
                double middle=(low+high)*.5;
                if(evaluate(a,b,c,d,middle)>0)low=middle;else high=middle;
            }
            return fmax(0.,low*(1-1e-10));
        }
    }
    return 1.;
}

__device__ inline double bound(const double3* x,const double3* displacement,
                               const double3* rest,double retain,int& invalid)
{
    double3 a=subtract(x[1],x[0]),b=subtract(x[2],x[0]),c=subtract(x[3],x[0]);
    double3 da=subtract(displacement[1],displacement[0]);
    double3 db=subtract(displacement[2],displacement[0]);
    double3 dc=subtract(displacement[3],displacement[0]);
    double determinant=triple(a,b,c);
    double rest_determinant=triple(subtract(rest[1],rest[0]),subtract(rest[2],rest[0]),subtract(rest[3],rest[0]));
    if(!isfinite(determinant)||!isfinite(rest_determinant)||determinant==0||rest_determinant==0
        ||(determinant>0)!=(rest_determinant>0))
    {invalid=1;return 0;}
    double p1=(triple(da,b,c)+triple(a,db,c)+triple(a,b,dc))/determinant;
    double p2=(triple(da,db,c)+triple(da,b,dc)+triple(a,db,dc))/determinant;
    double p3=triple(da,db,dc)/determinant;
    if(!isfinite(p1)||!isfinite(p2)||!isfinite(p3)){invalid=2;return 0;}
    return first_threshold(p3,p2,p1,1-retain);
}
}
