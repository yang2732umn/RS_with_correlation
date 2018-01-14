#include "consider_covariance.h"
double ST(double a,double b){//sign(a)(|a|-b)_+
    double c;
    if(abs(a)<b) c=0;
    else{
        c=abs(a)-b;
        c=c*sign(a);
    }
    return c;
}

