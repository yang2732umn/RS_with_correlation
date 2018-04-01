#include "consider_covariance.h"
double vec_linfnorm(const vector<double> & x){//l-infinity norm: max of abs
    int n=x.size();
    double norm=0;
    for (int i=0; i<n; ++i) {
        if (norm<abs(x[i])) norm=abs(x[i]);
    }
    return norm;
}
