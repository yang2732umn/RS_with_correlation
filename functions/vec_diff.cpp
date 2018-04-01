#include "consider_covariance.h"
vector<double> vec_diff(const vector<double> & x,const vector<double> & y){
    int n=x.size();
    vector<double> z(n);
    for (int i=0; i<n; ++i) {
        z[i]=x[i]-y[i];
    }
    return z;
}
