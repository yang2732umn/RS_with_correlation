#include "consider_covariance.h"
//vector square norm
double vec_square_norm(const vector<double> & x){
    int n=x.size();
    double sum=0;
    for (int i=0; i<n; ++i) {
        sum+=x[i]*x[i];
    }
    return sum;
}