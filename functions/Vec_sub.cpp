#include "consider_covariance.h"
VectorXd Vec_sub(const VectorXd & a, const vector<int> &b){
    int newsize=b.size();
    VectorXd c(newsize);
    for (size_t j=0; j<newsize; ++j) {
        c(j)=a(b[j]);
    }
    return c;
}
