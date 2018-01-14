#include "consider_covariance.h"
VectorXd Vec_sub(const VectorXd & a, const vector<int> &b){
    //get the subVec of a with indices in b
    int newsize=b.size();
    VectorXd c(newsize);
    for (size_t j=0; j<newsize; ++j) {
        c(j)=a(b[j]);
    }
    return c;
}