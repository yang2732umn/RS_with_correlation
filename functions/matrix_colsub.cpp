#include "consider_covariance.h"
MatrixXd matrix_colsub(const MatrixXd &a, const vector<int> &b){
    int newsize=b.size();
    MatrixXd c(a.rows(),newsize);
    for (size_t j=0; j<newsize; ++j) {
        c.col(j)=a.col(b[j]);
    }
    return c;
}
