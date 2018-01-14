#include "consider_covariance.h"
MatrixXd matrix_rowsub(const MatrixXd & a, const vector<int> &b){
    //get the submatrix of a, which is the columns with indices in b
    int newsize=b.size();
    MatrixXd c(newsize,a.cols());
    for (size_t j=0; j<newsize; ++j) {
        c.row(j)=a.row(b[j]);
    }
    return c;
}