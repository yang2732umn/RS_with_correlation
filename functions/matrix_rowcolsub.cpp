#include "consider_covariance.h"
MatrixXd matrix_rowcolsub(const MatrixXd &a, const vector<int> &b){
    MatrixXd c=matrix_rowsub(a,b);
    c=matrix_colsub(c,b);
    return c;
}
