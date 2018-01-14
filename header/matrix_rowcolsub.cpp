#include "consider_covariance.h"
MatrixXd matrix_rowcolsub(const MatrixXd &a, const vector<int> &b){
    //get submatric of a with rows and cols in b
    MatrixXd c=matrix_rowsub(a,b);
    c=matrix_colsub(c,b);
    return c;
}