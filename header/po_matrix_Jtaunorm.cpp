#include "consider_covariance.h"
double po_matrix_Jtaunorm(double tau,const MatrixXd& X){//only calculate 11,12,...,1n,22,23,...2n.....//X elements are all positive 
    double norm=0;
    int i,j;
    for (i=0; i<X.rows(); ++i) {
        for (j=i; j<X.cols(); ++j) {
            norm+=min(X(i,j),tau);
        }
    }
    return norm;
} 
