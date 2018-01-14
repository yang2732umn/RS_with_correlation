#include "consider_covariance.h"
double matrix_Jtaunorm(double tau,const MatrixXd& X){//only uses upper triangular
    double norm=0;
    int i,j;
    for (i=0; i<X.rows(); ++i) {
        for (j=i; j<X.cols(); ++j) {
            norm+=J_tau(X(i,j),tau);
        }
    }
    return norm;
} 
