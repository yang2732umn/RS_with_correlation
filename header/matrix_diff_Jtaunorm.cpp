#include "consider_covariance.h"
double matrix_diff_Jtaunorm(double tau,const MatrixXd& X,const MatrixXd& Y){//only calculate 11,12,...,1n,22,23,...2n.....
    double norm=0;
    int i,j;
    for (i=0; i<X.rows(); ++i) {
        for (j=i; j<X.cols(); ++j) {
            norm+=J_tau(X(i,j)-Y(i,j),tau) ;
        }
    }
    return norm;
} 
