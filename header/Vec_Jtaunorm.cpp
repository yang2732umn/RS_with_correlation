#include "consider_covariance.h"
double Vec_Jtaunorm(double tau,const VectorXd& X){
    double norm=0;
    for (int i=0; i<X.size(); ++i) {
        norm+=J_tau(X(i),tau);
    }
    return norm;
}

