#include "consider_covariance.h"
double cal_vec_pairdiff(const VectorXd& X){//|x1-x2|+|x1-x3|+|x2-x3|
    double diff=0;
    int size=X.size();
    for (int i=0; i<size-1; ++i) {
        for (int j=i+1; j<size; ++j) {
            diff=diff+abs(X(i)-X(j));
        }
    }
    return diff;
}