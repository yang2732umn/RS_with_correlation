#include "consider_covariance.h"
MatrixXd absm(const MatrixXd &x){
    //absolute value of a matrix
    int i,j;
    MatrixXd y(x.rows(),x.cols());
    for (i=0; i<x.rows(); ++i) {
        for (j=0; j<x.cols(); j++) {
            if (x(i,j)<0) {
                y(i,j)=-x(i,j);
            }
            else{
                y(i,j)=x(i,j);
            }
        }
    }
    return y;
}