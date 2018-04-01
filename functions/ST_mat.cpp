#include "consider_covariance.h"
MatrixXd ST_mat(const MatrixXd &a,double b){//sign(a)(|a|-b)_+
    MatrixXd d(a.rows(),a.cols());
    double c;
    for(int i=0;i<a.rows();++i){
        for (int j=0; j<a.cols(); ++j) {
            if(abs(a(i,j))<b) c=0;
            else{
                c=abs(a(i,j))-b;
                c=c*sign(a(i,j));
            }
            d(i,j)=c;
        }
    }
    return d;
}
