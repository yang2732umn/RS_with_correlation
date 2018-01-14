#include "consider_covariance.h"
double F_gamma_vec(const MatrixXd &y,const MatrixXd &beta,double gamma){//vec version
    //y is a matrix with samples in columns
    int n=y.cols();// n is sample size
    double obj=0;
    for (int i=0; i<n; i++) {
        VectorXd temp=y.col(i)-beta.col(i);
        obj+=temp.squaredNorm();
    }
    obj=obj*0.5;
    for(int t1=0;t1<n-1;++t1){
        for(int t2=t1+1;t2<n;++t2){
            VectorXd temp=beta.col(t1)-beta.col(t2);
            obj=obj+gamma*(temp.lpNorm<1>());
        }
    }
    return obj;
}