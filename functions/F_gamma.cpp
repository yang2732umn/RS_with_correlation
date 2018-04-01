#include "consider_covariance.h"
double F_gamma(const VectorXd &y,const VectorXd &beta,double gamma){
    int p=y.size();
    double obj=0.5*(y-beta).squaredNorm();
    for(int t1=0;t1<p-1;++t1){
        for(int t2=t1+1;t2<p;++t2){
            obj=obj+gamma*abs(beta[t1]-beta[t2]);
        }
    }
    return obj;
}

