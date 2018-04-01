#include "consider_covariance.h"
double D_gamma(const VectorXd &y,const VectorXd &lambda,double gamma){
    int p=y.size();
    VectorXd delta=VectorXd::Zero(p);
    VectorXi fdi,sdi;
    double obj=0;
    for (int i=0; i<p; ++i) {
        if(i<p-1){
            fdi=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
            for (int j=0; j<fdi.size(); ++j) {
                delta[i]+=lambda[fdi[j]];
            }
        }
        if(i>0){
            sdi=constructsdi(i+1, p);
            for (int j=0; j<sdi.size(); ++j) {
                delta[i]-=lambda[sdi[j]];
            }
        }
    }
    obj=-0.5*delta.squaredNorm();
    for(int t1=0;t1<p-1;++t1){
        for(int t2=t1+1;t2<p;++t2){
            int l=t1*(p-(t1+1)*0.5)+t2-t1-1;
            obj=obj-lambda[l]*(y[t1]-y[t2]);
        }
    }
    return obj;
}

