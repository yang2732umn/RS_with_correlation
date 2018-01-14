#include "consider_covariance.h"
VectorXd solve21_acc_ama(const VectorXd &y,double gamma, int maxIter, double Tol)
{
    int iter=0;
    int p=y.size();
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    VectorXd lambda_old=VectorXd::Zero(p*(p-1)/2);
    VectorXd lambda=lambda_old;
    VectorXd S=lambda;
    int l,j;
    VectorXi fdi,sdi;
    double objF,objD;
    VectorXd delta=VectorXd::Zero(p);
    VectorXd u=VectorXd::Zero(p);
    VectorXd up;
    while (iter<1) {
        lambda_old=lambda;
        delta=VectorXd::Zero(p);
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
        u=y+delta;
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                lambda[l]=lambda[l]-mu*(u[t1]-u[t2]);
                lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
            }
        }
        ++iter;
    }
    while(iter<maxIter){
        S=lambda + ((double)(iter-1))/((double)(iter+2))*(lambda- lambda_old);
        delta=VectorXd::Zero(p);
        for (int i=0; i<p; ++i) {
            if(i<p-1){
                fdi=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
                for (int j=0; j<fdi.size(); ++j) {
                    delta[i]+=S[fdi[j]];
                }
            }
            if(i>0){
                sdi=constructsdi(i+1, p);
                for (int j=0; j<sdi.size(); ++j) {
                    delta[i]-=S[sdi[j]];
                }
            }
        }
        up=u;
        u=y+delta;
        lambda_old=lambda;
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                lambda[l]=S[l]-mu*(u[t1]-u[t2]);
                lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
            }
        }
        ++iter;
        maxdiff=(up-u).lpNorm<Infinity>();//(objF-objD)/(1+0.5*(objF+objD));
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 acc ama not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
    }
    return u;
}

