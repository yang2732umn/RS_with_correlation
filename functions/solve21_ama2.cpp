#include "consider_covariance.h"
VectorXd solve21_ama2(const VectorXd &y,double gamma, int maxIter, double Tol)// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
{
    int iter=0;
    int p=y.size();
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    VectorXd lambda_old=VectorXd::Zero(p*(p-1)/2);
    VectorXd lambda=lambda_old;
    int l,j;
    VectorXi fdi,sdi;
    double objF,objD;
    VectorXd delta=VectorXd::Zero(p);
    VectorXd u=VectorXd::Zero(p);
    VectorXd up;
    vector<VectorXi> fdis(p-1),sdis(p-1);
    for (int i=0; i<p; ++i) {
        if(i<p-1){
            fdis[i]=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
        }
        if(i>0){
            sdis[i-1]=constructsdi(i+1, p);
        }
    }
    while (iter<maxIter) {
        lambda_old=lambda;
        delta=VectorXd::Zero(p);
        for (int i=0; i<p; ++i) {
            if(i<p-1){
                fdi=fdis[i];
                for (int j=0; j<fdi.size(); ++j) {
                    delta[i]+=lambda[fdi[j]];
                }
            }
            if(i>0){
                sdi=sdis[i-1];
                for (int j=0; j<sdi.size(); ++j) {
                    delta[i]-=lambda[sdi[j]];
                }
            }
        }
        up=u;
        u=y+delta;
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                lambda[l]=lambda[l]-mu*(u[t1]-u[t2]);
                lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
            }
        }
        ++iter;
        maxdiff=(up-u).lpNorm<Infinity>();
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 ama2 not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
        cout<<"y="<<y.transpose()<<endl;
    }
    return u;
}
