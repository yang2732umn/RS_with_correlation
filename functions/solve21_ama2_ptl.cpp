#include "consider_covariance.h"
VectorXd solve21_ama2_ptl(int judge,double gamma,int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi){
    int iter=0;
    int p=y.size();
    int nc=0;
    for (int i=0; i<p; ++i) {
        nc+=cstrfdi[i].size();
    }
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    VectorXd lambda_old=VectorXd::Zero(nc);
    VectorXd lambda=lambda_old;
    int l,j;
    double objF,objD;
    VectorXd delta=VectorXd::Zero(p);
    VectorXd u=VectorXd::Zero(p);
    VectorXd up;
    vector<Vector2i> fdi,sdi;
    while (iter<maxIter) {
        lambda_old=lambda;
        delta=VectorXd::Zero(p);
        if(judge>0){
            for (int i=0; i<p; ++i) {
                fdi=cstrfdi[i];
                for (int j=0; j<fdi.size(); ++j) {
                    delta[i]+=lambda[fdi[j][1]];
                }
                sdi=cstrsdi[i];
                for (int j=0; j<sdi.size(); ++j) {
                    delta[i]-=lambda[sdi[j][1]];//lambda the same order as constraints
                }
            }
        }
        up=u;
        u=y+delta;
        l=0;
        if(judge>0){
            for (int i=0; i<p; ++i) {
                fdi=cstrfdi[i];
                for (int j=0; j<fdi.size(); ++j) {
                    lambda[l]=lambda[l]-mu*(u[i]-u[fdi[j][0]]);//seems correct, constraints always 1 as first user, then 2 as first user,...
                    lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
                    ++l;
                }
            }
        }
        ++iter;
        maxdiff=(up-u).lpNorm<Infinity>();//(objF-objD)/(1+0.5*(objF+objD));
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 ama2_ptl not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
    }
    return u;
}
