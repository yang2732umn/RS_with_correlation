#include "consider_covariance.h"
VectorXd solve21_acc_ama_ptl(int judge,double gamma, int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi)
{
    int iter=0;
    int p=y.size();
    int nc=0;
    for (int i=0; i<p; ++i) {
        nc+=cstrfdi[i].size();
    }
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    VectorXd lambda_old=VectorXd::Zero(nc);
    VectorXd lambda=lambda_old;
    VectorXd S=lambda;
    int l,j;
    double objF,objD;
    VectorXd delta=VectorXd::Zero(p);
    VectorXd u=VectorXd::Zero(p);
    VectorXd up;
    vector<Vector2i> fdi,sdi;
    while (iter<1) {
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
    }
    while(iter<maxIter){
        S=lambda + ((double)(iter-1))/((double)(iter+2))*(lambda- lambda_old);
        delta=VectorXd::Zero(p);
        if (judge>0) {
            for (int i=0; i<p; ++i) {
                fdi=cstrfdi[i];
                for (int j=0; j<fdi.size(); ++j) {
                    delta[i]+=S[fdi[j][1]];
                }
                sdi=cstrsdi[i];
                for (int j=0; j<sdi.size(); ++j) {
                    delta[i]-=S[sdi[j][1]];
                }
            }
        }
        up=u;
        u=y+delta;
        lambda_old=lambda;
        l=0;
        if (judge>0) {
            for (int i=0; i<p; ++i) {
                fdi=cstrfdi[i];
                for (int j=0; j<fdi.size(); ++j) {
                    lambda[l]=S[l]-mu*(u[i]-u[fdi[j][0]]);
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
        cout<<"solve21 acc ama not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
    }
    return u;
}

