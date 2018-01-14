#include "consider_covariance.h"
VectorXd solve21_ama2_ptl(int judge,double gamma,int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi){
    //not use acc
    // use ama to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
    //judge is whether there is constraint or not, if >0, there is, if =0 there isn't
    //partially connected
    //for each sample yi, it's a scalar
    //Vector2i first coordinate is index of the other, second coordinate is number of constraint
    //also solves the case of gamma=0 correctly
    int iter=0;
    int p=y.size();
    int nc=0;
    for (int i=0; i<p; ++i) {
        nc+=cstrfdi[i].size();
    }
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    //cout<<"mu="<<mu<<endl;
    VectorXd lambda_old=VectorXd::Zero(nc);
    //lambda is in sequence of cstrfdi, 12, 13, 14, 23, 24,34
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
        //cout<<"objF="<<objF<<", objD="<<objD<<endl;
        //cout<<"maxdiff of acc ama is "<<maxdiff<<endl;
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 ama2_ptl not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
        //cout<<"y="<<y.transpose()<<endl;
    }
    return u;
}