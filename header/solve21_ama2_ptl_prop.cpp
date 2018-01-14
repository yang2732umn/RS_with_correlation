#include "consider_covariance.h"
VectorXd solve21_ama2_ptl_prop(int m,double gamma, int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi){
    //not use acc, last element is special case
    //use ama to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
    //used only if there is constraint
    //partially connected
    //for each sample yi, it's a scalar
    //Vector2i first coordinate is index of the other, second coordinate is number of constraint
    //also solves the case of gamma=0 correctly
    int p=y.size();
    VectorXd u=VectorXd::Zero(p);
    VectorXd up;
    
    int iter=0;
    
    int nc=0;
    
    for (int i=0; i<p; ++i) {
        nc+=cstrfdi[i].size();
    }
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    //cout<<"mu="<<mu<<endl;
    VectorXd lambda_old=VectorXd::Zero(nc);
    //lambda is in sequence of cstrfdi, 12, 13, 14, 23, 24, 34
    VectorXd lambda=lambda_old;
    int l,j;
    VectorXd delta=VectorXd::Zero(p);
    
    if(cstrsdi[p-1].size()==0){//the last one not involved
        u[p-1]=y[p-1];
        VectorXd y1=y;
        y1.conservativeResize(p-1);
        vector<vector<Vector2i>> cstrfdi1=cstrfdi,cstrsdi1=cstrsdi;
        cstrfdi1.resize(cstrfdi1.size()-1);
        cstrsdi1.resize(cstrsdi1.size()-1);
        VectorXd u1=solve21_ama2_ptl(1, gamma, maxIter,Tol,y1,cstrfdi1,cstrsdi1);
        for (int i=0; i<u1.size(); ++i) {
            u[i]=u1[i];
        }
    }
    else{
        vector<Vector2i> fdi,sdi;
        while (iter<maxIter) {
            lambda_old=lambda;
            delta=VectorXd::Zero(p);
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
            
            delta[p-1]=delta[p-1]/m;
            up=u;
            u=y+delta;
            l=0;
            for (int i=0; i<p; ++i) {
                fdi=cstrfdi[i];
                for (int j=0; j<fdi.size(); ++j) {
                    int secondu=fdi[j][0];
                    lambda[l]=lambda[l]-mu*(u[i]-u[secondu]);//seems correct, constraints always 1 as first user, then 2 as first user,...
                    if(secondu!=p-1) lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
                    else lambda[l]=fmin(fmax(lambda[l],-m*gamma),m*gamma);
                    ++l;
                }
            }
            ++iter;
            maxdiff=(up-u).lpNorm<Infinity>();
            //cout<<"objF="<<objF<<", objD="<<objD<<endl;
            //cout<<"maxdiff of acc ama is "<<maxdiff<<endl;
            if(maxdiff<Tol) break;
        }
        if(maxdiff>Tol){
            cout<<"solve21 acc ama not converged, maxdiff is "<<maxdiff<<endl;
            cout<<"problem size is "<<p<<endl;
            //cout<<"y="<<y.transpose()<<endl;
        }
    }
    
    //if(maxdiff<=Tol) cout<<"solve21 took "<<iter<<" iterations to converge."<<endl;
    //cout<<"iter is "<<iter<<endl;
    //cout<<"solve21 acc ama final obj is "<<objF<<endl;
    //cout<<"final lambda="<<lambda.transpose()<<endl;
    return u;
}

