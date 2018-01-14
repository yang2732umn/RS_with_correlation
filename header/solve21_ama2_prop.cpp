#include "consider_covariance.h"
VectorXd solve21_ama2_prop(const VectorXd &y,int m,double gamma, int maxIter, double Tol)
// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
//this is for propagation version v4_3
//m is the number of u_i's whose y_i is 0.
//y is real y + last idle value(may not be 0)
//ama2 is similar to acc_ama, but just without acceleration
//all connected
{
    int p=y.size();
    VectorXd u=VectorXd::Zero(p);
    tri trip;
    
    int not0=(y.array()!= 0.0).any();
    if (not0==0) {
        return u;
        cout<<"Early quit"<<endl;
    }
    
    int iter=0;
    
    double maxdiff,mu=1.0/(p+m-1);//mu numerator must be 1.0 rather than 1, otherwise mu=0
    //cout<<"mu="<<mu<<endl;
    VectorXd lambda_old=VectorXd::Zero(p*(p-1)/2);
    VectorXd lambda=lambda_old;
    int l,i,j;
    VectorXi fdi,sdi;
    //double objF,objD;
    VectorXd delta=VectorXd::Zero(p);
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
    
    trip.first.resize(p*(p-1)/2);trip.second.resize(p*(p-1)/2);
    l=0;
    for (int i=0; i<p-1; ++i) {
        for (int j=i+1; j<p; ++j) {
            trip.first(l)=i;trip.second(l)=j;
            ++l;
        }
    }
    
    while (iter<maxIter) {
        lambda_old=lambda;
        delta=VectorXd::Zero(p);
        for (int i=0; i<p; ++i){
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
        delta[p-1]=delta[p-1]/m;
        up=u;
        u=y+delta;
        
        for (l=0; l<(p-1)*p/2; ++l){
            int t1=trip.first(l);
            int t2=trip.second(l);
            lambda[l]=lambda[l]-mu*(u[t1]-u[t2]);
            if (t2!=p-1) lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
            else lambda[l]=fmin(fmax(lambda[l],-m*gamma),m*gamma);
        }
        
        /*l=0;
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                //l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                lambda[l]=lambda[l]-mu*(u[t1]-u[t2]);
                if (t2!=p-1) lambda[l]=fmin(fmax(lambda[l],-gamma),gamma);
                else lambda[l]=fmin(fmax(lambda[l],-m*gamma),m*gamma);
                l=l+1;
            }
        }*/
        
        ++iter;
        maxdiff=(up-u).lpNorm<Infinity>();
        //cout<<"objF="<<objF<<", objD="<<objD<<endl;
        //cout<<"maxdiff of acc ama is "<<maxdiff<<endl;
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 ama2 not converged, maxdiff is "<<maxdiff<<", problem size is "<<p<<endl;
        //cout<<"y="<<y.transpose()<<endl;
    }
    //if(maxdiff<=Tol) cout<<"solve21 took "<<iter<<" iterations to converge."<<endl;
    //cout<<"iter is "<<iter<<endl;
    //cout<<"solve21 acc ama final obj is "<<objF<<endl;
    //cout<<"final lambda="<<lambda.transpose()<<endl;
    return u;
}


