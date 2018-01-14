#include "consider_covariance.h"
VectorXd solve21_ama_ptl(const VectorXd &y,const MatrixXi &pair,double lambda, int maxIter, double Tol){// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
    //partially connected
    int iter=0;
    int p=y.size();
    double maxdiff;
    double mu=1.0/p;//mu in ADMM is set 0.1 here
    int nc=pair.rows();
    
    VectorXd b=VectorXd::Zero(nc);
    VectorXd v=b;
    VectorXd beta=y;//start algorithm from y
    VectorXd temp;
    double maxbeta=0,maxb=0,maxv=0;
    int l,j;
    double obj;
    vector<vector<Vector2i>> fdi(p);vector<vector<Vector2i>> sdi(p);
    for (int i=0; i<nc; ++i) {
        Vector2i a;
        a[0]=pair(i,1);a[1]=i;
        fdi[pair(i,0)].push_back(a);
        a[0]=pair(i,0);
        sdi[pair(i,1)].push_back(a);
    }
    /*for (int i=0; i<p; ++i) {
     if (fdi[i].size()>0) {
     cout<<"fdi["<<i<<"] is "<<endl;
     cout<<(fdi[i][0]).transpose()<<endl;
     }
     if (sdi[i].size()>0) {
     cout<<"sdi["<<i<<"] is "<<endl;
     cout<<(sdi[i][0]).transpose()<<endl;
     }
     }*/
    
    while(iter<maxIter){
        maxdiff=0;
        temp=VectorXd::Zero(p);
        for (int i=0; i<p; ++i) {
            for (j=0; j<fdi[i].size(); ++j) {
                temp[i]+=v[fdi[i][j][1]];
            }
            for (j=0; j<sdi[i].size(); ++j) {
                temp[i]-=v[sdi[i][j][1]];
            }
        }
        temp=y-temp;
        maxbeta=(temp-beta).lpNorm<Infinity>();
        maxdiff=max(maxdiff,maxbeta);
        beta=temp;
        
        //cout<<"beta is "<<beta.transpose()<<endl;
        temp.resize(nc);
        for(int t1=0;t1<nc;++t1){
            //cout<<"beta[pair(t1,0)] is "<<beta[pair(t1,0)]<<endl;
            //cout<<"beta[pair(t1,1)] is "<<beta[pair(t1,1)]<<endl;
            //cout<<"v[t1] is "<<v[t1]<<endl;
            //cout<<"mu is "<<mu<<endl;
            temp(t1)=beta[pair(t1,0)]-beta[pair(t1,1)]+v[t1]/mu;
            //cout<<"temp[t1] is "<<temp[t1]<<endl;
            
        }
        //cout<<"temp is "<<temp.transpose()<<endl;
        temp=ST_vec(temp,lambda/mu);
        maxb=(temp-b).lpNorm<Infinity>();
        maxdiff=max(maxdiff,maxb);
        b=temp;
        for(int t1=0;t1<nc;++t1){
            temp(t1)=beta[pair(t1,0)]-beta[pair(t1,1)]-b[t1];
        }
        temp=mu*temp;
        maxv=temp.lpNorm<Infinity>();
        maxdiff=max(maxdiff,maxv);
        v=v+temp;
        ++iter;
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 ama not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
        cout<<"y is "<<y.transpose()<<endl;
    }
    //if(maxdiff<=Tol) cout<<"solve21 took "<<iter<<" iterations to converge."<<endl;
    if(isnan(beta)) cout<<"beta is "<<(beta).transpose()<<endl;
    obj=0.5*(y-beta).squaredNorm();
    for(int t1=0;t1<nc;++t1){
        obj=obj+lambda*abs(beta[pair(t1,0)]-beta[pair(t1,1)]);
    }
    //cout<<"solve21 ama final obj is "<<obj<<endl;
    return beta;
}
