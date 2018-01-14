#include "consider_covariance.h"

VectorXd solve21(const VectorXd &y,double lambda, int maxIter, double Tol){// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
    int iter=0;
    double maxdiff,mu=0.5;//mu in ADMM is set 0.1 here
    int p=y.size();
    VectorXd b=VectorXd::Zero(p*(p-1)/2);
    VectorXd v=b;
    VectorXd beta=y;//start algorithm from y
    VectorXd temp(p),temp0;
    double maxbeta=0,maxb=0,maxv=0;
    int l,j;
    VectorXi fdi,sdi;
    double obj;
    while(iter<maxIter){
        maxdiff=0;
        temp0=v-mu*b;
        temp=VectorXd::Zero(p);
        for (int i=0; i<p; ++i) {
            if(i<p-1){
                fdi=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
                for (j=0; j<fdi.size(); j++) {
                    temp[i]+=temp0[fdi[j]];
                }
            }
            if(i>0){
                sdi=constructsdi(i+1, p);
                for (j=0; j<sdi.size(); j++) {
                    temp[i]-=temp0[sdi[j]];
                }
            }
        }
        temp=y-temp;
        MatrixXd mtemp2=-MatrixXd::Ones(p,p);
        mtemp2.diagonal()=VectorXd::Ones(p)*(p-1);
        MatrixXd mtemp=MatrixXd::Identity(p,p)+mu*mtemp2;//should always be PD
        //if(!mtemp.ldlt().isPositive()) cout<<"Not invertible!!"<<endl;
        //else cout<<"Invertible"<<endl;
        //EigenSolver<MatrixXd> es(mtemp,false);
        //VectorXd eigenv=es.eigenvalues().real();
        //cout<<"Eigenvalues of mtemp are "<<endl;
        //cout<<eigenv.transpose()<<endl;
        temp=mtemp.inverse()*temp;
        maxbeta=(temp-beta).lpNorm<Infinity>();
        maxdiff=max(maxdiff,maxbeta);
        beta=temp;
        
        temp.resize(p*(p-1)/2);
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                temp(l)=beta[t1]-beta[t2]+v[l]/mu;
            }
        }
        temp=ST_vec(temp,lambda/mu);
        maxb=(temp-b).lpNorm<Infinity>();
        maxdiff=max(maxdiff,maxb);
        b=temp;
        
        
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                temp(l)=beta[t1]-beta[t2]-b[l];
            }
        }
        temp=mu*temp;
        maxv=temp.lpNorm<Infinity>();
        maxdiff=max(maxdiff,maxv);
        v=v+temp;
        ++iter;
        //cout<<"solve21 maxdiff is "<<maxdiff<<endl;
        //cout<<"iter is "<<iter<<endl;
        
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 not converged, maxdiff is "<<maxdiff<<endl;
    }
    //if(maxdiff<=Tol) cout<<"solve21 took "<<iter<<" iterations to converge."<<endl;
    if(isnan(beta)) cout<<"beta is "<<(beta).transpose()<<endl;
    obj=0.5*(y-beta).squaredNorm();
    for(int t1=0;t1<p-1;++t1){
        for(int t2=t1+1;t2<p;++t2){
            obj=obj+lambda*abs(beta[t1]-beta[t2]);
        }
    }
    cout<<"solve21 admm final obj is "<<obj<<endl;
    return beta;
}


