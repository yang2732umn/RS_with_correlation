#include "consider_covariance.h"
VectorXd solve21_hock3(VectorXd &A,double lambda){
    //faster than hock when K is large
    VectorXd trueA=A;
    int K=A.size();
    MatrixXi fusions=MatrixXi::Zero(K,K);
    VectorXd betas(K);
    VectorXd newc=VectorXd::Zero(K);
    
    for(int i=0;i<K;++i){
        for (int j=0;(j<K); ++j) {
            if (j==i) continue;
            newc[i]+=(A[j]-A[i]<-1e-4)-(A[j]-A[i]>1e-4);//must use this, if use order, result bad
        }
    }
    vector<pair<double,int>> V;
    VectorXi ordermats(K);
    VectorXi new_ordermats(K);
    //main iteration
    for(int iter=1;iter<=K-1;++iter){
        V.resize(0);
        for(int i=0;i<K;++i){
            pair<double,int>P=make_pair(A[i],i);
            V.push_back(P);
        }
        sort(V.begin(),V.end());
        for (int i=0; i<K; ++i) {
            int orig=V[i].second;
            ordermats[orig]=i;
        }//2n+nlogn<n^2 for large n
        
        betas=A-lambda*newc;
        
        V.resize(0);
        for(int i=0;i<K;++i){
            pair<double,int>P=make_pair(betas[i],i);
            V.push_back(P);
        }
        sort(V.begin(),V.end());
        for (int i=0; i<K; ++i) {
            int orig=V[i].second;
            new_ordermats[orig]=i;
        }
        
        for (int k=0; k<K; ++k) {
            for(int kp=k+1;kp<K;++kp){
                fusions(k,kp)=fusions(k,kp)||((ordermats[k]-1==ordermats[kp])&&(new_ordermats[k]<new_ordermats[kp]))||
				((ordermats[k]+1==ordermats[kp])&&(new_ordermats[k]>new_ordermats[kp]))||
				(abs(A[k]-A[kp])<1e-4);
                //fusions(k,kp) = (fusions(k,kp)>0);
            }
        }
        for (int k=0; k<K; ++k) {
            for(int kp=k+1;kp<K;++kp){
                for (int o=0; (o<K); ++o) {
                    if (o<k) {
                        fusions(k,kp) = fusions(k,kp)||(fusions(o,k) && fusions(o,kp));
                    }
                    if (k<o&&o<kp) {
                        fusions(k,kp) = fusions(k,kp)||(fusions(k,o) && fusions(o,kp));
                    }
                    if (o>kp) {
                        fusions(k,kp) = fusions(k,kp)||(fusions(k,o) && fusions(kp,o));
                    }
                    if(fusions(k,kp)) break;
                }
            }
        }
        vector<int> temp(K);
        for (int k=0; k<K; ++k) {
            temp[k]=k;
        }
        vector<vector<int>> classls;
        for (int k=0; k<temp.size(); ++k) {
            int l=temp[k];
            vector<int> classl;
            vector<int> classlp;
            classlp.push_back(l);
            double fusemean = trueA[l];
            int denom = 1;
            for(int o=k+1;(o<temp.size());++o){
                if (fusions(l,temp[o])) {
                    fusemean = fusemean+trueA(temp[o]);
                    denom = denom+1;
                    classl.push_back(o);
                    classlp.push_back(temp[o]);
                }
            }
            classls.push_back(classlp);
            A[l] = fusemean/denom;
            for (int o=classl.size()-1; o>=0; o--) {
                A[temp[classl[o]]]=A[l];
                temp.erase(temp.begin()+classl[o]);
            }
        }
        newc=VectorXd::Zero(K);
        for(int i=0;i<K;++i){
            for (int j=0; (j<K); ++j) {
                if(j==i) continue;
                newc[i]=newc[i]+(A[j]-A[i]<-1e-4)-(A[j]-A[i]>1e-4);
            }
        }
    }
    betas=A-lambda*newc;
    /*double obj=0.5*(trueA-betas).squaredNorm();
     for (int i=0; i<K; ++i) {
     for (int j=i+1; j<K; ++j) {
     obj=obj+lambda*abs(betas[i]-betas[j]);
     }
     }
     cout<<"obj hock2="<<obj<<endl;*/
    return betas;
}


