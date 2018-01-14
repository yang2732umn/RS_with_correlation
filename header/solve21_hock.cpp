#include "consider_covariance.h"
VectorXd solve21_hock(VectorXd &A,double lambda){
    VectorXd trueA=A;
    int K=A.size();
    MatrixXi fusions=MatrixXi::Zero(K,K);
    VectorXd betas(K);
    VectorXd newc=VectorXd::Zero(K);
    
    
    for(int i=0;i<K;++i){
        for (int j=0;(j<K); ++j) {
            if (j==i) continue;
            newc[i]+=(A[j]-A[i]<-1e-4)-(A[j]-A[i]>1e-4);//
        }
    }
    VectorXi ordermats(K);
    VectorXi new_ordermats(K);
    //main iteration
    for(int iter=1;iter<=K-1;++iter){
        
        for(int k=0;k<K;++k){
            ordermats[k]=1;
            for(int l=0;(l<K);++l){
                if (l==k) continue;
                ordermats[k]+=(A[k]-A[l]>1e-4);
                if (l<k) {
                    ordermats[k]+=(abs(A[k]-A[l])<1e-4);
                }
            }
        }
        betas=A-lambda*newc;
        for(int k=0;k<K;++k){
            new_ordermats[k]=1;
            for(int l=0;(l<K);++l){
                if (l==k) continue;
                new_ordermats[k]+=(betas[k]-betas[l]>1e-4);
                if (l<k) {
                    new_ordermats[k]+=(abs(betas[k]-betas[l])<1e-4);
                }
            }
        }
        for (int k=0; k<K; ++k) {
            for(int kp=k+1;kp<K;++kp){
                fusions(k,kp)=fusions(k,kp)||((ordermats[k]-1==ordermats[kp])&&(new_ordermats[k]<new_ordermats[kp]))||
				((ordermats[k]+1==ordermats[kp])&&(new_ordermats[k]>new_ordermats[kp]))||
				(abs(A[k]-A[kp])<1e-4);
                //fusions(k,kp) = (fusions(k,kp)>0);
                fusions(kp,k)=fusions(k,kp);
            }
        }
        for (int k=0; k<K; ++k) {
            for(int kp=k+1;kp<K;++kp){
                for (int o=0; (o<K); ++o) {
                    if(o==k||o==kp) continue;
                    fusions(k,kp) = fusions(k,kp)||(fusions(k,o) && fusions(kp,o));
                    if(fusions(k,kp)) break;
                }
                fusions(kp,k)=fusions(k,kp);
            }
        }
        vector<int> temp(K);
        for (int k=0; k<K; ++k) {
            temp[k]=k;
        }
        for (int k=0; k<temp.size(); ++k) {
            int l=temp[k];
            vector<int> classl;
            double fusemean = trueA[l];
            int denom = 1;
            for(int o=0;(o<temp.size());++o){
                if(temp[o]==l) continue;
                if (fusions(l,temp[o])) {
                    fusemean +=trueA(temp[o]);
                    denom = denom+1;
                    classl.push_back(o);
                }
                
            }
            A[l] = fusemean/denom;
            sort(classl.begin(),classl.end(),greater<int>());
            for (int o=0; o<classl.size(); ++o) {
                A[temp[classl[o]]]=A[l];
                temp.erase(temp.begin()+classl[o]);
            }
        }
        
        /*for (int k=0; k<K; ++k) {
         double fusemean = trueA[k];
         int denom = 1;
         for(int o=0;(o<K);++o){
         if(o==k) continue;
         fusemean = fusemean+fusions(k,o)*trueA(o);
         denom = denom+fusions(k,o);
         }
         A[k] = fusemean/denom;
         }*/
        newc=VectorXd::Zero(K);
        for(int i=0;i<K;++i){
            for (int j=0; (j<K); ++j) {
                if(j==i) continue;
                newc[i]+=(A[j]-A[i]<-1e-4)-(A[j]-A[i]>1e-4);
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
     cout<<"obj hock="<<obj<<endl;*/
    return betas;
}
