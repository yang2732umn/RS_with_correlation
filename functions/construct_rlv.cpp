#include "consider_covariance.h"
vector<Omega_1rlv> construct_rlv(int p,const vector<rated_user> &user){//rlv is in order, only contain those item pairs rated simultaneously by at least one user
    int n=user.size();
    vector<Omega_1rlv> rlv(p*(p-1)/2);
    Omega_1rlv rlv1;
    int good=0;
    int l=0;
    
    VectorXi firstno(p*(p-1)/2);
    VectorXi secno(p*(p-1)/2);
    for (int i=0; i<p-1; ++i) {
        for (int j=i+1; j<p; ++j) {
            firstno[l]=i;
            secno[l]=j;
            ++l;
        }
    }
# pragma omp parallel private(rlv1)
    {
# pragma omp for
        for (good=0;good<secno.size();++good) {
            rlv1.item1=firstno[good];
            rlv1.item2=secno[good];
            rlv1.userno.resize(0);
            rlv1.firstin.resize(0);
            rlv1.secin.resize(0);
            int count=0;
            for(int k=0;k<n;++k){
                int findi=existindex(user[k].item, rlv1.item1);
                int findj=existindex(user[k].item, rlv1.item2);
                if((findi!=-1)&&(findj!=-1)){
                    ++count;
                    rlv1.userno.push_back(k);
                    rlv1.firstin.push_back(findi);
                    rlv1.secin.push_back(findj);
                }
            }
            if(count==0){
                rlv1.userno.push_back(-1);
                rlv1.firstin.push_back(-1);
                rlv1.secin.push_back(-1);
            }
            rlv[good]=rlv1;
        }
    }
    vector<Omega_1rlv> rlv2;
    for (good=0; good<rlv.size(); ++good) {
        if (rlv[good].userno[0]!=-1) {
            rlv2.push_back(rlv[good]);
        }
    }
    return rlv2;
}
