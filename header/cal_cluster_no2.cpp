#include "consider_covariance.h"
int cal_cluster_no2(const MatrixXd &X){
    int ncluster=1;
    int flag;
    int size2=X.cols();
    int n=(1+sqrt(1+8*size2))/2;
    VectorXd v(X.rows());
    for(int i=1;i<n;++i){
        flag=1;
        VectorXi sdi=constructsdi(i+1,n);
        if(i==2) cout<<"sdi="<<(sdi).transpose()<<endl;
        for(int j=0;j<sdi.size(); ++j){
            v=X.col(sdi[j]);
            if(v.norm()==0){
                flag=0;
                break;
            }
        }
        if(flag==1){
            ncluster=ncluster+1;
        }
    }
    return ncluster;
}
