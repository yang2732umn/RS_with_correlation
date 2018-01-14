#include "consider_covariance.h"
int cal_cluster_no2(const MatrixXd &X){
    //X is a matrix with n*(n-1)/2 columns
    int ncluster=1;
    int flag;
    int size2=X.cols();
    int n=(1+sqrt(1+8*size2))/2;
    //cout<<"n="<<n<<endl;
    VectorXd v(X.rows());
    for(int i=1;i<n;++i){
        flag=1;
        VectorXi sdi=constructsdi(i+1,n);
        if(i==2) cout<<"sdi="<<(sdi).transpose()<<endl;
        for(int j=0;j<sdi.size(); ++j){
            //cout<<"flag="<<flag<<endl;
            v=X.col(sdi[j]);
            if(v.norm()==0){
                flag=0;
                break;
            }
            //if(v.norm()>0) cout<<"col "<<sdi[j]<<" !=0"<<endl;
        }
        if(flag==1){
            //cout<<"i="<<i<<endl;
            ncluster=ncluster+1;
        }
    }
    return ncluster;
}
