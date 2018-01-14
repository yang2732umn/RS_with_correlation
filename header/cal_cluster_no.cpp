#include "consider_covariance.h"
int cal_cluster_no(const MatrixXd &X){
    //X is a matrix with n columns
    int ncluster=1;
    int flag;
    int n=X.cols();
    //cout<<"n="<<n<<endl;
    VectorXd v(X.rows());
    vector<VectorXd> center;
    center.reserve(2);
    center.push_back(X.col(0));
    
    for(int i=1;i<n;++i){
        flag=1;
        for(int j=0;j<i; ++j){
            //cout<<"flag="<<flag<<endl;
            v=X.col(i)-X.col(j);
            if(v.lpNorm<1>()/v.size()<1e-3){
                flag=0;
                break;
            }
            //if(v.norm()>0) cout<<"col "<<sdi[j]<<" !=0"<<endl;//
        }
        if(flag==1){
            //cout<<"i="<<i<<endl;
            center.push_back(X.col(i));
            ncluster=ncluster+1;
        }
    }
    //cout<<"centers are"<<endl;
    for (int i=0; i<center.size(); i++) {
        //cout<<center[i].transpose()<<endl;
    }
    
    return ncluster;
}

