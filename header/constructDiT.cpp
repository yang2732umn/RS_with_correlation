#include "consider_covariance.h"
MatrixXd constructDiT(int m, int mi,const vector<int> &a){
    MatrixXd DiT=MatrixXd::Zero(m,mi);
    for (int i=0; i<mi; ++i) {
        DiT(a[i],i)=1;
    }
    return DiT;
}
