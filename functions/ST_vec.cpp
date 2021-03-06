#include "consider_covariance.h"
VectorXd ST_vec(const VectorXd &a,double b){//sign(a)(|a|-b)_+
    VectorXd d(a.size());
    double c;
    for(int i=0;i<a.size();++i){
        if(abs(a[i])<b) c=0;
        else{
            c=abs(a[i])-b;
            c=c*sign(a[i]);
        }
        d[i]=c;
    }
    return d;
}
