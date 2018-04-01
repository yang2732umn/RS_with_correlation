#include "consider_covariance.h"
ArrayXd Arraysign(const ArrayXd &x){
    int size=x.size();
    ArrayXd sign(size);
    int i;
    for (i=0; i<size; ++i) {
        if(x[i]>0) sign[i]=1;
        if(x[i]==0) sign[i]=0;
        if(x[i]<0) sign[i]=-1;
    }
    return sign;
}
