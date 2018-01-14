#include "consider_covariance.h"
VectorXi constructsdi(int i, int n){
    VectorXi a(i-1);
    for (int j=1; j<=i-1; ++j) {
        a[j-1]=(j-1)*(n-j*0.5)+(i-j)-1;
    }
    return a;
}
