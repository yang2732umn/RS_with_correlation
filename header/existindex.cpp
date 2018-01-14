
#include "consider_covariance.h"
int existindex(const vector<int> & a, int b){
    int i,j;
    for (i=0; i<a.size(); ++i) {
        if(a[i]==b){
            j=i;
            break;
        }
    }
    if(i==a.size()) j=-1;
    return j;
}
