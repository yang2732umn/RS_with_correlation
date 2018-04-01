#include "consider_covariance.h"
VectorXi unique_veci(const VectorXi &a){
    set<int> s;
    int size = a.size();
    for(int i = 0; i < size; ++i) s.insert(a(i));
    VectorXi c(s.size());
    set<int>::iterator it=s.begin();
    for(int i=0;i<s.size();++i){
        c[i]=*it;
        ++it;
    }
    return c;
}
