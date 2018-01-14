#include "consider_covariance.h"
VectorXi unique_veci(const VectorXi &a){
    //get unique values of a VectorXi
    set<int> s;
    int size = a.size();
    for(int i = 0; i < size; ++i) s.insert(a(i));
    //set automatically orders objects
    VectorXi c(s.size());
    set<int>::iterator it=s.begin();
    for(int i=0;i<s.size();++i){
        c[i]=*it;
        ++it;
    }
    return c;
}