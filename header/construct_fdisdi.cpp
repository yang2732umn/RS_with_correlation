#include "consider_covariance.h"
void construct_fdisdi(const int& n,vector<VectorXi> &fdi,vector<VectorXi> &sdi){//for all i from 1 to n
    //fdi and sdi should be of size n-1 already
#pragma omp parallel for//use parallel because already for all i
    for (int i=0; i<n; ++i) {
        if(i<n-1){
            fdi[i]=VectorXi::LinSpaced(n-i-1,i*(n-(i+1)*0.5),i*(n-(i+1)*0.5)+n-i-2);//第几个约束(i,)
        }
        if(i>0){
            sdi[i-1]=constructsdi(i+1, n);//第几个约束(,i)
        }
    }
}