#include "consider_covariance.h"
double cal_beta_fuse_abs(const tri &trin,const MatrixXd &mubeta,const vector<vector<int>> &diffB){
    double betafuse=0;
    int n=mubeta.rows();
    int l,i,j,k;
#pragma omp parallel for private(i,j,k) reduction(+:betafuse) 
	for (l=0; l<trin.first.size(); ++l) {
		i=trin.first(l);
		j=trin.second(l);
		for (k=0; k<diffB[l].size(); ++k) {
                betafuse+=abs(mubeta(i,diffB[l][k])-mubeta(j,diffB[l][k]));
        }
    }
    /* 
for (int i=0; i<n-1; ++i) {
        for(int j=i+1;j<n;++j){
            int l=i*(n-(i+1)*0.5)+j-i-1;
            for (int k=0; k<diffB[l].size(); ++k) {
                betafuse+=abs(mubeta(i,diffB[l][k])-mubeta(j,diffB[l][k]));
            }
        }
    }
 */
    return betafuse;
}