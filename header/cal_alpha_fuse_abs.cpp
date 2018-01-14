#include "consider_covariance.h"
double cal_alpha_fuse_abs(const tri &trip,const MatrixXd &mualpha,const vector<vector<int>> &diffA){
    double alphafuse=0;
    int p=mualpha.cols();
    int l,i,j,k;
#pragma omp parallel for private(i,j,k) reduction(+:alphafuse)
	for (l=0; l<trip.first.size(); ++l) {
		i=trip.first(l);
		j=trip.second(l);
        for (k=0; k<diffA[l].size(); ++k) {
            alphafuse+=abs(mualpha(diffA[l][k],i)-mualpha(diffA[l][k],j));
        }
    }
    return alphafuse;
}
