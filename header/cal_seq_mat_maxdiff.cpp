#include "consider_covariance.h"
double cal_seq_mat_maxdiff(const vector<MatrixXd> matseq1,const vector<MatrixXd> matseq2){
    double diff=0;
    int n=matseq1.size();//two seqs should have the same number of matrices
    VectorXd alldiff(n);
# pragma omp parallel for
    for (int i=0; i<n; ++i) {
        alldiff[i]=absm(matseq1[i]-matseq2[i]).maxCoeff();
    }
    diff=alldiff.lpNorm<Infinity>();
    return diff;
}
