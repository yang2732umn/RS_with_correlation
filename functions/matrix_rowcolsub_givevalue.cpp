#include "consider_covariance.h"
void matrix_rowcolsub_givevalue(MatrixXd &a, const vector<int> &b,const MatrixXd &c){
    for (int i=0; i<b.size(); ++i) {
        for (int j=0; j<b.size(); ++j) {
            a(b[i],b[j])=c(i,j);
        }
    }
}
