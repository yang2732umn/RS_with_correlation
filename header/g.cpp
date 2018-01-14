#include "consider_covariance.h"
double g(const MatrixXd& Omega,const vector<rated_user>& user,double lambda3,const vector<MatrixXd>& Sis){
    //not used
    double s=0;
#pragma omp parallel
    {
# pragma omp for reduction(+:s)
        for (int i=0; i<user.size(); ++i) {
            MatrixXd Omegai=matrix_rowcolsub(Omega, user[i].item);
            s=s+(Omegai*Sis[i]).trace()-log(Omegai.determinant());
        }
    }
    s=s/2;
    s=s+lambda3*(absm(Omega).sum()-absm(Omega).trace());//only l1-norm of off diagonal
    return s;
} 