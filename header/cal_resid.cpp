#include "consider_covariance.h"
vector<VectorXd> cal_resid(const MatrixXd &mualpha,const MatrixXd &mubeta,const vector<rated_user> &user,const MatrixXd& movie,const MatrixXd& users){
    int n=mubeta.cols();
    vector<VectorXd> resid(n);
    //calculate residuals for each user
#pragma omp parallel for//will not exist further parallelization outside(because is for all users already), can use parallel here
    for (int i=0; i<n; ++i) {//construct pretilde
        MatrixXd alphause=matrix_colsub(mualpha,user[i].item);
        MatrixXd movieuse=matrix_colsub(movie,user[i].item);
        VectorXd useri=users.col(i);
        VectorXd btemp=mubeta.col(i);
        resid[i]=alphause.transpose()*useri+movieuse.transpose()*btemp;
        resid[i]=user[i].rating-resid[i];
    }
    return resid;
}
