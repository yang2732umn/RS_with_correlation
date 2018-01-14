#include "consider_covariance.h"
double cal_likelihood_d(const dcluster_var_l1_para &C){
    //diagonal precision matrix, each person different sigma
    int n=C.users.cols();
    int p=C.movie.cols();
    MatrixXd solu=C.users.transpose()*C.mualpha1+C.mubeta1.transpose()*C.movie;
    double temp3=0;
    vector<VectorXd> resid=cal_resid(C.mualpha1,C.mubeta1,C.x.user,C.movie,C.users);
# pragma omp parallel for reduction(+:temp3)
    for (int i=0; i<C.x.user.size(); ++i) {
        double si=resid[i].dot(resid[i]);
        temp3+=C.wis1[i]*si-(C.x.user[i].item.size())*log(C.wis1[i]);
    }
    return temp3;//gives -log likelihood
}
