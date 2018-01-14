#include "consider_covariance.h"
double cal_novariance_likelihood(MatrixXd mualpha, MatrixXd mubeta, vector<rated_user> user, MatrixXd movie, MatrixXd users){
    double obj=0;
    MatrixXd solu=users*mualpha+mubeta*movie.transpose();
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:obj)
        for (int i=0; i<user.size(); ++i) {
            resid.resize(user[i].item.size());
            for(int j=0;j<user[i].item.size();++j){
                resid[j]=solu(user[i].userno,user[i].item[j])-user[i].rating[j];
            }
            obj=obj+0.5*resid.squaredNorm();
        }
    }
    return obj;
    
}