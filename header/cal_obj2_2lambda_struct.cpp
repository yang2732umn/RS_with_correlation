#include "consider_covariance.h"
double cal_obj2_2lambda_struct(const two_lambda_struct &A){
    //user should be train.user
    int n=A.users.rows();
    int p=A.movie.rows();
    MatrixXd solu=A.users*A.mualpha+A.mubeta*A.movie.transpose();
    double obj=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:obj)
        for (int i=0; i<n-1; ++i) {
            for(int j=i+1;j<n;++j){
                obj=obj+(A.mubeta.row(i)-A.mubeta.row(j)).lpNorm<1>()*A.lambda1/2;
            }
        }
# pragma omp for reduction(+:obj)
        for (int i=0; i<p-1; ++i) {
            for(int j=i+1;j<p;++j){
                obj=obj+(A.mualpha.col(i)-A.mualpha.col(j)).lpNorm<1>()*A.lambda2/2;
            }
        }
        
# pragma omp for reduction(+:obj)
        for (int i=0; i<A.user.size(); ++i) {
            resid.resize(A.user[i].item.size());
            for(int j=0;j<A.user[i].item.size();++j){
                resid[j]=solu(A.user[i].userno,A.user[i].item[j])-A.user[i].rating[j];
            }
            
            obj=obj+0.5*resid.squaredNorm();
        }
    }
    return obj;
}
