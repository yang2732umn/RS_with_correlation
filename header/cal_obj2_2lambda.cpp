#include "consider_covariance.h"
double cal_obj2_2lambda(const MatrixXd& mualpha,const MatrixXd& mubeta, double lambda1, double lambda2,const vector<rated_user>& user,const MatrixXd& movie, const MatrixXd& users){
    //user should be train.user
    int n=users.rows();
    int p=movie.rows();
    MatrixXd solu=users*mualpha+mubeta*movie.transpose();
    double obj=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:obj)
        for (int i=0; i<mubeta.rows()-1; ++i) {
            for(int j=i+1;j<mubeta.rows();++j){
                obj=obj+(mubeta.row(i)-mubeta.row(j)).lpNorm<1>()*lambda1/2;
            }
        }
# pragma omp for reduction(+:obj)
        for (int i=0; i<mualpha.cols()-1; ++i) {
            for(int j=i+1;j<mualpha.cols();++j){
                obj=obj+(mualpha.col(i)-mualpha.col(j)).lpNorm<1>()*lambda2/2;
            }
        }
        
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
