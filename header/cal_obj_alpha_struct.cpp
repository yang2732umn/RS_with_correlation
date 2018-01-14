#include "consider_covariance.h"
double cal_obj_alpha_struct(const obj_beta_struct &A){
    //user should be train.user
    int n=A.users.rows();
    int p=A.movie.rows();
    MatrixXd solu=A.users*A.mualpha+A.mubeta*A.movie.transpose();
    double obj=0,temp=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:obj)
        for (int i=0; i<A.mualpha.cols()-1; ++i) {
            for(int j=i+1;j<A.mualpha.cols();++j){
                obj=obj+(A.mualpha.col(i)-A.mualpha.col(j)).lpNorm<1>();
            }
        }
# pragma omp single
        {
            obj=obj*A.lambda1/(2*A.c5);
        }
        
# pragma omp for reduction(+:temp)
        for (int i=0; i<A.user.size(); ++i) {
            resid.resize(A.user[i].item.size());
            for(int j=0;j<A.user[i].item.size();++j){
                resid[j]=solu(A.user[i].userno,A.user[i].item[j])-A.user[i].rating[j];
            }
            MatrixXd Si=resid*resid.transpose();
            MatrixXd Omegai=A.Omegais[i];
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
            temp=temp+(Si*Omegai).trace();
        }
    }
    obj=obj+0.5*temp/A.c1;
    obj=obj*100;
    return obj;
}

