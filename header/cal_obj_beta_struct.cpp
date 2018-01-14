#include "consider_covariance.h"
double cal_obj_beta_struct(const obj_beta_struct &A){//modified to l2 norm for TLP
    int n=A.users.rows();
    int p=A.movie.rows();
    MatrixXd solu=A.users*A.mualpha+A.mubeta*A.movie.transpose();
    double obj=0,temp=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:obj)
        for (int i=0; i<A.mubeta.rows()-1; ++i) {
            for(int j=i+1;j<A.mubeta.rows();++j){
                obj=obj+(A.mubeta.row(i)-A.mubeta.row(j)).lpNorm<2>();
            }
        }
# pragma omp single
        {
            obj=obj*A.lambda1/(2*A.c4);
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
            temp=temp+((Si*Omegai).trace());
        }
    }
    obj=obj+temp*0.5/A.c1;
    obj=obj*100;
    return obj;
}

