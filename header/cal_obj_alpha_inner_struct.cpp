#include "consider_covariance.h"
double cal_obj_alpha_inner_struct(const obj_alpha_inner_struct &A,const vector<vector<int>>& diffA){//modified for TLP
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
                int l=i*(p-(i+1)*0.5)+j-i-1;
                if (diffA[l].size()>0) {
                    obj=obj+(A.mualpha.col(i)-A.mualpha.col(j)-(*(A.theta2)).col(l)+(*(A.u2)).col(l)).squaredNorm();
                }
            }
        }
# pragma omp single
        {
            obj=obj*A.rho_alpha/2;
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
    obj=obj+0.5*temp;
    return obj;
}

