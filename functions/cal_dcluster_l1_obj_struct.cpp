#include "consider_covariance.h"
double cal_dcluster_l1_obj_struct(const dclust_var_l1_obj_struct& A){
    int n=A.users.cols();
    int p=A.movie.cols();
    int udim2=A.udim2;
    int mdim2=A.mdim2;
    MatrixXd solu=A.users.transpose()*A.mualpha+(A.mubeta).transpose()*A.movie;
    
    double obj=0,temp1=0,temp2=0,temp3=0,temp4=0,temp5=0,temp6=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:temp1)
        for (int i=0; i<n-1; ++i) {
            for(int j=i+1;j<n;++j){
                temp1=temp1+(A.mubeta.col(i)-A.mubeta.col(j)).lpNorm<1>();
            }
        }
# pragma omp for reduction(+:temp2)
        for (int i=0; i<p-1; ++i) {
            for(int j=i+1;j<p;++j){
                temp2=temp2+(A.mualpha.col(i)-A.mualpha.col(j)).lpNorm<1>();
            }
        }
# pragma omp for reduction(+:temp3)
        for (int i=0; i<n; ++i) {
            int mi=A.user[i].item.size();
            resid.resize(mi);
            for(int j=0;j<mi;++j){
                resid[j]=solu(A.user[i].userno,A.user[i].item[j])-A.user[i].rating[j];
            }
            double error=resid.dot(resid);
            temp3=temp3+A.wis(i)*error-mi*log(A.wis(i));
        }
# pragma omp for reduction(+:temp4)
        for (int i=0; i<n-1; ++i) {
            for(int j=i+1;j<n;++j){
                temp4=temp4+abs(A.wis(i)-A.wis(j));
            }
        }
# pragma omp for reduction(+:temp5)
        for (int i=0; i<n; ++i) {
            VectorXd temp=A.mubeta.col(i).head(mdim2);
            temp5=temp5+cal_vec_pairdiff(temp);
        }
# pragma omp for reduction(+:temp6)
        for (int i=0; i<n; ++i) {
            VectorXd temp=A.mualpha.col(i).head(udim2);
            temp6=temp6+cal_vec_pairdiff(temp);
        }
    }
    obj=obj+temp1*A.lambda1/(2*A.c4)+temp2*A.lambda1/(2*A.c5)+temp3*0.5/A.c1+temp4*A.lambda2/A.c3+temp5*A.lambda1/(2*A.c7)+temp6*A.lambda1/(2*A.c6);
    obj=obj*100;
    return obj;
    
}

