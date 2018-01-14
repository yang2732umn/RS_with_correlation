#include "consider_covariance.h"
double cal_obj_prop_struct_3(const vector<MatrixXd>& Sis,const obj_struct &A)
//lambda3=lambda1, used for with L1 penalty on Omega
{
    //user should be train.user
    int n=(*A.mubeta).rows();
    int p=(*A.mualpha).cols();
    double obj=0,temp1=0,temp2=0,temp3=0,temp4=0,temp5=0;
# pragma omp parallel
    {
# pragma omp for reduction(+:temp1)
        for (int i=0; i<n-1; ++i) {
            for(int j=i+1;j<n;++j){
                temp1=temp1+((*A.mubeta).row(i)-(*A.mubeta).row(j)).lpNorm<1>();
            }
        }
# pragma omp for reduction(+:temp2)
        for (int i=0; i<p-1; ++i) {
            for(int j=i+1;j<p;++j){
                temp2=temp2+((*A.mualpha).col(i)-(*A.mualpha).col(j)).lpNorm<1>();
            }
        }
# pragma omp for reduction(+:temp3,temp4)
        for (int i=0; i<n; ++i) {
            MatrixXd Omegai=matrix_rowcolsub((*A.Omegais)[i],(*A.user)[i].item);
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
//            LLT<MatrixXd> llt(Omegai);
//            if(llt.info() == NumericalIssue) cout<<"Omega "<<i<<" is not positive definite"<<endl;
//            double determ=llt.matrixL().determinant();
            temp3=temp3+(Sis[i]*Omegai).trace()-(*A.log_det)[i];//*log(determ);
            temp4=temp4+(absm((*A.Omegais)[i]).sum()-absm((*A.Omegais)[i]).trace());
        }
    }
    //cout<<"liki part is "<<temp3*0.5/A.c1<<" ,alpha part is "<<temp2*A.lambda1/(2*A.c5)<<" ,beta part is "<<temp1*A.lambda1/(2*A.c4)<<", Omega part is "<<A.off_diff*A.lambda3/(A.c6)<<endl;
    obj=obj+temp1*A.lambda1/(2*A.c4)+temp2*A.lambda1/(2*A.c5)+temp3*0.5/A.c1+temp4*A.lambda2/A.c3+A.off_diff*A.lambda3/(2*A.c6);
    obj=obj*100;
    return obj;
}