#include "consider_covariance.h"
double cal_obj_Omega2_struct(const obj_Omega2_struct &A){
    int n=A.users.rows();
    int p=A.movie.rows();
    double obj=0,temp=0;
# pragma omp parallel
    {
# pragma omp for reduction(+:obj)
        for (int i=0; i<n; ++i) {
            MatrixXd Si=A.Sis[i];
            MatrixXd Omegai=A.Omegais[i];
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
            obj=obj+((Si*Omegai).trace()-log((Omegai).determinant()));
        }
# pragma omp single
        {
            obj=obj*0.5/A.c1;
        }
# pragma omp for reduction(+:temp)
        for (int i=0; i<n; ++i) {//
            MatrixXd Omegai=A.Omegais[i];
            temp=temp+(absm(Omegai).sum()-absm(Omegai).trace());
        }
    }
    obj=obj+A.lambda2*temp/A.c3;
    obj=obj+A.lambda3*A.off_diff/A.c6;
    obj=obj*100;
    return obj;
}
