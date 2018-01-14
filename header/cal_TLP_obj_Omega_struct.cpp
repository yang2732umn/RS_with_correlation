#include "consider_covariance.h"
double cal_TLP_obj_Omega_struct(const TLP_obj_Omega_struct &A){
    //use double off_diff
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
        /*# pragma omp for reduction(+:temp)
         for (int i=0; i<n; ++i) {//
         MatrixXd Omegai=A.Omegais[i];
         for (int j=0; j<Omegai.rows()-1; ++j) {
         for (int l=j+1; l<Omegai.rows(); ++l) {
         temp+=J_tau(Omegai(j,l),A.tau);
         }
         }
         }*/
    }
    //obj=obj+2*A.lambda2*temp/(A.c3*A.tau);
    //cout<<"l1 norm part of Omega is "<<temp<<endl;
    obj=obj+A.lambda3*A.off_diff/(A.c6*A.tau);
    obj=obj*100;
    return obj;
}