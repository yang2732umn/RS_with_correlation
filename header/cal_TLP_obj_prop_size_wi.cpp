#include "consider_covariance.h"
double cal_TLP_obj_prop_size_wi(int n,int c1,int c3,int c4,int c5,int c6,double lambda1,double lambda2,double lambda3,double tau,double alphafuse_tau,double betafuse_tau,double Omegasize_tau,
	double Omegafuse_tau, const VectorXd &log_det,const vector<MatrixXd>& Sis,const vector<rated_user> &user,const vector<MatrixXd>& Omegais){//with info
    //user should be train.user
    //also with size contraint on Omega
    double obj=0,temp3=0;
    int i;
    MatrixXd Omegai;
# pragma omp parallel for private(Omegai)  reduction(+:temp3)
	for (i=0; i<n; ++i) {
		Omegai=matrix_rowcolsub((Omegais)[i],user[i].item);  
		if(isnan(Omegai)) {
			cout<<"NaN exists in Omegais[i]"<<endl;
			cout<<"Omegais["<<i<<"] is "<<endl;
			cout<<Omegai<<endl;
		}
		temp3+=(Sis[i]*Omegai).trace()-(log_det)[i];
	}
	obj=betafuse_tau*lambda1/(2*c4*tau)+alphafuse_tau*lambda1/(2*c5*tau)+temp3*0.5/c1+Omegafuse_tau*lambda3/(2*c6*tau)+Omegasize_tau*lambda2/(c3*tau);
    obj=obj*100;
    return obj; 
}
