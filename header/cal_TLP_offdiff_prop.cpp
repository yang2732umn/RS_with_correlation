#include "consider_covariance.h"
double cal_TLP_offdiff_prop(double tau,const vector<MatrixXd>& Omegais){
    //this calculates TLP  diff terms for Omega
    double temp5=0;
    int p=Omegais[1].rows();
    int n=Omegais.size();
/* 
#pragma omp parallel for reduction(+:temp5)
    for (int l=0; l<trin.first.size(); ++l) {
    	int i=trin.first(l);int j=trin.second(l);
        temp5+=matrix_Jtaunorm(tau,Omegais[i]-Omegais[j]);
    }
 */ 
 	MatrixXd mtemp,mtemp2;
    int i,j;
    double temp6=0;
    for (i=0;i<n-1;++i) {
    	mtemp=Omegais[i];
    #pragma omp parallel for private(mtemp2,j,temp6) reduction(+:temp5)//using private is better than not using private, don't know why
    	for(j=i+1;j<n;++j){
    		mtemp2=(mtemp-Omegais[j]).cwiseAbs(); 
    		//temp5+=(mtemp2.sum()+mtemp2.trace())/2;   
    		temp5+=po_matrix_Jtaunorm(tau,mtemp2);
    	}
    }
    return temp5;
} 