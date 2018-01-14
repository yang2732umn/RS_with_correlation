#include "consider_covariance.h"
double cal_TLP_offdiff_prop2(double tau,vector<MatrixXd>* Omegais){
    //this calculates TLP  diff terms for Omega
    double temp5=0;
    int p=(*Omegais)[0].rows();
    int n=Omegais->size();//(*Omegais).size();
/* 
#pragma omp parallel for reduction(+:temp5)
    for (int l=0; l<trin.first.size(); ++l) {
    	int i=trin.first(l);int j=trin.second(l);
        temp5+=matrix_Jtaunorm(tau,Omegais[i]-Omegais[j]);
    }
 */ MatrixXd mtemp,mtemp2;
    int i,j;
    for (i=0;i<n-1;++i) {
    	mtemp=(Omegais->at(i));
    #pragma omp parallel for private(mtemp2,j) reduction(+:temp5)//using private is better than not using private, don't know why
    	for(j=i+1;j<n;++j){
    		mtemp2=mtemp-(Omegais->at(j));    
    		temp5+=matrix_Jtaunorm(tau,mtemp2);
    	}
    }
    return temp5;
} 