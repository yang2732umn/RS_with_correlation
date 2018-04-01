#include "consider_covariance.h"
double cal_offdiff_prop(const tri &trin,const vector<MatrixXd>& Omegais){
    double temp5=0;
    int p=Omegais[1].rows();
    int n=Omegais.size();
    int i,j;
/* 
#pragma omp parallel for reduction(+:temp5)
    for (int i=0; i<n-1; ++i) {
        for (int j=i+1; j<n; ++j) {//why 0 and n-1 before?
            temp5+=(Omegais[i]-Omegais[j]).lpNorm<1>();
            temp5+=(Omegais[i].diagonal()-Omegais[j].diagonal()).lpNorm<1>();
        }
    }
 */    
    /* 
#pragma omp parallel for reduction(+:temp5)
    for (int l=0; l<trin.first.size(); ++l) {
    	int i=trin.first(l);int j=trin.second(l);
    	temp5+=(Omegais[i]-Omegais[j]).lpNorm<1>();
        temp5+=(Omegais[i].diagonal()-Omegais[j].diagonal()).lpNorm<1>();
    }
    temp5=temp5/2;
 */
    MatrixXd mtemp,mtemp2;
    for (i=0;i<n-1;++i) {
    	mtemp=Omegais[i];
    #pragma omp parallel for private(mtemp2,j)  reduction(+:temp5)//using private is better than not using private, don't know why
    	for(j=i+1;j<n;++j){
    		mtemp2=mtemp-Omegais[j];
    		temp5+=mtemp2.lpNorm<1>();
        	temp5+=mtemp2.diagonal().lpNorm<1>();
    	}
    }
    temp5=temp5/2;
    return temp5;
}
