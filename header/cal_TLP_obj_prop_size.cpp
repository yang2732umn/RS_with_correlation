#include "consider_covariance.h"
double cal_TLP_obj_prop_size(const vector<MatrixXd>& Sis,const TLP_obj_struct &A){
    int n=(*A.mubeta).rows();
    int p=(*A.mualpha).cols();
    int mdim=(*A.mubeta).cols();
    int udim=(*A.mualpha).rows();
    double obj=0,temp1=0,temp2=0,temp3=0,temp4=0,temp5=0;
    int i,j,l,k;
    MatrixXd Omegai;

	for (i=0; i<n-1; ++i) {
		for(j=i+1;j<n;++j){
		# pragma omp parallel for reduction(+:temp1)
			for (l=0; l<mdim; ++l) {
				temp1+=J_tau((*A.mubeta)(i,l)-(*A.mubeta)(j,l),A.tau);
			}
		}
	}
	for (i=0; i<p-1; ++i) {
		for(j=i+1;j<p;++j){
			# pragma omp parallel for reduction(+:temp2)
			for (l=0; l<udim; ++l) {
				temp2+=J_tau((*A.mualpha)(l,i)-(*A.mualpha)(l,j),A.tau);
			}
		}
	}
# pragma omp parallel for private(j,Omegai,k)  reduction(+:temp3,temp4)
	for (i=0; i<n; ++i) {
		Omegai=matrix_rowcolsub((*A.Omegais)[i],(*A.user)[i].item);
		for (j=0; j<p-1; ++j) {
			for (k=j+1; k<p; ++k) {
				temp4+=J_tau((*A.Omegais)[i](j,k),A.tau);
			}
		}
		if(isnan(Omegai)) {
			cout<<"NaN exists in Omegais[i]"<<endl;
			cout<<"Omegais["<<i<<"] is "<<endl;
			cout<<Omegai<<endl;
		}
		// double deter=1;
		temp3+=(Sis[i]*Omegai).trace()-(A.log_det)[i];
	}

    obj=temp1*A.lambda1/(2*A.c4*A.tau)+temp2*A.lambda1/(2*A.c5*A.tau)+temp3*0.5/A.c1+A.off_diff*A.lambda3/(2*A.c6*A.tau)+temp4*A.lambda2/(A.c3*A.tau);
    obj=obj*100;
    return obj; 
}
