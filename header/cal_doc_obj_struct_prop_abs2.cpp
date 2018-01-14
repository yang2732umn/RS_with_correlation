#include "consider_covariance.h"
double cal_doc_obj_struct_prop_abs2(double c,const MatrixXd& betapre,const MatrixXd& alphapre,const vector<MatrixXd>& Omegaispre,const vector<MatrixXd>& Sis,const doc_obj_struct &A){
    //inside doc, abs obj value, but with quadratic adjustment for non-convexity
    //used in Cluster_TLP_scale_prop_size3
    //user should be train.user
    //with L1 penalty on Omega size also
    //within doc, using previous alpha, beta and Omega to determine set
    //diffA is for alpha, diffB for beta, diffO is for Omega
    int n=(*A.mubeta).rows();
    int p=(*A.mualpha).cols();
    int mdim=(*A.mubeta).cols();
    int udim=(*A.mualpha).rows();
    double obj=0,temp3=0,temp5=0,temp6=0;
# pragma omp parallel
    {
# pragma omp for reduction(+:temp3,temp5,temp6)
        for (int i=0; i<n; ++i) {
            MatrixXd Omegai=matrix_rowcolsub((*A.Omegais)[i],(*A.user)[i].item);
            MatrixXd Omegaipre=matrix_rowcolsub(Omegaispre[i],(*A.user)[i].item);
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
			temp3+=(Sis[i]*Omegai).trace()-(*A.log_det)[i];         
            temp5+=Omegai.squaredNorm();
            temp6+=(Omegaipre.array()*Omegai.array()).sum();
        }
    }//lambda2=0
    obj=obj+temp3*0.5/A.c1+A.alphafuse*A.lambda1*0.5/(A.c5*A.tau)+A.betafuse*A.lambda1*0.5/(A.c4*A.tau)+A.lambda3*0.5/(A.c6*A.tau)*A.Omegafusesize.first+A.lambda2/(A.c3*A.tau)*A.Omegafusesize.second;//check def of c1-c6
    temp5+=(*A.mubeta).squaredNorm()+(*A.mualpha).squaredNorm();
    temp6+=(betapre.array()*(*A.mubeta).array()).sum();
    temp6+=(alphapre.array()*(*A.mualpha).array()).sum();
    obj=obj+c*temp5/A.c1-2*c*temp6/A.c1;
    obj=obj*100;  
    return obj;
}

