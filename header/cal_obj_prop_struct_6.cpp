#include "consider_covariance.h"
double cal_obj_prop_struct_6(double c,const tri &trin,const tri &trip,const MatrixXd& betapre,const MatrixXd& alphapre,const vector<MatrixXd>& Omegaispre,const vector<MatrixXd>& Sis,const obj_struct &A){//lambda3=lambda1
    //use for inner doc(admm) objective function for L1 version with L1 penalty on Omega (Cluster_p_inADMM_scale_struct_v6)
    //c is the constant used for quadratic term, how to determine/solve for c
    //Omegaispre is from previous doc
    int n=(*A.mubeta).rows();
    int p=(*A.mualpha).cols();
    double obj=0,temp1=0,temp2=0,temp3=0,temp4=0,temp5=0,temp6=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:temp1)
        for (int l=0; l<trin.first.size(); ++l) {
        	int i=trin.first(l);int j=trin.second(l);
            temp1=temp1+((*A.mubeta).row(i)-(*A.mubeta).row(j)).lpNorm<1>();
        }
# pragma omp for reduction(+:temp2)
		for (int l=0; l<trip.first.size(); ++l) {
        	int i=trip.first(l);int j=trip.second(l);
            temp2=temp2+((*A.mualpha).col(i)-(*A.mualpha).col(j)).lpNorm<1>();
        }
# pragma omp for reduction(+:temp3,temp5,temp6)
        for (int i=0; i<(*A.user).size(); ++i) {
            MatrixXd Omegai=matrix_rowcolsub((*A.Omegais)[i],(*A.user)[i].item);
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
//            LLT<MatrixXd> llt(Omegai);
//            if(llt.info() == NumericalIssue) cout<<"Omega "<<i<<" is not positive definite"<<endl;
//            double determ=llt.matrixL().determinant();
            /*double determ=Omegai.determinant();
            if (determ<=0) {
                cout<<"Omega "<<i<<" negative determined or undetermined!"<<endl;
            }*/
            temp3=temp3+(Sis[i]*Omegai).trace()-(*A.log_det)[i];//*log(determ);
            //temp4=temp4+(absm((*A.Omegais)[i]).sum()-absm((*A.Omegais)[i]).trace());
            temp5+=Omegai.squaredNorm();
            temp6+=(Omegaispre[i].array()*Omegai.array()).sum();
        }
    }
    //cout<<"liki part is "<<temp3*0.5/A.c1<<" ,alpha part is "<<temp2*A.lambda1/(2*A.c5)<<" ,beta part is "<<temp1*A.lambda1/(2*A.c4)<<", Omega part is "<<A.off_diff*A.lambda3/(A.c6)<<endl;
    obj=obj+temp1*A.lambda1/(2*A.c4)+temp2*A.lambda1/(2*A.c5)+temp3*0.5/A.c1+2*A.Osize*A.lambda2/A.c3+A.off_diff*A.lambda3/(2*A.c6);
    temp5+=(*A.mubeta).squaredNorm()+(*A.mualpha).squaredNorm();
    obj=obj+c*temp5/A.c1;
    temp6+=(betapre.array()*(*A.mubeta).array()).sum();
    temp6+=(alphapre.array()*(*A.mualpha).array()).sum();
    obj=obj-2*c*temp6/A.c1; 
    obj=obj*100;
    return obj;
}


