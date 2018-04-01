#include "consider_covariance.h"
double cal_offdiff(const vector<MatrixXd>& Omegais,const vector<Omega_1rlv>& rlv,const vector<rated_1itemmore>& itemmore)
{
    
    double temp5=0;
    int p=itemmore.size();
# pragma omp parallel
    {
# pragma omp for reduction(+:temp5)
        for (int i=0; i<rlv.size(); ++i) {//ith pair
            for (int j=0; j<rlv[i].userno.size()-1; ++j) {
                MatrixXd Omegaj=Omegais[rlv[i].userno[j]];
                int ji1=rlv[i].firstin[j];
                int ji2=rlv[i].secin[j];
                double a1=Omegaj(ji1,ji2);
                for (int l=j+1; l<rlv[i].userno.size(); ++l) {
                    MatrixXd Omegal=Omegais[rlv[i].userno[l]];
                    int li1=rlv[i].firstin[l];
                    int li2=rlv[i].secin[l];
                    temp5=temp5+abs(a1-Omegal(li1,li2));
                }
            }
        }
# pragma omp for reduction(+:temp5)
        for (int i=0; i<p; ++i) {//ith pair
            int usersize=itemmore[i].user.size();
            for (int j=0; j<usersize-1; ++j) {
                int userno1=itemmore[i].user[j];
                int index1=itemmore[i].numberforuser[j];
                double a1=Omegais[userno1](index1,index1);
                for (int l=j+1; l<usersize; ++l) {
                    int userno2=itemmore[i].user[l];
                    int index2=itemmore[i].numberforuser[l];
                    temp5=temp5+abs(a1-Omegais[userno2](index2,index2));
                }
            }
        }
    }
    return temp5;
}
