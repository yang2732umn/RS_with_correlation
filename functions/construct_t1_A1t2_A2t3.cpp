#include "consider_covariance.h"
void construct_t1_A1t2_A2t3(const int& n,const int& mdim,const int& mdim2,const vector<VectorXi>& fdibeta,const vector<VectorXi>& sdibeta,const vector<VectorXi>& fdibetainner,const vector<VectorXi>& sdibetainner,const MatrixXd& mubeta2, MatrixXd* gamma,MatrixXd* theta,MatrixXd* u,MatrixXd* eta,MatrixXd* v, VectorXd &t1,VectorXd& A1t2,VectorXd& A2t3){
    int i,j,k;
    VectorXi fdi,sdi;
    A1t2=VectorXd::Zero(A1t2.size());
    A2t3=VectorXd::Zero(A2t3.size());
#pragma omp parallel for private(j,k,fdi,sdi)
    for (i=0; i<n; ++i) {
        t1.segment(i*mdim,mdim)=mubeta2.col(i)+(*gamma).col(i);
        if(i<n-1){
            fdi=fdibeta[i];
            for (j=0; j<fdi.size(); ++j) {
                A1t2.segment(i*mdim,mdim)+=((*theta).col(fdi[j])-(*u).col(fdi[j]));
            }
        }
        if(i>0){
            sdi=sdibeta[i-1];
            for (j=0; j<sdi.size(); ++j) {
                A1t2.segment(i*mdim,mdim)-=((*theta).col(sdi[j])-(*u).col(sdi[j]));
            }
        }
        
        for (j=0; j<mdim2; ++j) {
            if(j<mdim2-1){
                fdi=fdibetainner[j];
                for (k=0; k<fdi.size(); ++k) {
                    A2t3(i*mdim+j)+=((*eta)(fdi[k],i)-(*v)(fdi[k],i));
                }
            }
            if(j>0){
                sdi=sdibetainner[j-1];
                for (k=0; k<sdi.size(); ++k) {
                    A2t3(i*mdim+j)-=((*eta)(sdi[k],i)-(*v)(sdi[k],i));
                }
            }
        }
    }
    
}

