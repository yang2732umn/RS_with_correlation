#include "consider_covariance.h"
void construct_Zinv(const int& n,const int& mdim,const int& mdim2,MatrixXd& Zinv, MatrixXd& A2){
    MatrixXd partXb=MatrixXd::Identity(mdim,mdim);
    MatrixXd tl=(1.0/(mdim2+1))*MatrixXd::Ones(mdim2,mdim2)+(1.0/(mdim2+1))*MatrixXd::Identity(mdim2,mdim2);//Xb topleft
    partXb.topLeftCorner(mdim2,mdim2)=tl;
    MatrixXd partXc=(1.0/(n+1))*MatrixXd::Identity(mdim,mdim);
    tl=(1.0/((mdim2+n+1)*(n+1)))*MatrixXd::Ones(mdim2,mdim2)+(1.0/(mdim2+n+1))*MatrixXd::Identity(mdim2,mdim2);//Xc topleft
    partXc.topLeftCorner(mdim2,mdim2)=tl;
    MatrixXd partXa=2*MatrixXd::Identity(mdim,mdim);
    partXa.topLeftCorner(mdim2,mdim2)=(-1)*MatrixXd::Ones(mdim2,mdim2)+(mdim2+2)*MatrixXd::Identity(mdim2,mdim2);
    MatrixXd partY=partXb*partXc;
    MatrixXd partX=partXa*partY;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i!=j) {
                Zinv.block(i*mdim, j*mdim, mdim, mdim) = partY;
            }
            else{
                Zinv.block(i*mdim, j*mdim, mdim, mdim) = partX;
            }
            
        }
    }
    MatrixXd C1=MatrixXd::Zero(mdim2*(mdim2-1)/2,mdim2);
#pragma omp parallel for
    for (int i=0; i<mdim2-1; ++i) {
        for (int j=i+1; j<mdim2; ++j) {
            int k=i*(mdim2-(i+1)*0.5)+j-i-1;
            C1(k,i)=1;
            C1(k,j)=-1;
        }
    }
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        A2.block(i*mdim2*(mdim2-1)/2, i*mdim,mdim2*(mdim2-1)/2,mdim2) =C1;
    }
}

