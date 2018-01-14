#include "consider_covariance.h"

//designed for lambda1!=lambda3, also works for lambda1==lambda3
void rule2assignstarter(int nlayer, int i,int j,int k,int l,MatrixXd &betastart,MatrixXd &alphastart, vector<MatrixXd>& Omegaisstart,const layers &L){
    //if i,j,k,l, nlayer=3; if i,j,k, nlayer=2
    if (nlayer==2) {
        if (i>0&&j==0&&k==0) {
            betastart=L.mubeta[0];
            alphastart=L.mualpha[0];
            Omegaisstart=L.Omegais[0];
        }
        if (j>0&&k==0) {
            betastart=L.mubeta[1];
            alphastart=L.mualpha[1];
            Omegaisstart=L.Omegais[1];
        }
    }
    if (nlayer==3) {
        if (i>0&&j==0&&k==0&&l==0) {
            betastart=L.mubeta[0];
            alphastart=L.mualpha[0];
            Omegaisstart=L.Omegais[0];
        }
        if (j>0&&k==0&&l==0) {
            betastart=L.mubeta[1];
            alphastart=L.mualpha[1];
            Omegaisstart=L.Omegais[1];
        }
        if (k>0&&l==0) {
            betastart=L.mubeta[2];
            alphastart=L.mualpha[2];
            Omegaisstart=L.Omegais[2];
        }
    }
    
}

