#include "consider_covariance.h"

void rule1assignstarter(int nlayer, int i,int j,MatrixXd &betastart,MatrixXd &alphastart, vector<MatrixXd>& Omegaisstart,const layers &L){
    if (nlayer==1) {
        if (i>0&&j==0) {//this is rule1
            betastart=L.mubeta[0];
            alphastart=L.mualpha[0];
            Omegaisstart=L.Omegais[0];
        }
    }
}

