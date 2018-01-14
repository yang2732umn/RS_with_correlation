#include "consider_covariance.h"

//designed for lambda1==lambda3, and just for L1, the same as rule2 nlayer=2
void rule1assignstarter(int nlayer, int i,int j,MatrixXd &betastart,MatrixXd &alphastart, vector<MatrixXd>& Omegaisstart,const layers &L){
    //if i,j, nlayer=1, always used for nlayer==1
    if (nlayer==1) {
        if (i>0&&j==0) {//this is rule1
            betastart=L.mubeta[0];
            alphastart=L.mualpha[0];
            Omegaisstart=L.Omegais[0];
        }
    }
}

