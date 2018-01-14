#include "consider_covariance.h"

//designed for lambda1==lambda3
void rule1assignlayer(int nlayer, int i,int j,int lam1size,const MatrixXd &betanow,const MatrixXd &alphanow ,const vector<MatrixXd> & Omegaisnow,layers &L){
    //if i,j, nlayer=1(L1) currently just for L1, nlayer always 1
    L.mualpha.resize(nlayer);
    L.mubeta.resize(nlayer);
    L.Omegais.resize(nlayer);
    if (nlayer==1) {
        if (lam1size>1&&j==0&&(i!=lam1size-1)) {//this is rule1
            L.mubeta[0]=betanow;
            L.mualpha[0]=alphanow;
            L.Omegais[0]=Omegaisnow;
        }
    }
}



