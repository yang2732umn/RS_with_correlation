#include "consider_covariance.h"

//also works for lambda1==lambda3
void rule2assignlayer(int nlayer, int i,int j,int k,int l,int lam1size,int lam2size,int lam3size,const MatrixXd &betanow,const MatrixXd &alphanow ,const vector<MatrixXd> & Omegaisnow,layers &L){
    //if i,j,k,l, nlayer=3; if i,j,k, nlayer=2
    L.mualpha.resize(nlayer);
    L.mubeta.resize(nlayer);
    L.Omegais.resize(nlayer);
    if (nlayer==2) {
        if (lam1size>1&&j==0&&k==0&&i<lam1size-1) {//layer1
            L.mubeta[0]=betanow;
            L.mualpha[0]=alphanow;
            L.Omegais[0]=Omegaisnow;
        }
        if(lam2size>1&&k==0&&j<lam2size){//layer2
            L.mubeta[1]=betanow;
            L.mualpha[1]=alphanow;
            L.Omegais[1]=Omegaisnow;
        }
    }
    if(nlayer==3) {
        if (lam1size>1&&j==0&&k==0&&l==0) {//layer1
            L.mubeta[0]=betanow;
            L.mualpha[0]=alphanow;
            L.Omegais[0]=Omegaisnow;
        }
        if(lam2size>1&&k==0&&l==0){//layer2
            L.mubeta[1]=betanow;
            L.mualpha[1]=alphanow;
            L.Omegais[1]=Omegaisnow;
        }
        if(lam3size>1&&l==0){//layer3
            L.mubeta[2]=betanow;
            L.mualpha[2]=alphanow;
            L.Omegais[2]=Omegaisnow;
        }
    }
}



