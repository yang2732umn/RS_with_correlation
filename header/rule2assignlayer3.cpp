#include "consider_covariance.h"

//also works for lambda1==lambda3
void rule2assignlayer3(int nlayer, int i,int j,int k,int l,int lam1size,int lam2size,int lam3size,const MatrixXd &betanow,const MatrixXd &alphanow,const MatrixXd& unow,const MatrixXd& u2now,const MatrixXd& thetanow,const MatrixXd& theta2now,const vector<MatrixXd> & Omegaisnow,const vector<vector<double>>& Xnow,const vector<vector<double>>& Znow,const vector<vector<double>>& eta_now,const vector<vector<double>>& y_now,layers3 &L,string &namenow){ 
    //if i,j,k,l, nlayer=3; if i,j,k, nlayer=2
    if (nlayer==2) {
        if (lam1size>1&&j==0&&k==0&&i<lam1size-1) {//layer1
            L.mubeta[0]=betanow;
            L.mualpha[0]=alphanow;
            L.Omegais[0]=Omegaisnow;
            L.u[0]=unow;
            L.theta[0]=thetanow;
            L.u2[0]=u2now;
            L.theta2[0]=theta2now;
            L.X[0]=Xnow;
            L.Z[0]=Znow;
            L.eta[0]=eta_now;
            L.y[0]=y_now;
            L.names[0]=namenow;
        }
        if(lam2size>1&&k==0&&j<lam2size-1){//layer2
            L.mubeta[1]=betanow;
            L.mualpha[1]=alphanow;
            L.Omegais[1]=Omegaisnow;
            L.u[1]=unow;
            L.theta[1]=thetanow;
            L.u2[1]=u2now;
            L.theta2[1]=theta2now;
            L.X[1]=Xnow;
            L.Z[1]=Znow;
            L.eta[1]=eta_now;
            L.y[1]=y_now;
            L.names[1]=namenow;
        }
    }
    if(nlayer==3) {
        if (lam1size>1&&j==0&&k==0&&l==0&&i<lam1size-1) {//layer1
            L.mubeta[0]=betanow;
            L.mualpha[0]=alphanow;
            L.Omegais[0]=Omegaisnow;
            L.u[0]=unow;
            L.theta[0]=thetanow;
            L.u2[0]=u2now;
            L.theta2[0]=theta2now;
            L.X[0]=Xnow;
            L.Z[0]=Znow;
            L.eta[0]=eta_now;
            L.y[0]=y_now;
            L.names[0]=namenow;
        }
        if(lam2size>1&&k==0&&l==0&&j<lam2size-1){//layer2
            L.mubeta[1]=betanow;
            L.mualpha[1]=alphanow;
            L.Omegais[1]=Omegaisnow;
            L.u[1]=unow;
            L.theta[1]=thetanow;
            L.u2[1]=u2now;
            L.theta2[1]=theta2now;
            L.X[1]=Xnow;
            L.Z[1]=Znow;
            L.eta[1]=eta_now;
            L.y[1]=y_now;
            L.names[1]=namenow;
        }
        if(lam3size>1&&l==0&&k<lam3size-1){//layer3 
            L.mubeta[2]=betanow;
            L.mualpha[2]=alphanow;
            L.Omegais[2]=Omegaisnow;
            L.u[2]=unow;
            L.theta[2]=thetanow;
            L.u2[2]=u2now;
            L.theta2[2]=theta2now;
            L.X[2]=Xnow;
            L.Z[2]=Znow;
            L.eta[2]=eta_now;
            L.y[2]=y_now;
            L.names[2]=namenow;
        }
    }
}



