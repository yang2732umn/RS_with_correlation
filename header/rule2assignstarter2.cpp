#include "consider_covariance.h"

//designed for lambda1!=lambda3, also works for lambda1==lambda3
void rule2assignstarter2(int nlayer, int i,int j,int k,int l,MatrixXd &betastart,MatrixXd &alphastart, MatrixXd& ustart,MatrixXd& u2start, MatrixXd& thetastart,MatrixXd& theta2start, vector<MatrixXd>& Omegaisstart, vector<MatrixXd>& Zisstart,vector<MatrixXd>& Uisstart,const layers2 &L){
    //if i,j,k,l, nlayer=3; if i,j,k, nlayer=2
    if (nlayer==2) {
        if (i>0&&j==0&&k==0) {
            betastart=L.mubeta[0];
            alphastart=L.mualpha[0];
            Omegaisstart=L.Omegais[0];
            thetastart=L.theta[0];
            theta2start=L.theta2[0];
            ustart=L.u[0];
            u2start=L.u2[0];
            Zisstart=L.Zis[0];
            Uisstart=L.Uis[0];
            cout<<"Using "<<L.names[0]<<" as start."<<endl;
        }
        if (j>0&&k==0) {
            betastart=L.mubeta[1];
            alphastart=L.mualpha[1];
            Omegaisstart=L.Omegais[1];
            thetastart=L.theta[1];
            theta2start=L.theta2[1];
            ustart=L.u[1];
            u2start=L.u2[1];
            Zisstart=L.Zis[1];
            Uisstart=L.Uis[1];
            cout<<"Using "<<L.names[1]<<" as start."<<endl;
        }
    }
    if (nlayer==3) {
        if (i>0&&j==0&&k==0&&l==0) {
            betastart=L.mubeta[0];
            cout<<"here"<<endl;
            cout<<"L.mubeta[0].row(0)="<<endl<<L.mubeta[0].row(0)<<endl;
            alphastart=L.mualpha[0];
            Omegaisstart=L.Omegais[0];
            thetastart=L.theta[0];
            theta2start=L.theta2[0];
            ustart=L.u[0];
            u2start=L.u2[0];
            Zisstart=L.Zis[0];
            Uisstart=L.Uis[0];
            cout<<"Using "<<L.names[0]<<" as start."<<endl;
        }
        if (j>0&&k==0&&l==0) {
            betastart=L.mubeta[1];
            alphastart=L.mualpha[1];
            Omegaisstart=L.Omegais[1];
            thetastart=L.theta[1];
            theta2start=L.theta2[1];
            ustart=L.u[1];
            u2start=L.u2[1];
            Zisstart=L.Zis[1];
            Uisstart=L.Uis[1];
            cout<<"Using "<<L.names[1]<<" as start."<<endl;
        }
        if (k>0&&l==0) {
            betastart=L.mubeta[2];
            alphastart=L.mualpha[2];
            Omegaisstart=L.Omegais[2];
            thetastart=L.theta[2];
            theta2start=L.theta2[2];
            ustart=L.u[2];
            u2start=L.u2[2];
            Zisstart=L.Zis[2];
            Uisstart=L.Uis[2];
            cout<<"Using "<<L.names[2]<<" as start."<<endl;
        }
    }
    
}

