#include "consider_covariance.h"

//designed for lambda1!=lambda3, also works for lambda1==lambda3
void rule2assignstarter3(int nlayer, int i,int j,int k,int l,MatrixXd &betastart,MatrixXd &alphastart, MatrixXd& ustart,MatrixXd& u2start, MatrixXd& thetastart,MatrixXd& theta2start, vector<MatrixXd>& Omegaisstart, vector<vector<double>>& Xstart,vector<vector<double>>& Zstart,vector<vector<double>>& eta_start,vector<vector<double>>& y_start,const layers3 &L){
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
            Xstart=L.X[0];
            Zstart=L.Z[0];
            eta_start=L.eta[0];
            y_start=L.y[0];
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
            Xstart=L.X[1];
            Zstart=L.Z[1];
            eta_start=L.eta[1];
            y_start=L.y[1];
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
            Xstart=L.X[0];
            Zstart=L.Z[0];
            eta_start=L.eta[0];
            y_start=L.y[0];
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
            Xstart=L.X[1];
            Zstart=L.Z[1];
            eta_start=L.eta[1];
            y_start=L.y[1];
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
            Xstart=L.X[2];
            Zstart=L.Z[2];
            eta_start=L.eta[2];
            y_start=L.y[2];
            cout<<"Using "<<L.names[2]<<" as start."<<endl;
        }
    }
    
}

