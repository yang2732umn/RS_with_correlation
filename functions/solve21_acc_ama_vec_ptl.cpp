#include "consider_covariance.h"
MatrixXd solve21_acc_ama_vec_ptl(const MatrixXd &y,double gamma, int maxIter, double Tol, const vector<vector<int>> diffB){
    int iter=0;
    int p=y.cols();
    int dim=y.rows();
    double maxdiff,mu=1.0/p;//mu numerator must be 1.0 rather than 1, otherwise mu=0
    MatrixXd lambda_old=MatrixXd::Zero(dim,p*(p-1)/2);
    MatrixXd lambda=lambda_old;
    MatrixXd S=lambda;
    int l,j;
    VectorXi fdi,sdi;
    MatrixXd delta;
    MatrixXd u=MatrixXd::Zero(dim,p);
    MatrixXd up;
    while (iter<2) {
        lambda_old=lambda;
        delta=MatrixXd::Zero(dim,p);
#pragma omp parallel for private(fdi,sdi,j)
        for (int i=0; i<p; ++i) {
            if(i<p-1){
                fdi=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
                for (j=0; j<fdi.size(); ++j) {
                    VectorXd temp=VectorXd::Zero(dim);
                    for (int tt=0; tt<diffB[fdi[j]].size(); ++tt) {
                        int index=diffB[fdi[j]][tt];
                        temp[index]=lambda(index,fdi[j]);
                    }
                    delta.col(i)+=temp;
                }
            }
            if(i>0){
                sdi=constructsdi(i+1, p);
                for (j=0; j<sdi.size(); ++j) {
                    VectorXd temp=VectorXd::Zero(dim);
                    for (int tt=0; tt<diffB[sdi[j]].size(); ++tt) {
                        int index=diffB[sdi[j]][tt];
                        temp[index]=lambda(index,sdi[j]);
                    }
                    delta.col(i)-=temp;
                }
            }
        }
        u=y+delta;
#pragma omp parallel for private(l)
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                VectorXd temp0=mu*(u.col(t1)-u.col(t2));
                VectorXd temp=VectorXd::Zero(dim);
                for (int tt=0; tt<diffB[l].size(); ++tt) {
                    int index=diffB[l][tt];
                    temp[index]=lambda(index,l)-temp0[index];
                }
                proj_linf(temp,gamma);
                lambda.col(l)=temp;
            }
        }
        ++iter;
    }
    while(iter<maxIter){
        S=lambda + ((double)(iter-1))/((double)(iter+2))*(lambda- lambda_old);
        delta=MatrixXd::Zero(dim,p);
#pragma omp parallel for private(fdi,sdi,j)
        for (int i=0; i<p; ++i) {
            if(i<p-1){
                fdi=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
                for (j=0; j<fdi.size(); ++j) {
                    VectorXd temp=VectorXd::Zero(dim);
                    for (int tt=0; tt<diffB[fdi[j]].size(); ++tt) {
                        int index=diffB[fdi[j]][tt];
                        temp[index]=S(index,fdi[j]);
                    }
                    delta.col(i)+=temp;
                }
            }
            if(i>0){
                sdi=constructsdi(i+1, p);
                for (j=0; j<sdi.size(); ++j) {
                    VectorXd temp=VectorXd::Zero(dim);
                    for (int tt=0; tt<diffB[sdi[j]].size(); ++tt) {
                        int index=diffB[sdi[j]][tt];
                        temp[index]=S(index,sdi[j]);
                    }
                    delta.col(i)-=temp;
                }
            }
        }
        up=u;
        u=y+delta;
        lambda_old=lambda;
#pragma omp parallel for private(l)
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                VectorXd temp0=mu*(u.col(t1)-u.col(t2));
                VectorXd temp=VectorXd::Zero(dim);
                for (int tt=0; tt<diffB[l].size(); ++tt) {
                    int index=diffB[l][tt];
                    temp[index]=S(index,l)-temp0[index];
                }
                proj_linf(temp,gamma);
                lambda.col(l)=temp;
            }
        }
        ++iter;
        maxdiff=(absm(up-u)).maxCoeff();//(objF-objD)/(1+0.5*(objF+objD));
        if(maxdiff<Tol) break;
    }
    if(maxdiff>Tol){
        cout<<"solve21 acc ama not converged, maxdiff is "<<maxdiff<<endl;
        cout<<"problem size is "<<p<<endl;
        cout<<"y="<<y.transpose()<<endl;
    }
    return u;
}
