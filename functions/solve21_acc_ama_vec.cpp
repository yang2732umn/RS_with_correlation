#include "consider_covariance.h"
MatrixXd solve21_acc_ama_vec(const MatrixXd &y,double gamma, int maxIter, double Tol){// use ADMM to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
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
                    delta.col(i)+=lambda.col(fdi[j]);
                }
            }
            if(i>0){
                sdi=constructsdi(i+1, p);
                for (j=0; j<sdi.size(); ++j) {
                    delta.col(i)-=lambda.col(sdi[j]);
                }
            }
        }
        u=y+delta;
#pragma omp parallel for private(l)
        for(int t1=0;t1<p-1;++t1){
            for(int t2=t1+1;t2<p;++t2){
                l=t1*(p-(t1+1)*0.5)+t2-t1-1;
                VectorXd temp=lambda.col(l)-mu*(u.col(t1)-u.col(t2));
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
                    delta.col(i)+=S.col(fdi[j]);
                }
            }
            if(i>0){
                sdi=constructsdi(i+1, p);
                for (j=0; j<sdi.size(); ++j) {
                    delta.col(i)-=S.col(sdi[j]);
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
                VectorXd temp=S.col(l)-mu*(u.col(t1)-u.col(t2));
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
