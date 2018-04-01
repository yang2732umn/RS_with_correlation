#include "consider_covariance.h"
result LS_Lasso_Cluster_4_fan3_p_admm_2lambda(rated_user_and_item x, MatrixXd movie, MatrixXd users, MatrixXd mualpha1, MatrixXd mubeta1, double lambda1, double lambda2, double rho,double Tol, int maxIter){
    int n=users.rows();
    int p=movie.rows();
    int udim=users.cols();
    int mdim=movie.cols();
    
    MatrixXd mualpha2=mualpha1;
    MatrixXd mubeta2=mubeta1;
    MatrixXd* theta=new MatrixXd;//can also use: MatrixXd* theta2=new MatrixXd, and later use *theta2(same as *theta1)
    MatrixXd* u=new MatrixXd;
    
    double maxdiff=100;
    int iter=0;
    int i,j;
    int l;
    VectorXi fdi(1),sdi(1);
    VectorXd temp,atemp(2),a(2),btemp(2),b(2);
    MatrixXd mtemp(n,n);
    double obj1,obj2,maxalpha,maxbeta,maxu,maxtheta;
    VectorXd change, change2;
    double ttt;
    vector<VectorXd> pretilde(n);
    vector<rated_1itemmore> itemmore=construct_itemmore(x);
    obj1=cal_obj2_2lambda(mualpha1, mubeta1, lambda1,lambda2, x.user, movie, users);
    cout<<"obj="<<obj1<<endl;
    while (maxdiff>2*Tol&iter<maxIter) {
        {
            int beta_iter=0;
            *theta=MatrixXd::Zero(mubeta2.cols(),(n*(n-1)/2));
            *u=*theta;
            while(beta_iter<maxIter){
                maxbeta=0;
                MatrixXd mubeta1_inner=mubeta2;
#pragma omp parallel for private(j,fdi,sdi,temp,atemp,btemp,mtemp)
                for(i=0;i<n;++i){
                    temp=rho*(mubeta1_inner.transpose()).rowwise().sum();
                    atemp=rho*mubeta1_inner.row(i);
                    temp=temp-atemp;
                    if(i<n-1){
                        fdi=VectorXi::LinSpaced(n-i-1,i*(n-(i+1)*0.5),i*(n-(i+1)*0.5)+n-i-2);
                        mtemp.resize(mdim,fdi.size());
                        for (j=0; j<fdi.size(); ++j) {
                            mtemp.col(j)=(*theta).col(fdi[j])-(*u).col(fdi[j]);
                        }
                        temp=temp+rho*mtemp.rowwise().sum();
                    }
                    
                    if(i>0){
                        sdi=constructsdi(i+1, n);
                        mtemp.resize(mdim,sdi.size());
                        for (j=0; j<sdi.size(); ++j) {
                            mtemp.col(j)=(*theta).col(sdi[j])-(*u).col(sdi[j]);
                        }
                        temp=temp-rho*mtemp.rowwise().sum();
                    }
                    MatrixXd alphause=matrix_colsub(mualpha1,x.user[i].item);
                    VectorXd useri=users.row(i);
                    atemp=x.user[i].rating-(alphause.transpose())*useri;
                    MatrixXd movieuse=matrix_rowsub(movie,x.user[i].item);
                    temp=temp+(movieuse.transpose())*atemp;
                    mtemp.resize(mdim,mdim);
                    mtemp=movieuse.transpose()*movieuse;
                    mtemp=mtemp+rho*(n-1)*MatrixXd::Identity(mdim,mdim);
                    mubeta2.row(i)=mtemp.inverse()*temp;
                }
                maxbeta=absm(mubeta2-mubeta1_inner).maxCoeff();
                
                maxtheta=0;
                maxu=0;
                change.resize((*theta).cols());
                change2.resize(change.size());
#pragma omp parallel for private(j,l,temp,atemp,btemp,a,b)
                for (i=0; i<(n-1); ++i) {
                    for (j=(i+1); j<n; ++j) {
                        l=i*(n-(i+1)*0.5)+j-i-1;
                        atemp=mubeta2.row(i)-mubeta2.row(j);
                        btemp=(*u).col(l);
                        atemp=atemp+btemp;
                        a=ST_vec(atemp,lambda1/(2*rho));
                        atemp=a-(*theta).col(l);
                        (*theta).col(l)=a;
                        change[l]=atemp.lpNorm<Infinity>();
                        atemp=mubeta2.row(i)-mubeta2.row(j);
                        atemp=atemp-a;
                        change2[l]=atemp.lpNorm<Infinity>();
                        (*u).col(l)=(*u).col(l)+atemp;
                    }
                }
                maxtheta=change.lpNorm<Infinity>();
                maxu=change2.lpNorm<Infinity>();
                beta_iter+=1;
                ttt=max(max(maxbeta,maxtheta),maxu);
                if(ttt<Tol){
                    break;
                }
            }
            
            {   cout<<"beta_iter is "<<beta_iter<<endl;
                if(ttt>=Tol){
                    cout<<"beta didn't converge! maxdiff is "<<ttt<<endl;
                }
                cout<<"beta done"<<endl;
                obj2=cal_obj2_2lambda(mualpha1, mubeta2, lambda1,lambda2, x.user, movie, users);
                cout<<"obj="<<obj2<<endl;
                if (obj1-obj2<0.001) {
                }
                else{
                    obj1=obj2;
                }
            }
            
            int alpha_iter=0;
            *theta=MatrixXd::Zero(mualpha2.rows(),p*(p-1)/2);
            *u=*theta;
            while(alpha_iter<maxIter){
                maxalpha=0;
                MatrixXd mualpha1_inner=mualpha2;
#pragma omp parallel for private(btemp)
                for (i=0; i<n; ++i) {
                    MatrixXd alphause=matrix_colsub(mualpha1_inner,x.user[i].item);
                    MatrixXd movieuse=matrix_rowsub(movie,x.user[i].item);
                    VectorXd useri=users.row(i);
                    btemp=mubeta2.row(i);
                    pretilde[i]=alphause.transpose()*useri+movieuse*btemp;
                }
                
#pragma omp parallel for private(j,fdi,sdi,temp,atemp,mtemp,btemp)
                for(i=0;i<p;++i){
                    temp=rho*mualpha1_inner.rowwise().sum();
                    atemp=rho*mualpha1_inner.col(i);
                    temp=temp-atemp;
                    if(i<p-1){
                        fdi=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
                        mtemp.resize(udim,fdi.size());
                        for (j=0; j<fdi.size(); ++j) {
                            mtemp.col(j)=(*theta).col(fdi[j])-(*u).col(fdi[j]);
                        }
                        temp=temp+rho*mtemp.rowwise().sum();
                    }
                    
                    if(i>0){
                        sdi=constructsdi(i+1, p);
                        mtemp.resize(udim,sdi.size());
                        for (j=0; j<sdi.size(); ++j) {
                            mtemp.col(j)=(*theta).col(sdi[j])-(*u).col(sdi[j]);
                        }
                        temp=temp-rho*mtemp.rowwise().sum();
                    }
                    
                    for (j=0; j<itemmore[i].user.size(); ++j) {
                        int userno=itemmore[i].user[j];
                        int index=itemmore[i].numberforuser[j];
                        VectorXd tt=(x.user[userno].rating[index]-(movie.row(i)).dot(mubeta2.row(userno)))*users.row(userno);
                        temp=temp+tt;
                    }
                    mtemp=rho*(p-1)*MatrixXd::Identity(udim,udim);
                    for (j=0; j<itemmore[i].user.size(); ++j) {
                        int userno=itemmore[i].user[j];
                        btemp=users.row(userno);
                        mtemp=mtemp+btemp*btemp.transpose();
                    }
                    mualpha2.col(i)=mtemp.inverse()*temp;
                }
                maxalpha=absm(mualpha2-mualpha1_inner).maxCoeff();
                
                maxtheta=0;
                maxu=0;
                change.resize((*theta).cols());
                change2.resize((*theta).cols());
#pragma omp parallel for private(j,l,btemp,b)
                for (i=0; i<(p-1); ++i) {
                    for (j=(i+1); j<p; ++j) {
                        l=i*(p-(i+1)*0.5)+j-i-1;
                        btemp=mualpha2.col(i)-mualpha2.col(j)+(*u).col(l);
                        b=ST_vec(btemp,lambda2/(2*rho));
                        btemp=(*theta).col(l);
                        (*theta).col(l)=b;
                        btemp=btemp-b;
                        change[l]=btemp.lpNorm<Infinity>();
                        btemp=mualpha2.col(i)-mualpha2.col(j)-(*theta).col(l);
                        change2[l]=btemp.lpNorm<Infinity>();
                        (*u).col(l)=(*u).col(l)+btemp;
                    }
                }
                maxtheta=change.lpNorm<Infinity>();
                maxu=change2.lpNorm<Infinity>();
                alpha_iter+=1;
                ttt=max(max(maxalpha,maxtheta),maxu);
                if(ttt<Tol){
                    break;
                }
                
            }
            {   cout<<"alpha_iter is "<<alpha_iter<<endl;
                if(ttt>=Tol){
                    cout<<"alpha didn't converge!maxdiff is "<<ttt<<endl;
                }
                cout<<"alpha done"<<endl;
                obj2=cal_obj2_2lambda(mualpha2, mubeta2,  lambda1,lambda2,  x.user, movie, users);
                cout<<"obj="<<obj2<<endl;
                if (obj1-obj2<0.001) {
                }
                else{
                    obj1=obj2;
                }
            }
            maxdiff=max(absm(mubeta2-mubeta1).maxCoeff(),absm(mualpha2-mualpha1).maxCoeff());//also calculating max in theta2-theta1, gamma2-gamma1 cause memory trouble
            cout<<"maxdiff="<<maxdiff<<endl;
            
            mubeta1=mubeta2;
            mualpha1=mualpha2;
            iter=iter+1;
            cout<<"iter is "<<iter<<endl;
        }
    }
    if (iter==maxIter) {cout<<"maxdiff is "<<maxdiff<<endl;}
    else cout<<"iter is "<<iter<<endl;
    delete theta;delete u;
    
    mtemp.resize(n,p);
    mtemp=users*mualpha2+mubeta2*movie.transpose();
    cout<<"maxdiff is "<<maxdiff<<endl;
    result re;
    re.mualpha=mualpha2;
    re.mubeta=mubeta2;
    re.solu=mtemp;
    re.maxdiff=maxdiff;
    return re;
}//



