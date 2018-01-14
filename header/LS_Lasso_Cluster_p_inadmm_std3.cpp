#include "consider_covariance.h"
result LS_Lasso_Cluster_p_inadmm_std3(const rated_user_and_item & x, const MatrixXd & movie, const MatrixXd & users, MatrixXd & mualpha1, MatrixXd& mubeta1, double & lambda1, double & rho,double  Tol, int maxIter){
    //fix Omega at I
    //the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood
    //also deal with if parallel computing of alpha doesn't work
    //use inexact admm
    //first use r=0.8, after converge, use r=1
    int n=users.rows();
    int p=movie.rows();
    int udim=users.cols();
    int mdim=movie.cols();
    //cout<<"done here "<<endl;
    
    MatrixXd mualpha2=mualpha1;
    MatrixXd mubeta2=mubeta1;
    MatrixXd* theta=new MatrixXd;
    MatrixXd* u=new MatrixXd;
    MatrixXd* theta2=new MatrixXd;
    MatrixXd* u2=new MatrixXd;
    
    *theta=MatrixXd::Zero(mubeta2.cols(),(n*(n-1)/2));
    *u=*theta;
    *theta2=MatrixXd::Zero(mualpha2.rows(),p*(p-1)/2);
    *u2=*theta2;
    
    double rt=1-0.2;//Does it always work? For a different n and p? How to control the part using r=0.8? Now use iter 10次后没有下降，停止;
    //rt 不是越小越快
    double maxdiff=100;
    int iter=0;
    int i,j;
    int l;
    VectorXi fdi(1),sdi(1);
    VectorXd temp,atemp(2),a(2),btemp(2),b(2);
    MatrixXd mtemp(n,n);
    //cout<<"done here "<<endl;
    double obj1,obj2,maxalpha,maxbeta,maxu,maxtheta,objt0,objt10;
    VectorXd change, change2;
    double ttt;
    vector<VectorXd> pretilde(n);
    vector<rated_1itemmore> itemmore=construct_itemmore(x);
    
    int c1=0;
    int c4=n*(n-1)/2*mdim;
    int c5=p*(p-1)/2*udim;
    for (int kk=0; kk<n; ++kk) {
        int kkk=x.user[kk].item.size();
        c1+=kkk;
    }
    two_lambda_struct A;
    A.lambda1=c1*lambda1/c4;
    A.lambda2=c1*lambda1/c5;
    A.mualpha=mualpha1;
    A.mubeta=mubeta1;
    A.movie=movie;
    A.users=users;
    A.user=x.user;
    obj1=cal_obj2_2lambda_struct(A)/c1*100;
    cout<<"obj="<<obj1<<endl;
    time_t tstart, tend;
    
    double lambda_beta=c1*lambda1/c4;
    double lambda_alpha=c1*lambda1/c5;
    double rho_alpha=c1*rho/c5;
    double rho_beta=c1*rho/c4;
    //construct mtemp and tempB for each user
    vector<MatrixXd> mtempB(n);
    vector<VectorXd> tempB(n);
    vector<VectorXd> tempB0(n);
    MatrixXd* movieuse=new MatrixXd[n];
#pragma omp parallel for
    for(i=0;i<n;++i){
        movieuse[i]=matrix_rowsub(movie,x.user[i].item);
        mtempB[i]=movieuse[i].transpose()*movieuse[i];
        mtempB[i]=mtempB[i]+rt*rho_beta*n*MatrixXd::Identity(mdim,mdim);
        mtempB[i]=mtempB[i].inverse();
    }
    
    vector<MatrixXd> mtempA(p);
    vector<VectorXd> tempA(p);
    vector<VectorXd> tempA0(p);
#pragma omp parallel for private(j,btemp)
    for(i=0;i<p;++i){
        mtempA[i]=rt*rho_alpha*p*MatrixXd::Identity(udim,udim);
        MatrixXd mtemp2(udim,itemmore[i].user.size());
        for (j=0; j<itemmore[i].user.size(); ++j) {
            int userno=itemmore[i].user[j];
            int index=itemmore[i].numberforuser[j];
            mtemp2.col(j)=users.row(userno);
        }
        mtempA[i]=mtempA[i]+mtemp2*mtemp2.transpose();
        mtempA[i]= mtempA[i].inverse();
    }
    VectorXd betasum,alphasum;
    MatrixXd alphapre,betapre,alphause;
    VectorXd useri;
    
    vector<VectorXi> fdibeta(n-1),sdibeta(n-1),fdialpha(p-1),sdialpha(p-1);
#pragma omp parallel for
    for (i=0; i<n; ++i) {
        if(i<n-1){
            fdibeta[i]=VectorXi::LinSpaced(n-i-1,i*(n-(i+1)*0.5),i*(n-(i+1)*0.5)+n-i-2);
        }
        if(i>0){
            sdibeta[i-1]=constructsdi(i+1, n);
        }
    }
#pragma omp parallel for
    for (i=0; i<p; ++i) {
        if(i<p-1){
            fdialpha[i]=VectorXi::LinSpaced(p-i-1,i*(p-(i+1)*0.5),i*(p-(i+1)*0.5)+p-i-2);
        }
        if(i>0){
            sdialpha[i-1]=constructsdi(i+1, p);
        }
    }
    
    tri trin,trip;
    trin.first.resize(n*(n-1)/2);trin.second.resize(n*(n-1)/2);
    trip.first.resize(p*(p-1)/2);trip.second.resize(p*(p-1)/2);
    l=0;
    for (int i=0; i<n-1; ++i) {
        for (int j=i+1; j<n; ++j) {
            trin.first(l)=i;trin.second(l)=j;
            ++l;
        }
    }
    l=0;
    for (int i=0; i<p-1; ++i) {
        for (int j=i+1; j<p; ++j) {
            trip.first(l)=i;trip.second(l)=j;
            ++l;
        }
    }
    
    
    objt0=obj1;
    
    tstart=time(0);
    while (iter<maxIter) {
        //update mubeta
        int beta_iter=0;
#pragma omp parallel for private(alphause,useri)//working
        for(i=0;i<n;++i){
            alphause=matrix_colsub(mualpha1,x.user[i].item);
            useri=users.row(i);
            tempB0[i]=x.user[i].rating-(alphause.transpose())*useri;
            tempB0[i]=(movieuse[i].transpose())*tempB0[i];
        }
        while(beta_iter<maxIter){
            betapre=mubeta2;
            maxbeta=0;
            tempB=tempB0;
            betasum=mubeta2.colwise().sum();
#pragma omp parallel for private(j,fdi,sdi,mtemp,atemp)//working
            for(i=0;i<n;++i){
                if(i<n-1){
                    fdi=fdibeta[i];
                    mtemp.resize(mdim,fdi.size());
                    for (j=0; j<fdi.size(); ++j) {
                        mtemp.col(j)=(*theta).col(fdi[j])-(*u).col(fdi[j]);
                    }
                    tempB[i]=tempB[i]+rho_beta*mtemp.rowwise().sum();
                }
                if(i>0){
                    sdi=sdibeta[i-1];
                    mtemp.resize(mdim,sdi.size());
                    for (j=0; j<sdi.size(); ++j) {
                        mtemp.col(j)=(*theta).col(sdi[j])-(*u).col(sdi[j]);
                    }
                    tempB[i]=tempB[i]-rho_beta*mtemp.rowwise().sum();
                }
                tempB[i]=tempB[i]+rho_beta*betasum;
                atemp=(rt-1)*(2*n)*rho_beta*mubeta2.row(i);
                tempB[i]=tempB[i]+atemp;
                mubeta2.row(i)=mtempB[i]*tempB[i];
            }
            maxbeta=absm(mubeta2-betapre).maxCoeff();
            maxtheta=0;
            maxu=0;
            change.resize((*theta).cols());
            change2.resize(change.size());
            /*#pragma omp parallel for private(j,l,temp,atemp,btemp,a,b)//working
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
             }*/
            
#pragma omp parallel for private(i,j,temp,atemp,btemp,a,b)//working, all are not 8 times or even 2 times as fast.
            for (l=0; l<n*(n-1)/2; ++l){
                i=trin.first(l);
                j=trin.second(l);
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
            
            maxtheta=change.lpNorm<Infinity>();
            maxu=change2.lpNorm<Infinity>();
            beta_iter=beta_iter+1;
            ttt=max(max(maxbeta,maxtheta),maxu);
            cout<<"beta_iter is "<<beta_iter<<", maxbeta="<<maxbeta<<", maxu="<<maxu<<", maxtheta="<<maxtheta<<endl;
            if(ttt<2*Tol){
                break;
            }
            if (iter<300&&beta_iter>10) {//beta_iter previously >400
                break;
            }
            
        }
        
        {   cout<<"beta_iter is "<<beta_iter<<endl;
            if(ttt>=Tol){
                cout<<"beta didn't converge! maxdiff is "<<ttt<<endl;
            }
            cout<<"beta done"<<endl;
            A.mubeta=mubeta2;
            obj2=cal_obj2_2lambda_struct(A)/c1*100;
            cout<<"obj="<<obj2<<endl;
            obj1=obj2;
            
        }
        
        //update mualpha
        int alpha_iter=0;
#pragma omp parallel for private(j,mtemp)//working
        for(i=0;i<p;++i){
            mtemp=MatrixXd::Zero(udim,itemmore[i].user.size());
            for (j=0; j<itemmore[i].user.size(); ++j) {
                int userno=itemmore[i].user[j];
                int index=itemmore[i].numberforuser[j];
                double ttt=x.user[userno].rating[index]-(movie.row(i)).dot(mubeta2.row(userno));
                //cout<<"x.user[i].rating.size() is "<<x.user[i].rating.size()<<endl;
                mtemp.col(j)=ttt*users.row(userno);
            }
            tempA0[i]=mtemp.rowwise().sum();
        }
        
        while(alpha_iter<maxIter){
            maxalpha=0;
            alphapre=mualpha2;
            tempA=tempA0;
            alphasum=mualpha2.rowwise().sum();
            //#pragma omp parallel for private(fdi,mtemp,j,sdi)
            //working? No. p too small=30, unparallel seems better. Still unclear why, above two parallels also are on p, but is a little faster
            for(i=0;i<p;++i){
                if(i<p-1){
                    fdi=fdialpha[i];
                    mtemp.resize(udim,fdi.size());
                    for (j=0; j<fdi.size(); ++j) {
                        int tt=fdi[j];
                        mtemp.col(j)=(*theta2).col(tt)-(*u2).col(tt);
                    }
                    tempA[i]=tempA[i]+rho_alpha*mtemp.rowwise().sum();
                }
                if(i>0){
                    sdi=sdialpha[i-1];
                    mtemp.resize(udim,sdi.size());
                    for (j=0; j<sdi.size(); ++j) {
                        int tt=sdi[j];
                        mtemp.col(j)=(*theta2).col(tt)-(*u2).col(tt);
                    }
                    tempA[i]=tempA[i]-rho_alpha*mtemp.rowwise().sum();
                }
                tempA[i]=tempA[i]+rho_alpha*alphasum;
                tempA[i]=tempA[i]+(rt-1)*(2*p)*rho_alpha*mualpha2.col(i);
                mualpha2.col(i)=mtempA[i]*tempA[i];
            }
            maxalpha=absm(mualpha2-alphapre).maxCoeff();
            
            maxtheta=0;
            maxu=0;
            change.resize((*theta2).cols());
            change2.resize((*theta2).cols());
            /*#pragma omp parallel for private(j,l,btemp,b)//working
             for (i=0; i<(p-1); ++i) {
             for (j=(i+1); j<p; ++j) {
             l=i*(p-(i+1)*0.5)+j-i-1;
             btemp=mualpha2.col(i)-mualpha2.col(j)+(*u2).col(l);
             b=ST_vec(btemp,lambda1/(2*rho));
             btemp=(*theta2).col(l);
             (*theta2).col(l)=b;
             btemp=btemp-b;
             change[l]=btemp.lpNorm<Infinity>();
             btemp=mualpha2.col(i)-mualpha2.col(j)-(*theta2).col(l);
             change2[l]=btemp.lpNorm<Infinity>();
             (*u2).col(l)=(*u2).col(l)+btemp;
             //cout<<"theta kk "<<kk<<" done"<<endl;
             }
             }*/
            
#pragma omp parallel for private(i,j,btemp,b)//working, but worse than not directly parallizing l
            for (l=0; l<p*(p-1)/2; ++l){
                i=trip.first(l);
                j=trip.second(l);
                btemp=mualpha2.col(i)-mualpha2.col(j)+(*u2).col(l);
                b=ST_vec(btemp,lambda1/(2*rho));
                btemp=(*theta2).col(l);
                (*theta2).col(l)=b;
                btemp=btemp-b;
                change[l]=btemp.lpNorm<Infinity>();
                btemp=mualpha2.col(i)-mualpha2.col(j)-(*theta2).col(l);
                change2[l]=btemp.lpNorm<Infinity>();
                (*u2).col(l)=(*u2).col(l)+btemp;
            }
            
            
            maxtheta=change.lpNorm<Infinity>();
            maxu=change2.lpNorm<Infinity>();
            alpha_iter=alpha_iter+1;
            ttt=max(max(maxalpha,maxtheta),maxu);
            //cout<<"alpha maxdiff is "<<ttt<<endl;
            if(ttt<2*Tol/*||ttt>100*/){//also changed here from parallel to unparallel
                break;
            }
            if (iter<300&&alpha_iter>10) {//alpha_iter previously >400
                break;
            }
            
        }
        
        {   cout<<"alpha_iter is "<<alpha_iter<<endl;
            if(ttt>=Tol){
                cout<<"alpha didn't converge!maxdiff is "<<ttt<<endl;
            }
            cout<<"alpha done"<<endl;
            A.mualpha=mualpha2;
            obj2=cal_obj2_2lambda_struct(A)/c1*100;
            cout<<"obj="<<obj2<<endl;
            if (iter%10==0) {
                objt10=obj2;
                if(objt10>objt0) {
                    cout<<"Break here, iter="<<iter<<endl;//if rt=0.8 increases obj constantly, stop
                    break;
                }
                objt0=objt10;
            }
            if (abs(obj1-obj2)<1e-6&&iter>620) {//fast, so use high precision
                break;
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
    tend=time(0);
    
    cout<<"First part done. It took "<<difftime(tend, tstart)<<" s."<<endl;  
    
    //rt=1;
#pragma omp parallel for
    for(i=0;i<n;++i){
        movieuse[i]=matrix_rowsub(movie,x.user[i].item);
        mtempB[i]=movieuse[i].transpose()*movieuse[i];
        mtempB[i]=mtempB[i]+rho_beta*n*MatrixXd::Identity(mdim,mdim);//correctisized
        mtempB[i]=mtempB[i].inverse();
    }
#pragma omp parallel for private(j,btemp)
    for(i=0;i<p;++i){
        mtempA[i]=rho_alpha*p*MatrixXd::Identity(udim,udim);
        MatrixXd mtemp2(udim,itemmore[i].user.size());
        for (j=0; j<itemmore[i].user.size(); ++j) {
            int userno=itemmore[i].user[j];
            int index=itemmore[i].numberforuser[j];
            mtemp2.col(j)=users.row(userno);
        }
        mtempA[i]=mtempA[i]+mtemp2*mtemp2.transpose();
        mtempA[i]= mtempA[i].inverse();
    }
    tstart=time(0);
    while (iter<maxIter) {
        //update mubeta
        int beta_iter=0;
#pragma omp parallel for private(alphause,useri)
        for(i=0;i<n;++i){
            alphause=matrix_colsub(mualpha1,x.user[i].item);
            useri=users.row(i);
            tempB0[i]=x.user[i].rating-(alphause.transpose())*useri;
            tempB0[i]=(movieuse[i].transpose())*tempB0[i];
        }
        while(beta_iter<maxIter){
            betapre=mubeta2;
            maxbeta=0;
            tempB=tempB0;
            betasum=mubeta2.colwise().sum();
#pragma omp parallel for private(j,fdi,sdi,mtemp,atemp)
            for(i=0;i<n;++i){
                if(i<n-1){
                    fdi=fdibeta[i];
                    mtemp.resize(mdim,fdi.size());
                    for (j=0; j<fdi.size(); ++j) {
                        mtemp.col(j)=(*theta).col(fdi[j])-(*u).col(fdi[j]);
                    }
                    tempB[i]=tempB[i]+rho_beta*mtemp.rowwise().sum();
                }
                if(i>0){
                    sdi=sdibeta[i-1];
                    mtemp.resize(mdim,sdi.size());
                    for (j=0; j<sdi.size(); ++j) {
                        mtemp.col(j)=(*theta).col(sdi[j])-(*u).col(sdi[j]);
                    }
                    tempB[i]=tempB[i]-rho_beta*mtemp.rowwise().sum();
                }
                tempB[i]=tempB[i]+rho_beta*betasum;
                mubeta2.row(i)=mtempB[i]*tempB[i];
            }
            maxbeta=absm(mubeta2-betapre).maxCoeff();
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
            beta_iter=beta_iter+1;
            ttt=max(max(maxbeta,maxtheta),maxu);
            //cout<<"beta_iter is "<<beta_iter<<endl;
            //cout<<"maxbeta="<<maxbeta<<", maxu="<<maxu<<", maxtheta="<<maxtheta<<endl;
            if(ttt<2*Tol){
                break;
            }
            if (iter<300&&beta_iter>400) {
                break;
            }
            
        }
        
        {   cout<<"beta_iter is "<<beta_iter<<endl;
            if(ttt>=Tol){
                cout<<"beta didn't converge! maxdiff is "<<ttt<<endl;
            }
            cout<<"beta done"<<endl;
            A.mubeta=mubeta2;
            obj2=cal_obj2_2lambda_struct(A)/c1*100;
            cout<<"obj="<<obj2<<endl;
            /*if (obj1-obj2<0.001) {
             //break;
             }
             else{
             obj1=obj2;
             }*/
        }
        
        //update mualpha
        int alpha_iter=0;
#pragma omp parallel for private(j,mtemp)
        for(i=0;i<p;++i){
            mtemp=MatrixXd::Zero(udim,itemmore[i].user.size());
            for (j=0; j<itemmore[i].user.size(); ++j) {
                int userno=itemmore[i].user[j];
                int index=itemmore[i].numberforuser[j];
                double ttt=x.user[userno].rating[index]-(movie.row(i)).dot(mubeta2.row(userno));
                //cout<<"x.user[i].rating.size() is "<<x.user[i].rating.size()<<endl;
                mtemp.col(j)=ttt*users.row(userno);
            }
            tempA0[i]=mtemp.rowwise().sum();
        }
        
        while(alpha_iter<maxIter){
            maxalpha=0;
            alphapre=mualpha2;
            tempA=tempA0;
            alphasum=mualpha2.rowwise().sum();
#pragma omp parallel for private(fdi,mtemp,j,sdi)
            for(i=0;i<p;++i){
                if(i<p-1){
                    fdi=fdialpha[i];
                    mtemp.resize(udim,fdi.size());
                    for (j=0; j<fdi.size(); ++j) {
                        int tt=fdi[j];
                        mtemp.col(j)=(*theta2).col(tt)-(*u2).col(tt);
                    }
                    tempA[i]=tempA[i]+rho_alpha*mtemp.rowwise().sum();
                }
                if(i>0){
                    sdi=sdialpha[i-1];
                    mtemp.resize(udim,sdi.size());
                    for (j=0; j<sdi.size(); ++j) {
                        int tt=sdi[j];
                        mtemp.col(j)=(*theta2).col(tt)-(*u2).col(tt);
                    }
                    tempA[i]=tempA[i]-rho_alpha*mtemp.rowwise().sum();
                }
                tempA[i]=tempA[i]+rho_alpha*alphasum;
                mualpha2.col(i)=mtempA[i]*tempA[i];
            }
            maxalpha=absm(mualpha2-alphapre).maxCoeff();
            
            maxtheta=0;
            maxu=0;
            change.resize((*theta2).cols());
            change2.resize((*theta2).cols());
#pragma omp parallel for private(j,l,btemp,b)
            for (i=0; i<(p-1); ++i) {
                for (j=(i+1); j<p; ++j) {
                    l=i*(p-(i+1)*0.5)+j-i-1;
                    btemp=mualpha2.col(i)-mualpha2.col(j)+(*u2).col(l);
                    b=ST_vec(btemp,lambda1/(2*rho));
                    btemp=(*theta2).col(l);
                    (*theta2).col(l)=b;
                    btemp=btemp-b;
                    change[l]=btemp.lpNorm<Infinity>();
                    btemp=mualpha2.col(i)-mualpha2.col(j)-(*theta2).col(l);
                    change2[l]=btemp.lpNorm<Infinity>();
                    (*u2).col(l)=(*u2).col(l)+btemp;
                    //cout<<"theta kk "<<kk<<" done"<<endl;
                }
            }
            maxtheta=change.lpNorm<Infinity>();
            maxu=change2.lpNorm<Infinity>();
            alpha_iter=alpha_iter+1;
            ttt=max(max(maxalpha,maxtheta),maxu);
            //cout<<"alpha maxdiff is "<<ttt<<endl;
            if(ttt<2*Tol/*||ttt>100*/){//also changed here from parallel to unparallel
                break;
            }
            if (iter<300&&alpha_iter>400) {
                break;
            }
            
        }
        
        {   cout<<"alpha_iter is "<<alpha_iter<<endl;
            if(ttt>=Tol){
                cout<<"alpha didn't converge!maxdiff is "<<ttt<<endl;
            }
            cout<<"alpha done"<<endl;
            A.mualpha=mualpha2;
            obj2=cal_obj2_2lambda_struct(A)/c1*100;
            cout<<"obj="<<obj2<<endl;
            if (abs(obj1-obj2)<1e-5&&iter>620) {//used 1e-6 before, change this part from 1e-6 to 1e-5 to 1e-4, 1e-3 is too large, obj changes from 4.98 to 5.50
                break;
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
    tend=time(0);
    cout<<"Second part done. It took "<<difftime(tend, tstart)<<" s."<<endl;
    
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
    re.obj=obj2;
    return re;
}//this is novariance 
