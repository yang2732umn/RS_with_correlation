#include "consider_covariance.h"
result LS_Lasso_Cluster_4_fan3_p_admm_std(rated_user_and_item x, MatrixXd movie, MatrixXd users, MatrixXd mualpha1, MatrixXd mubeta1, double lambda1, double rho,double Tol, int maxIter){
    //fix Omega at I
    //the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood,
    //also deal with if parallel computing of alpha doesn't work
    
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
    
    double maxdiff=100;
    int iter=0;
    int i,j;
    int l;
    VectorXi fdi(1),sdi(1);
    VectorXd temp,atemp(2),a(2),btemp(2),b(2);
    MatrixXd mtemp(n,n);
    //cout<<"done here "<<endl;
    double obj1,obj2,maxalpha,maxbeta,maxu,maxtheta;
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
        mtempB[i]=mtempB[i]+rho_beta*(n-1)*MatrixXd::Identity(mdim,mdim);
        mtempB[i]=mtempB[i].inverse();
    }
    vector<MatrixXd> mtempA(p);
    vector<VectorXd> tempA(p);
    vector<VectorXd> tempA0(p);
#pragma omp parallel for private(j,btemp)
    for(i=0;i<p;++i){
        mtempA[i]=rho_alpha*(p-1)*MatrixXd::Identity(udim,udim);
        MatrixXd mtemp2(udim,itemmore[i].user.size());
        for (j=0; j<itemmore[i].user.size(); ++j) {
            int userno=itemmore[i].user[j];
            int index=itemmore[i].numberforuser[j];
            mtemp2.col(j)=users.row(userno);
        }
        mtempA[i]=mtempA[i]+mtemp2*mtemp2.transpose();
        mtempA[i]= mtempA[i].inverse();
    }
    
    
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
    
    
    while (iter<maxIter) {
        //update mubeta
        int beta_iter=0;
        if (iter==0) {
            *theta=MatrixXd::Zero(mubeta2.cols(),(n*(n-1)/2));
            *u=*theta;
        }
#pragma omp parallel for
        for(i=0;i<n;++i){
            MatrixXd alphause=matrix_colsub(mualpha1,x.user[i].item);
            VectorXd useri=users.row(i);
            tempB0[i]=x.user[i].rating-(alphause.transpose())*useri;
            tempB0[i]=(movieuse[i].transpose())*tempB0[i];
        }
        while(beta_iter<maxIter){
            maxbeta=0;
            MatrixXd mubeta1_inner=mubeta2;
            double maxbeta_inner=0;
            int beta_iter_inner=0;
            tempB=tempB0;
#pragma omp parallel for private(j,fdi,sdi,mtemp)
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
            }
            while (beta_iter_inner<maxIter) {
                MatrixXd mubeta1_inner2=mubeta2;
#pragma omp parallel for private(temp,atemp)
                for(i=0;i<n;++i){
                    temp=(mubeta1_inner2.transpose()).rowwise().sum();
                    atemp=mubeta1_inner2.row(i);
                    temp=rho_beta*(temp-atemp);
                    temp+=tempB[i];
                    mubeta2.row(i)=mtempB[i]*temp;
                }
                maxbeta_inner=absm(mubeta2-mubeta1_inner2).maxCoeff();
                ++beta_iter_inner;
                if (maxbeta_inner<Tol) {
                    break;
                }
            }
            //cout<<"beta_iter_inner="<<beta_iter_inner<<endl;
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
            if (obj1-obj2<0.001) {
                //break;
            }
            else{
                obj1=obj2;
            }
        }
        
        //update mualpha
        int alpha_iter=0;
        if (iter==0) {
            *theta2=MatrixXd::Zero(mualpha2.rows(),p*(p-1)/2);
            *u2=*theta2;
        }
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
            MatrixXd mualpha1_inner=mualpha2;
            double maxalpha_inner=0;
            int alpha_iter_inner=0;
            tempA=tempA0;
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
            }
            
            while (alpha_iter_inner<maxIter) {
                MatrixXd mualpha1_inner2=mualpha2;
                /*#pragma omp parallel for private(temp,atemp)
                 for(i=0;i<p;++i){
                 temp=mualpha1_inner2.rowwise().sum();
                 atemp=mualpha1_inner2.col(i);
                 temp=rho_alpha*(temp-atemp);
                 temp+=tempA[i];
                 mualpha2.col(i)=mtempA[i]*temp;
                 }*/
#pragma omp parallel for private(temp,atemp)
                for(i=0;i<p;++i){
                    temp=mualpha2.rowwise().sum();
                    atemp=mualpha2.col(i);
                    temp=rho_alpha*(temp-atemp);
                    temp+=tempA[i];
                    mualpha2.col(i)=mtempA[i]*temp;
                }
                maxalpha_inner=absm(mualpha2-mualpha1_inner2).maxCoeff();
                //cout<<"maxalpha_inner="<<maxalpha_inner<<endl;
                ++alpha_iter_inner;
                if (maxalpha_inner<Tol/*||maxalpha_inner>1*/) {
                    //cout<<"most inner for alpha used "<<alpha_iter_inner<<" iterations"<<endl;
                    //cout<<"maxalpha_inner="<<maxalpha_inner<<endl;
                    break;
                }
            }
            
            //cout<<"alpha_iter_inner="<<alpha_iter_inner<<endl;
            maxalpha=absm(mualpha2-mualpha1_inner).maxCoeff();
            
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
                    // cout<<"theta kk "<<kk<<" done"<<endl;
                }
            }
            maxtheta=change.lpNorm<Infinity>();
            maxu=change2.lpNorm<Infinity>();
            alpha_iter+=1;
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
            if (abs(obj1-obj2)<1e-6&&iter>620) {//
                break;
            }
            else{
                obj1=obj2;
            }
        }
        
        maxdiff=max(absm(mubeta2-mubeta1).maxCoeff(),absm(mualpha2-mualpha1).maxCoeff());//also calculating max in theta2-theta1, gamma2-gamma1 cause memory trouble
        cout<<"maxdiff="<<maxdiff<<endl;//
        
        mubeta1=mubeta2;
        mualpha1=mualpha2;
        iter=iter+1;
        cout<<"iter is "<<iter<<endl;
        
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
    re.obj=obj2;
    return re;
}//this is novariance

