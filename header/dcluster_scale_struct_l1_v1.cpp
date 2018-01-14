#include "consider_covariance.h"
dresult dcluster_scale_struct_l1_v1(const int outer_first, dcluster_var_l1_para &C)
//only estimates different wi=1/sigma_i^2, use double clustering on alpha and beta, L1 penalty
//for alpha,beta, ADMM is multiblock(convex), linear convergence
//for wi, ADMM is 2 block
//double clustering is only for part of alpha and beta, last udim2 and mdim2 elements
{
    
    double Omega_eigen_time=0;
    int n=C.users.cols();
    int p=C.movie.cols();
    int udim=C.users.rows();
    int mdim=C.movie.rows();
    int udim2=C.udim2;
    int mdim2=C.mdim2;
    //cout<<"done here "<<endl;
    int c1=0;
    int c3=n*(n-1)/2;
    int c4=c3*mdim;//c4=n*(n-1)/2*mdim
    int c5=p*(p-1)/2*udim;
    for (int kk=0; kk<n; ++kk) {
        int kkk=C.x.user[kk].item.size();
        c1+=kkk;//c1=sum mi
    }
    vector<rated_1itemmore> itemmore=construct_itemmore(C.x);//is needed, records the info for each item
    int c6=p*udim2*(udim2-1)/2;
    int c7=n*mdim2*(mdim2-1)/2;
    for(int kk=0;kk<p;++kk){
        int usersize=itemmore[kk].user.size();
        cout<<"item "<<kk<<" rated by "<<usersize<<" people \n";
    }
    
    double maxdiff_outer=1;
    int iter_outer=0;
    int i,j,k,ik,il;//ik,il?
    int l;
    VectorXi fdi(1),sdi(1);
    VectorXd temp,atemp(2),btemp(2),a(2),b(2);
    MatrixXd mtemp(n,n);
    //cout<<"done here "<<endl;
    double obj1, obj2;
    vector<VectorXd> pretilde(n);//records initial error
    
    VectorXi mis(n);
#pragma omp parallel for
    for (i=0; i<n; ++i) {
        mis[i]=C.x.user[i].item.size();//mi for each user
    }
    
    MatrixXd mualpha2=C.mualpha1;
    MatrixXd mubeta2=C.mubeta1;
    VectorXd wis2=C.wis1;//C.Omegais1 is n p*p matrices, but only part of it is used.
    MatrixXd* theta=new MatrixXd;//all use theta, resize it for alpha and beta respectively
    MatrixXd* u=new MatrixXd;//all use u, resize it for alpha and beta respectively
    MatrixXd* theta2=new MatrixXd;
    MatrixXd* u2=new MatrixXd;
    MatrixXd* eta=new MatrixXd;//for double cluster
    MatrixXd* v=new MatrixXd;
    MatrixXd* eta2=new MatrixXd;
    MatrixXd* v2=new MatrixXd;
    MatrixXd* z=new MatrixXd;
    MatrixXd* gamma=new MatrixXd;
    MatrixXd* z2=new MatrixXd;
    MatrixXd* gamma2=new MatrixXd;
    *theta=MatrixXd::Zero(mdim,n*(n-1)/2);
    *u=*theta;
    *eta=MatrixXd::Zero(mdim2*(mdim2-1)/2,n);
    *v=*eta;
    *z=mubeta2;
    *gamma=MatrixXd::Zero(mdim,n);
    *theta2=MatrixXd::Zero(udim,p*(p-1)/2);
    *u2=*theta2;
    *eta2=MatrixXd::Zero(udim2*(udim2-1)/2,p);
    *v2=*eta2;
    *z2=mualpha2;
    *gamma2=MatrixXd::Zero(udim,p);
    VectorXd eta_w=VectorXd::Zero(n*(n-1)/2);//proxy variable for wi-wj
    VectorXd u_w=VectorXd::Zero(n*(n-1)/2);//lagrangian multiplier for wis
    VectorXd ci(n);//ci for w
    VectorXd si(n);
    pretilde=cal_resid(mualpha2,mubeta2,C.x.user,C.movie,C.users);
    
    cout<<"Initially error for user 1"<<" is "<<(pretilde[1]).transpose()<<endl;
    cout<<"Initially error for user 30"<<" is "<<(pretilde[30]).transpose()<<endl;
    cout<<"Initially error for user "<<n-1<<" is "<<(pretilde[n-1]).transpose()<<endl;
    
    
    double maxbeta=0,maxalpha=0,maxw=0,maxtheta=0,maxu=0,maxZ=0,maxgamma=0,maxeta=0,maxv=0,maxeta_w=0,maxu_w=0,ttt=0;
    VectorXd change,change2;
    dclust_var_l1_obj_struct A;
    A.mualpha=C.mualpha1;
    A.mubeta=C.mubeta1;
    A.wis=C.wis1;
    A.lambda1=C.lambda1;
    A.lambda2=C.lambda2;
    A.lambda3=C.lambda3;
    A.user=C.x.user;
    A.movie=C.movie;
    A.users=C.users;
    A.itemmore=itemmore;
    A.c1=c1;
    A.c3=c3;
    A.c4=c4;
    A.c5=c5;
    A.c6=c6;
    A.c7=c7;
    A.udim2=udim2;
    A.mdim2=mdim2;
    obj1=cal_dcluster_l1_obj_struct(A);//define function cal_dcluster_l1_obj_struct
    cout<<"obj initial="<<obj1<<endl;
    
    VectorXd zis(n), uis(n);
    double rho2=C.rho2;//for rho_Omega.
    double rho_beta=c1*C.rho/c4;//correct
    double rho_alpha=c1*C.rho/c5;//?
    cout<<"rho_alpha="<<rho_alpha<<endl;
    cout<<"rho_beta="<<rho_beta<<endl;
    double rho_w=c1*rho2/c3;
    double off_diff=0;//whole part?? add when solve wis later
    
    vector<VectorXi> fdibeta(n-1),sdibeta(n-1),fdialpha(p-1),sdialpha(p-1),fdibetainner(mdim2-1),sdibetainner(mdim2-1),fdialphainner(udim2-1),sdialphainner(udim2-1);
    construct_fdisdi(n,fdibeta,sdibeta);
    construct_fdisdi(p,fdialpha,sdialpha);
    construct_fdisdi(mdim2,fdibetainner,sdibetainner);
    construct_fdisdi(udim2,fdialphainner,sdialphainner);
    
    
    //for inverse for Zbeta, build inverse matrix
    MatrixXd invZb(n*mdim,n*mdim);
    MatrixXd A2beta=MatrixXd::Zero(n*mdim2*(mdim2-1)/2,n*mdim);
    construct_Zinv(n,mdim,mdim2,invZb,A2beta);
    
    //for inverse for Zalpha, build inverse matrix and A1alpha, A2alpha
    MatrixXd invZa(p*udim,p*udim);
    MatrixXd A2alpha=MatrixXd::Zero(p*udim2*(udim2-1)/2,p*udim);
    construct_Zinv(p,udim,udim2,invZa,A2alpha);
    
    VectorXd t1(n*mdim);
    VectorXd A1t2(n*mdim);
    VectorXd A2t3(n*mdim);
    VectorXd maxvec(7);
    
    VectorXd t1a(p*udim);
    VectorXd t3a(p*udim2*(udim2-1)/2);
    VectorXd A1t2a(p*udim);
    VectorXd A2t3a(p*udim);
    
    while (iter_outer<C.maxIter) {
        time_t tstart,tend;
        
        
        MatrixXd beta_old=mubeta2;
        MatrixXd alpha_old=mualpha2;
        VectorXd wis_old=wis2;//for calculating diff in each outer_iter
        
        //update mubeta
        int beta_iter=0;
        //construct mtemp and tempB for each user
        vector<MatrixXd> mtempB(n);
        vector<VectorXd> tempB0(n);
        vector<VectorXd> tempB(n);
        vector<VectorXd> tempA(p);//for alpha
        
#pragma omp parallel for
        for(i=0;i<n;++i){
            MatrixXd movieuse=matrix_colsub(C.movie,C.x.user[i].item);
            mtempB[i]=C.wis1[i]*movieuse*movieuse.transpose();
            mtempB[i]=mtempB[i]+rho_beta*(n-1)*MatrixXd::Identity(mdim,mdim);
            mtempB[i]=mtempB[i].inverse();
            
            MatrixXd alphause=matrix_colsub(C.mualpha1,C.x.user[i].item);
            VectorXd useri=C.users.col(i);
            tempB0[i]=C.x.user[i].rating-(alphause.transpose())*useri;
            tempB0[i]=C.wis1[i]*movieuse*tempB0[i];
        }
        
        double timer1=0,timer2=0,timer3=0,timer4=0;
        tstart=time(0);
        while(beta_iter<5000/*C.maxIter*/){
            high_resolution_clock::time_point start1 = high_resolution_clock::now();
            maxbeta=0;
            tempB=tempB0;
            //update mubeta
#pragma omp parallel for
            for (i=0; i<n; ++i) {
                tempB[i]+=rho_beta*((*z).col(i)-(*gamma).col(i));
                mubeta2.col(i)=mtempB[i]*tempB[i];
            }
            maxbeta=absm(mubeta2-C.mubeta1).maxCoeff();
            maxvec(0)=maxbeta;
            C.mubeta1=mubeta2;
            high_resolution_clock::time_point stop1 = high_resolution_clock::now();
            timer1+=(duration_cast<duration<double>>(stop1 - start1)).count();
            
            //update z and gamma
            //first build t1, A1t2 and A2t3
            
            MatrixXd Zpre=*z;
            construct_t1_A1t2_A2t3(n,mdim,mdim2,fdibeta,sdibeta,fdibetainner,sdibetainner,mubeta2,gamma,theta,u,eta,v,t1,A1t2,A2t3);
            
            start1=high_resolution_clock::now();
            VectorXd Ztemp=invZb*(t1+A1t2+A2t3);
            stop1 = high_resolution_clock::now();//
            timer2+=(duration_cast<duration<double>>(stop1 - start1)).count();
            //this step if use A1^T*t2, multiplication very slow, other steps doesn't use time almost
            //multiplication still very slow, dim too high: 1900
            
            //cout<<"dim(invZb) is "<<invZb.rows()<<", "<<invZb.cols()<<endl;
            
            change.resize(n);
            temp.resize(mdim);
            
            
#pragma omp parallel for private(temp)
            for (i=0; i<n; ++i) {
                (*z).col(i)=Ztemp.segment(i*mdim,mdim);
                temp=mubeta2.col(i)-(*z).col(i);
                change(i)=temp.lpNorm<Infinity>();
                (*gamma).col(i)+=temp;
            }
            
            maxZ=absm(Zpre-(*z)).maxCoeff();
            maxgamma=change.lpNorm<Infinity>();
            maxvec(1)=maxZ;
            maxvec(2)=maxgamma;
            
            //update theta and u
            start1=high_resolution_clock::now();
            maxtheta=0;
            maxu=0;
            change.resize((*theta).cols());//for theta
            change2.resize(change.size());//for u
#pragma omp parallel for private(j,l,temp,atemp,btemp,a)
            for (i=0; i<(n-1); ++i) {
                for (j=(i+1); j<n; ++j) {
                    l=i*(n-(i+1)*0.5)+j-i-1;
                    temp=(*z).col(i)-(*z).col(j);
                    btemp=(*u).col(l);
                    atemp=temp+btemp;
                    a=ST_vec(atemp,C.lambda1/(2*C.rho));
                    atemp=a-(*theta).col(l);
                    (*theta).col(l)=a;
                    change[l]=atemp.lpNorm<Infinity>();//change is for theta
                    atemp=temp-a;
                    change2[l]=atemp.lpNorm<Infinity>();
                    (*u).col(l)+=atemp;
                }
            }
            maxtheta=change.lpNorm<Infinity>();
            maxu=change2.lpNorm<Infinity>();
            maxvec(3)=maxtheta;
            maxvec(4)=maxu;
            stop1 = high_resolution_clock::now();
            timer3+=(duration_cast<duration<double>>(stop1 - start1)).count();
            
            //update eta and v
            start1=high_resolution_clock::now();
            maxeta=0;
            maxv=0;
            change.resize(n);//for eta
            change2.resize(n);//for v
            a.resize(mdim2*(mdim2-1)/2);
            
#pragma omp parallel for private(j,k,l,temp,atemp,a)
            for (i=0; i<n; ++i) {
                VectorXd temp(mdim2*(mdim2-1)/2);
                VectorXd atemp(mdim2*(mdim2-1)/2);
                for (j=0; j<mdim2-1; ++j) {
                    for (k=j+1; k<mdim2; ++k) {
                        l=j*(mdim2-(j+1)*0.5)+k-j-1;
                        temp(l)=(*z)(j,i)-(*z)(k,i);
                        atemp(l)=temp(l)+(*v)(l,i);
                    }
                }
                a=ST_vec(atemp,C.lambda1*c4/(2*C.rho*c7));
                atemp=a-(*eta).col(i);
                (*eta).col(i)=a;
                change(i)=atemp.lpNorm<Infinity>();//()/[] are both fine for Eigen::Vectors
                atemp=temp-a;
                change2(i)=atemp.lpNorm<Infinity>();
                (*v).col(i)+=atemp;
            }
            maxeta=change.lpNorm<Infinity>();
            maxv=change2.lpNorm<Infinity>();
            maxvec(5)=maxeta;
            maxvec(6)=maxv;
            stop1 = high_resolution_clock::now();
            timer4+=(duration_cast<duration<double>>(stop1 - start1)).count();
            
            beta_iter+=1;
            ttt=maxvec.lpNorm<Infinity>();
            //cout<<"beta_iter="<<beta_iter<<", ttt="<<ttt<<endl;
            
            bool judge=ttt<2*C.Tol;
            //judge=judge||(iter_outer<15&&beta_iter>10);
            //judge=judge||(iter_outer<30&&beta_iter>30);
            judge=judge||(iter_outer<200&&beta_iter>10);
            if(judge) break;
            
        }
        tend=time(0);
        if (iter_outer>=200) {
            cout<<"beta timer1="<<timer1<<", timer2="<<timer2<<", timer3="<<timer3<<", timer4="<<timer4<<endl;
        }
        
        cout<<"beta_iter is "<<beta_iter<<", with max diff " <<ttt<<endl;
        if(ttt>=2*C.Tol){
            cout<<"beta didn't converge!" <<endl;
        }
        cout<<"beta done"<<endl;
        A.mubeta=mubeta2;
        obj2=cal_dcluster_l1_obj_struct(A);
        cout<<"obj="<<obj2<<endl;
        
        
        
        cout<<"beta for v1 cost "<<difftime(tend, tstart)<<" secs. \n";
        
        //update mualpha
        int alpha_iter=0;
        mtempB.resize(p);//store matrix inverse for alpha
        tempB0.resize(p);
#pragma omp parallel for private(j)
        for(i=0;i<p;++i){
            mtempB[i]=rho_alpha*(p-1)*MatrixXd::Identity(udim,udim);
            int mj=itemmore[i].user.size();
            MatrixXd Omega_alpha=MatrixXd::Zero(mj,mj);//must initialize to 0
            VectorXd ritemp(mj);
            MatrixXd mtemp2(udim,mj);
            for (j=0; j<mj; ++j) {
                int userno=itemmore[i].user[j];
                int numberforuser=itemmore[i].numberforuser[j];
                mtemp2.col(j)=C.users.col(userno);
                Omega_alpha(j,j)=C.wis1[userno];
                VectorXd tempmovie=C.movie.col(i);
                ritemp(j)=C.x.user[userno].rating[numberforuser]-(tempmovie).dot(mubeta2.col(userno));
            }
            mtempB[i]=mtempB[i]+mtemp2*Omega_alpha*mtemp2.transpose();
            mtempB[i]=mtempB[i].inverse();
            tempB0[i]=mtemp2*Omega_alpha*ritemp;
        }
        
        while(alpha_iter<C.maxIter){
            //cout<<"using parallel for alpha"<<endl;
            maxalpha=0;
            tempA=tempB0;
#pragma omp parallel for
            for(i=0;i<p;++i){
                tempA[i]+=rho_alpha*((*z2).col(i)-(*gamma2).col(i));
                mualpha2.col(i)=mtempB[i]*tempA[i];
            }
            
            maxalpha=absm(mualpha2-C.mualpha1).maxCoeff();
            maxvec(0)=maxalpha;
            C.mualpha1=mualpha2;
            
            //update z2 and gamma2
            //first build t1a, t2a and t3a
            /*#pragma omp parallel for
             for (i=0; i<p; ++i) {
             t1a.segment(i*udim,udim)=mualpha2.col(i)+(*gamma2).col(i);//ok
             t3a.segment(i*udim2*(udim2-1)/2,udim2*(udim2-1)/2)=(*eta2).col(i)-(*v2).col(i);
             }
             
             #pragma omp parallel for private(j,fdi,sdi)//why this influences final result in second alpha_iter(first_iter both are 0), should reset A1t2a and A2t3a to 0 first
             for (i=0; i<p; ++i) {
             if(i<p-1){
             fdi=fdialpha[i];
             for (j=0; j<fdi.size(); ++j) {
             A1t2a.segment(i*udim,udim)+=((*theta2).col(fdi[j])-(*u2).col(fdi[j]));
             }
             }
             if(i>0){
             sdi=sdialpha[i-1];
             for (j=0; j<sdi.size(); ++j) {
             A1t2a.segment(i*udim,udim)-=((*theta2).col(sdi[j])-(*u2).col(sdi[j]));
             }
             }
             }*/
            construct_t1_A1t2_A2t3(p,udim,udim2,fdialpha,sdialpha,fdialphainner,sdialphainner,mualpha2,gamma2,theta2,u2,eta2,v2,t1a,A1t2a,A2t3a);
            MatrixXd Zpre=*z2;
            VectorXd Ztemp=invZa*(t1a+A1t2a+A2t3a);
            change.resize(p);
            temp.resize(udim);
            
            
#pragma omp parallel for private(temp)
            for (i=0; i<p; ++i) {
                (*z2).col(i)=Ztemp.segment(i*udim,udim);
                temp=mualpha2.col(i)-(*z2).col(i);
                change(i)=temp.lpNorm<Infinity>();
                (*gamma2).col(i)+=temp;
            }
            maxZ=absm(Zpre-(*z2)).maxCoeff();
            maxgamma=change.lpNorm<Infinity>();
            maxvec(1)=maxZ;
            maxvec(2)=maxgamma;
            
            //update theta2 and u2
            maxtheta=0;
            maxu=0;
            change.resize((*theta2).cols());//for theta2
            change2.resize(change.size());//for u2
#pragma omp parallel for private(j,l,temp,atemp,btemp,a)
            for (i=0; i<(p-1); ++i) {
                for (j=(i+1); j<p; ++j) {
                    l=i*(p-(i+1)*0.5)+j-i-1;
                    temp=(*z2).col(i)-(*z2).col(j);
                    btemp=(*u2).col(l);
                    atemp=temp+btemp;
                    a=ST_vec(atemp,C.lambda1/(2*C.rho));
                    atemp=a-(*theta2).col(l);
                    (*theta2).col(l)=a;
                    change[l]=atemp.lpNorm<Infinity>();//change is for theta
                    atemp=temp-a;
                    change2[l]=atemp.lpNorm<Infinity>();
                    (*u2).col(l)+=atemp;
                }
            }
            maxtheta=change.lpNorm<Infinity>();
            maxu=change2.lpNorm<Infinity>();
            maxvec(3)=maxtheta;
            maxvec(4)=maxu;
            
            
            //update eta2 and v2
            maxeta=0;
            maxv=0;
            change.resize(p);//for eta
            change2.resize(p);//for v
            a.resize(udim2*(udim2-1)/2);
#pragma omp parallel for private(j,k,l,temp,atemp,a)
            for (i=0; i<p; ++i) {
                VectorXd temp(udim2*(udim2-1)/2);
                VectorXd atemp(udim2*(udim2-1)/2);
                for (j=0; j<udim2-1; ++j) {
                    for (k=j+1; k<udim2; ++k) {
                        l=j*(udim2-(j+1)*0.5)+k-j-1;
                        temp(l)=(*z2)(j,i)-(*z2)(k,i);
                        atemp(l)=temp(l)+(*v2)(l,i);
                    }
                }
                a=ST_vec(atemp,C.lambda1*c5/(2*C.rho*c6));
                atemp=a-(*eta2).col(i);
                (*eta2).col(i)=a;
                change(i)=atemp.lpNorm<Infinity>();//()/[] are both fine for Eigen::Vectors
                atemp=temp-a;
                change2(i)=atemp.lpNorm<Infinity>();
                (*v2).col(i)+=atemp;
            }
            maxeta=change.lpNorm<Infinity>();
            maxv=change2.lpNorm<Infinity>();
            maxvec(5)=maxeta;
            maxvec(6)=maxv;
            
            alpha_iter+=1;
            ttt=maxvec.lpNorm<Infinity>();
            //cout<<"alpha_iter="<<alpha_iter<<", ttt="<<ttt<<endl;
            bool judge=ttt<2*C.Tol;
            //judge=judge||(iter_outer<15&&alpha_iter>10);
            //judge=judge||(iter_outer<30&&alpha_iter>30);
            judge=judge||(iter_outer<200&&alpha_iter>10);
            if(judge) break;
        }
        
        tend=time(0);
        
        cout<<"alpha, beta for v1 cost "<<difftime(tend, tstart)<<" secs. \n";
        cout<<"alpha_iter is "<<alpha_iter<<" with maxdiff "<<ttt<<endl;
        if(ttt>=2*C.Tol){
            cout<<"alpha didn't converge!" <<endl;
        }
        cout<<"alpha done"<<endl;
        A.mualpha=mualpha2;
        obj2=cal_dcluster_l1_obj_struct(A);
        cout<<"obj="<<obj2<<endl;
        
        
        //alpha, beta finished updating, next update w
        pretilde=cal_resid(mualpha2,mubeta2,C.x.user,C.movie,C.users);
#pragma omp parallel for
        for (i=0; i<n; ++i) {
            si[i]=(pretilde[i]).dot(pretilde[i]);
        }
        cout<<"si="<<si.transpose()<<endl;
        
        tstart=time(0);
        wis2=C.wis1;
        int Omega_iter=0;
        change.resize(n);
        high_resolution_clock::time_point start1 = high_resolution_clock::now();
        
        while(Omega_iter<C.maxIter){
            VectorXd ctemp=VectorXd::Zero(n);
#pragma omp parallel for private(j,fdi,sdi)
            for (i=0; i<n; ++i) {
                if(i<n-1){
                    fdi=fdibeta[i];
                    for (j=0; j<fdi.size(); ++j) {
                        ctemp[i]+=(eta_w(fdi[j])-u_w(fdi[j]));
                    }
                }
                if(i>0){
                    sdi=sdibeta[i-1];
                    for (j=0; j<sdi.size(); ++j) {
                        ctemp[i]-=(eta_w(sdi[j])-u_w(sdi[j]));
                    }
                }
            }
            int Omega_inner_iter=0;
            double diff;
            while(Omega_inner_iter<C.maxIter){
                VectorXd current=wis2;
                for (i=0; i<n; ++i) {
                    ci[i]=0.5*si[i]+rho_w*(-wis2.sum()+wis2[i]-ctemp[i]);//this should be sequential
                    wis2[i]=-ci[i]+sqrt(ci[i]*ci[i]+2*mis[i]*rho_w*(n-1));
                    wis2[i]=wis2[i]/(2*rho_w*(n-1));
                }
                diff=(wis2-current).lpNorm<Infinity>();
                if (diff<2*C.Tol) {
                    break;
                }
                ++Omega_inner_iter;
            }//got wis2
            //cout<<"Omega_inner_iter="<<Omega_inner_iter<<", with diff="<<diff<<endl;
            //cout<<"wis2="<<wis2.transpose()<<endl;
            maxw=(wis2-C.wis1).lpNorm<Infinity>();
            
            //update eta_w and u_w
            atemp.resize(n*(n-1)/2);
            btemp.resize(atemp.size());
#pragma omp parallel for private(j,l)
            for (i=0; i<(n-1); ++i) {
                for (j=(i+1); j<n; ++j) {
                    l=i*(n-(i+1)*0.5)+j-i-1;
                    atemp(l)=wis2(i)-wis2(j);
                }
            }
            btemp=atemp+u_w;
            a=ST_vec(btemp,C.lambda3/C.rho2);
            btemp=a-eta_w;
            maxeta_w=btemp.lpNorm<Infinity>();
            eta_w=a;
            
            btemp=atemp-eta_w;
            maxu_w=btemp.lpNorm<Infinity>();
            u_w+=btemp;
            
            ttt=max((max(maxw,maxeta_w)),maxu_w);
            //cout<<"atemp="<<atemp.head(10).transpose()<<endl;cout<<"eta_w="<<eta_w.head(10).transpose()<<endl;
            //cout<<"Omega_iter="<<Omega_iter<<", with ttt="<<ttt<<", maxw="<<maxw<<", maxeta_w="<<maxeta_w<<", maxu_w="<<maxu_w<<endl;
            ++Omega_iter;
            C.wis1=wis2;
            bool judge=(Omega_iter>1)&&ttt<2*C.Tol;
            //judge=judge||(iter_outer<15&&Omega_iter>10);
            //judge=judge||(iter_outer<30&&Omega_iter>30);
            judge=judge||(iter_outer<200&&Omega_iter>10);
            if(judge) break;//too strict? 2*C.Tol?((Omega_iter>1)&&ttt<0.1/sqrt(iter_outer))
        }
        
        high_resolution_clock::time_point stop1 = high_resolution_clock::now();
        Omega_eigen_time=Omega_eigen_time+duration_cast<milliseconds>(stop1 - start1).count();
        cout<<"Omega_eigen_time="<<Omega_eigen_time<<endl;
        cout<<"Above cost "<<(duration_cast<duration<double>>(stop1 - start1)).count()<<" s (using ms)"<<endl;
        tend=time(0);
        cout<<"Omega for v1 cost "<<difftime(tend, tstart)<<" secs. \n";
        cout<<"Omega_iter is "<<Omega_iter<<endl;
        cout<<"Omega maxdiff is "<<ttt<<endl;
        
        A.wis=wis2;
        obj2=cal_dcluster_l1_obj_struct(A);
        cout<<"obj="<<obj2<<endl;
        double perc=abs(obj2-obj1)/abs(obj1);
        cout<<"(obj1-obj2) in obj is "<<(obj1-obj2)<<endl;
        
        
        maxw=(wis_old-wis2).lpNorm<Infinity>();
        maxbeta=absm(mubeta2-beta_old).maxCoeff();
        maxalpha=absm(mualpha2-alpha_old).maxCoeff();
        
        maxdiff_outer=max(max(maxalpha,maxbeta),maxw);
        cout<<"maxdiff_outer="<<maxdiff_outer<<endl;
        cout<<"wis[30]="<<wis2(30)<<endl;
        //cout<<"wis="<<wis2.transpose()<<endl;
        beta_old=mubeta2;
        alpha_old=mualpha2;
        wis_old=wis2;
        iter_outer=iter_outer+1;
        cout<<"iter is "<<iter_outer<<endl;
        if(abs(obj1-obj2)<0.01/*&&iter_outer>200  5e-3 &&maxdiff_outer<1e-3*/){// for cold-start, this may be too strict, change to abs(obj1-obj2)/abs(obj1)<1e-4?
            //abs(obj1-obj2)<5e-3 still too strict? Change to abs(obj1-obj2)<0.01?or evern 0.1?
            if(outer_first==1&&iter_outer>205) break;//previously was 0.1,now running 0.01
            if(outer_first==0) break;
        }
        else{
            obj1=obj2;
        }
    }
    if (iter_outer==C.maxIter) {cout<<"maxdiff is "<<maxdiff_outer<<endl;}
    else cout<<"iter is "<<iter_outer<<endl;
    cout<<"Omega_eigen_time="<<Omega_eigen_time<<endl;
    
    delete theta,u,theta2,u2,eta,v,eta2,v2,z,gamma,z2,gamma2;
    
    mtemp.resize(n,p);
    mtemp=C.users.transpose()*mualpha2+mubeta2.transpose()*C.movie;//compact multi is faster than one by one
    cout<<"maxdiff is "<<maxdiff_outer<<endl;
    dresult re;
    re.mualpha=mualpha2;
    re.mubeta=mubeta2;
    re.wis=wis2;
    re.solu=mtemp;
    //cout<<"MSE_train is "<<cal_MSE(mtemp,C.x)<<endl;
    re.maxdiff=maxdiff_outer;
    re.obj=obj2;
    return re;
}//reference as para, change C value