#include "consider_covariance.h"
//Difference from _v6, this changes the iterations inside iter_ori
result Cluster_p_inADMM_scale_struct_v12(double c,Cluster_mnl_p_ADMM_scale_para3 &C){
    //difference from v8, use Cluster_mnl_p_ADMM_scale_para2, not pointers for theta, u, etc...
    //use d.o.c. for L1 version, this is convexity adjustment
    //does propagation of Omega
    //inexact admm
    //with L1 penalty on Omega
    //diff from v4: Apply ADMM together for alpha, beta(Inexact) and Omega
    //alpha change sometimes increase obj value? find alpha bug
    //v4 saves the matrix inversions for alpha and beta, also saves a part of temp for beta
    //add one more cycle for alpha and beta to minimize w.r.t.
    //this function uses ADMM for Omega
    //x should be train data
    int n=C.users.rows();
    int p=C.movie.rows();
    int udim=C.users.cols();
    int mdim=C.movie.cols();
    //cout<<"done here "<<endl;
    int c1=0,c2=0,c3=0,c6=0;
    int c4=n*(n-1)/2*mdim;
    int c5=p*(p-1)/2*udim; 
    for (int kk=0; kk<n; ++kk) {
        int kkk=C.x.user[kk].item.size();
        c1+=kkk;
        c2+=kkk*kkk;   
    }
    c3=n*p*(p-1)/2;
    vector<Omega_1rlv> rlv=construct_rlv(p,C.x.user);
    cout<<"rlv.size is "<<rlv.size()<<endl;
    vector<rated_1itemmore> itemmore=construct_itemmore(C.x);  
    c6=(p+rlv.size())*n*(n-1)/2;
    
    //MatrixXi nrated=MatrixXi::Zero(p,p);
    for(int kk=0;kk<p;++kk){
        int usersize=itemmore[kk].user.size();
        cout<<"item "<<kk<<" rated by "<<usersize<<" people \n";
    }
    //for (int k=0; k<rlv.size();++k) {
        //int usersize=rlv[k].userno.size();
        //cout<<"item1 "<<rlv[k].item1<<", item2 "<<rlv[k].item2<<" by "<<usersize<<endl;
        //nrated(rlv[k].item1,rlv[k].item2)=usersize;
        //nrated(rlv[k].item2,rlv[k].item1)=usersize;
    //}
    //cout<<"nrated="<<endl<<nrated<<endl;
    
    MatrixXd mualpha2=C.mualpha1;
    MatrixXd mubeta2=C.mubeta1;  
    
    vector<MatrixXd> Omegais2=C.Omegais1;//C.Omegais1 is n p*p matrices, but only part of it is used.
    //double useless=cal_offdiff_prop(trin,Omegais2);
    //cout<<"Omegais2[17]="<<endl<<Omegais2[17]<<endl;
    MatrixXd theta=C.theta;//all use theta, resize it for alpha and beta respectively
    MatrixXd u=C.u;//all use u, resize it for alpha and beta respectively
    MatrixXd theta2=C.theta2;
    MatrixXd u2=C.u2;
    
    double maxdiff_admm=1,maxdiff_inner=0;
    int doc_iter=0,admm_iter=0;
    int iter_ori;//iter for alpha, beta and Omega (original variable)
    int qua_iter,quasi_coor_iter;
    int i,j,k,l,ik,il;
    VectorXi fdi(1),sdi(1);
    VectorXd temp,atemp(2),btemp(2),a(2),b(2);
    MatrixXd mtemp(n,n);
    //cout<<"done here "<<endl;
    double obj1,obj2,obj_doc1,obj_doc2;
    double rt=1-0.3;//first rt smaller than 1
    vector<MatrixXd> Sis(n);
    vector<VectorXd> pretilde(n);
    
    tri trin,trip;
    trin.first.resize(n*(n-1)/2);trin.second.resize(n*(n-1)/2);
    l=0;
    for (i=0; i<n-1; ++i) {
        for (j=i+1; j<n; ++j) {
            trin.first(l)=i;trin.second(l)=j;
            ++l;
        }
    }
    
    trip.first.resize(p*(p-1)/2);trip.second.resize(p*(p-1)/2);
    l=0;
    for (i=0; i<p-1; ++i) {
        for (j=i+1; j<p; ++j) {
            trip.first(l)=i;trip.second(l)=j;
            ++l;
        }
    }
    
    vector<int> itemmore_idle(p),rlv_idle(rlv.size());
#pragma omp parallel for private(j)
    for (i=0; i<p; ++i) {
        itemmore_idle[i]=-1;
        //stl_vec_cout(itemmore[i].user);
        for (j=0; j<itemmore[i].user.size(); ++j) {
            if(j<itemmore[i].user[j]) break;
        }//result j is one user didn't rate this item
        itemmore_idle[i]=j;
        //cout<<"itemmore_idle["<<i<<"]="<<itemmore_idle[i]<<endl;
    }
#pragma omp parallel for private(l)
    for (i=0; i<rlv.size(); ++i) {
        rlv_idle[i]=-1;
        for (l=0; l<rlv[i].userno.size(); ++l) {
            if(l<rlv[i].userno[l]) break;
        }//result l is one user didn't rate this pair
        rlv_idle[i]=l;
    }
    
    int total_estimate_size=0;//store the number of parameters need to be estimated in Omega
    VectorXi size_est_diag(p);//number of parameters need to estimate for each (k,k) diagonal, it's in order of 0,1,2,...(p-1)th item
    //number rated i +1/0
#pragma omp parallel for reduction(+:total_estimate_size)
    for (i=0; i<p; ++i) {
        size_est_diag[i]=itemmore[i].user.size();
        if (itemmore_idle[i]!=-1) {
            size_est_diag[i]+=1;//count for those who didn't rate this item
        }
        total_estimate_size=total_estimate_size+size_est_diag[i];
    }
    VectorXi size_est_offdiag(rlv.size());//number of parameters need to estimate for each (k,l) offdiag, it's in order of rlv pair
#pragma omp parallel for reduction(+:total_estimate_size)
    for (i=0; i<rlv.size(); ++i) {
        size_est_offdiag[i]=rlv[i].userno.size();
        if (rlv_idle[i]!=-1) {
            size_est_offdiag[i]+=1;//count for those who didn't rate this item pair
        }
        total_estimate_size+=size_est_offdiag[i];
    }
    cout<<"Total number of parameters to be estimated in Omega is "<<total_estimate_size<<endl;
#pragma omp parallel for private(btemp)
    for (i=0; i<n; ++i) {//construct pretilde
        MatrixXd alphause=matrix_colsub(mualpha2,C.x.user[i].item);
        MatrixXd movieuse=matrix_rowsub(C.movie,C.x.user[i].item);
        VectorXd useri=C.users.row(i);
        btemp=mubeta2.row(i);
        pretilde[i]=alphause.transpose()*useri+movieuse*btemp;
        pretilde[i]=C.x.user[i].rating-pretilde[i];
        Sis[i]=pretilde[i]*(pretilde[i].transpose());
        if(i==1||i==25||i==70){
            cout<<"Initially error for user "<<i<<" is "<<(pretilde[i]).transpose()<<endl;
        }
    }
    
    
    double maxbeta=0,maxalpha=0,maxOmega=0,maxtheta_beta=0,maxu_beta=0,maxtheta_alpha=0,maxu_alpha=0,maxZ=0,maxU=0,tttalpha=0,tttbeta=0,tttOmega=0,maxbeta_doc=0,maxalpha_doc=0;
    VectorXd change(n),change2;
    obj_struct A;//B keeps previous iter
    A.mualpha=&C.mualpha1;
    A.mubeta=&C.mubeta1;
    A.Omegais=&C.Omegais1;
    A.lambda1=C.lambda1;
    A.lambda2=C.lambda2;
    A.lambda3=C.lambda3;
    A.user=&C.x.user;
    A.c1=c1;
    A.c2=c2;
    A.c3=c3;
    A.c4=c4;
    A.c5=c5;
    A.c6=c6;
    A.off_diff=cal_offdiff_prop(trin,*A.Omegais);
    cout<<"off_diff="<<A.off_diff<<endl;
    
    
    double rho2=C.rho2;//for rho_Omega
    double rho_beta=c1*C.rho/c4;
    double rho_alpha=c1*C.rho/c5;
    cout<<"rho_alpha="<<rho_alpha<<endl;
    cout<<"rho_beta="<<rho_beta<<endl;
    double rho1_Omega=c1*rho2/c6;
    double rho2_Omega=c1*rho2/c3;
    double rho_ab=C.lambda1/(2*C.rho);
    double rho_diff=C.lambda3/C.rho2;
    double rho_size=C.lambda2/C.rho2;
    double off_diff=0;
    
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
    
    MatrixXd alphause,movieuse,Omegaissub,betapre,alphapre;
    VectorXd useri;
    vector<MatrixXd> mtempB(n),Omegaispre;
    vector<VectorXd> tempB0(n),tempB(n),tempB1(n),tempA(p),tempA1(p);//B1 and A1 are for the part with only theta and u
    VectorXd betasum,alphasum;
    
    vector<vector<double>> X=C.X;//for X, surrogate variable of w off-diagonal for size constraint, size equals rlv.size()
    vector<vector<double>> Z=C.Z;//for X, dual variable of w off-diagonal for size constraint
    vector<vector<double>> size_constr=X;
    vector<vector<double>> eta=C.eta;//in the order of diagonal first (p) (0---(p-1)), then offdiag, (rlv.size(),same pair order as rlv), surrogate variable for difference constraint
    vector<vector<double>> y=C.y;//in the order of diagonal first (p) (0---(p-1)), then offdiag, (rlv.size(),same pair order as rlv)
    vector<vector<double>> diff_constr=eta;//in the order of diagonal first (p) (0---(p-1)), then offdiag, (rlv.size(),same pair order as rlv)
    vector<vector<double>> diff_constr_old,size_constr_old;
    //diff_constr is not just constraint of difference, it's w-w-eta+y;similarly, size constrain is w-x+z
    MatrixXd Omegais2sub;
    MatrixXd atempM,tempM,btempM;
    MatrixXd betadoc_pre=mubeta2,alphadoc_pre=mualpha2;
    vector<MatrixXd> Omegaisdoc_pre(n);//store sub-matrix directly
    vector<MatrixXd> Omegaissub_qua(n),Wis(n);
    vector<MatrixXd> Dis_pre(n),Ais(n),Sis_denom(n),muis(n);//Dis_pre: previous update direction for quadratic approximation; muis: change in update direction; Ais: WDW; Zpis: submatrix of Zis-Uis
    VectorXi ind1,ind2,sizeis(n);
    double beta=1,sigma=1e-2;
#pragma omp parallel for
    for (i=0; i<n; ++i) {
        Omegaissub_qua[i]=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);//Omegaissub in quadratic (t)
        sizeis[i]=Omegaissub_qua[i].rows();
        LLT<MatrixXd> llt(Omegaissub_qua[i]);
        Wis[i]=llt.solve(MatrixXd::Identity(sizeis[i],sizeis[i]));
        muis[i]=MatrixXd::Zero(sizeis[i],sizeis[i]);
    }
    
    vector<vector<int>> number_for_pair(n);//for each user, it's 1st,2nd pair rated by # people,1st, 3rd pair rated by #people...
#pragma omp parallel for private(j,k,l)
    for (i=0; i<n; ++i) {
        for (j=0; j<sizeis[i]; ++j) {
            int itemj=C.x.user[i].item[j];
            for (k=j+1; k<sizeis[i]; ++k) {
                int itemk=C.x.user[i].item[k];
                for (l=0; l<rlv.size(); ++l) {
                    if (rlv[l].item1==itemj&&rlv[l].item2==itemk) {
                        number_for_pair[i].push_back(rlv[l].userno.size());
                        break;
                    }
                }
            }
        }
    }
    
    //first part with rt!=1
    
    time_t tstart1,tend1,tstart2,tend2,tstart3,tend3,tstart4,tend4,tstart5,tend5,tstart6,tend6,tstart7,tend7;
    double time1=0,time2=0,time3=0,time4=0,time5=0,time6=0;
    
    obj_doc1=cal_obj_prop_struct_3(Sis,A);//the function value d.o.c. needs to minimize, d.o.c. check its decrease
    cout<<"obj_doc initial="<<obj_doc1<<endl;
    cout<<"Initial mubeta.block(0,0,5,5)="<<endl<<(*A.mubeta).block(0,0,5,5)<<endl;
    //cout<<"A.mubeta.block<2,2>(0,0)="<<(*A.mubeta).block<2,2>(0,0)<<endl;
    int a_b_converged=0;
    vector<vector<int>> usernos_diag(p),usernos_rlv(rlv.size());
#pragma omp parallel for
    for (i=0; i<p; ++i) {
        usernos_diag[i]=itemmore[i].user;
        if (itemmore_idle[i]!=-1) usernos_diag[i].push_back(itemmore_idle[i]);
    }
#pragma omp parallel for
    for (i=0; i<rlv.size(); ++i) {
       usernos_rlv[i]=rlv[i].userno;
        if (rlv_idle[i]!=-1) usernos_rlv[i].push_back(rlv_idle[i]);
    }
    vector<int> usernos=itemmore[i].user;
    while (doc_iter<C.maxIter){
        admm_iter=0;
#pragma omp parallel for
        for (i=0; i<n; ++i) {
            Omegaisdoc_pre[i]=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);
        }
        obj1=cal_obj_prop_struct_4(c,trin,trip,betadoc_pre,alphadoc_pre,Omegaisdoc_pre,Sis,A);//the function value admm needs to minimize, admm check its decrease
        tstart3=time(0);
        while(admm_iter<C.maxIter){//admm, for alpha, beta, Omega, use bcd
            tstart4=time(0);
            if (!a_b_converged){
                betapre=mubeta2;
                alphapre=mualpha2;
            }
            Omegaispre=Omegais2;
            if (!a_b_converged){
#pragma omp parallel for private(j,fdi,sdi,mtemp)
                for (i=0; i<n; ++i) {
                    tempB1[i]=VectorXd::Zero(mdim);
                    if(i<n-1){
                        fdi=fdibeta[i];
                        mtemp.resize(mdim,fdi.size());
                        for (j=0; j<fdi.size(); ++j) {
                            mtemp.col(j)=theta.col(fdi[j])-u.col(fdi[j]);
                        }
                        tempB1[i]=tempB1[i]+mtemp.rowwise().sum();
                    }
                    if(i>0){
                        sdi=sdibeta[i-1];
                        mtemp.resize(mdim,sdi.size());
                        for (j=0; j<sdi.size(); ++j) {
                            mtemp.col(j)=theta.col(sdi[j])-u.col(sdi[j]);
                        }
                        tempB1[i]=tempB1[i]-mtemp.rowwise().sum();
                    }
                    tempB1[i]*=rho_beta;
                    tempB1[i]+=(2*c*betadoc_pre.row(i));
                }
                
#pragma omp parallel for private(j,fdi,sdi,mtemp)
                for (i=0; i<p; ++i) {
                    tempA1[i]=VectorXd::Zero(udim);
                    if(i<p-1){
                        fdi=fdialpha[i];
                        mtemp.resize(udim,fdi.size());
                        for (j=0; j<fdi.size(); ++j) {
                            int tt=fdi[j];
                            mtemp.col(j)=theta2.col(tt)-u2.col(tt);
                        }
                        tempA1[i]+=mtemp.rowwise().sum();
                    }
                    if(i>0){
                        sdi=sdialpha[i-1];
                        mtemp.resize(udim,sdi.size());
                        for (j=0; j<sdi.size(); ++j) {
                            int tt=sdi[j];
                            mtemp.col(j)=theta2.col(tt)-u2.col(tt);
                        }
                        tempA1[i]-=mtemp.rowwise().sum();
                    }
                    tempA1[i]*=rho_alpha;
                    tempA1[i]+=(2*c*alphadoc_pre.col(i));
                }
            }
            iter_ori=0;
            while (iter_ori<C.maxIter) {
                if (!a_b_converged) {
                    MatrixXd mubeta_inner=mubeta2;
                    //update mubeta
                    mtempB.resize(n);
                    //construct mtempB and tempB for each user
#pragma omp parallel for private(movieuse,Omegaissub,alphause,useri)
                    for(i=0;i<n;++i){
                        movieuse=matrix_rowsub(C.movie,C.x.user[i].item);
                        Omegaissub=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);
                        mtempB[i]=movieuse.transpose()*Omegaissub*movieuse;
                        //inadmm, \eta=n*rho_beta*rt, checked~approx 0.02*100*0.7=1.4 for n=100
                        mtempB[i]=mtempB[i]+(rt*rho_beta*n+2*c)*MatrixXd::Identity(mdim,mdim);
                        
                        //if(i==10&&iter_ori==5){
                        //   cout<<"mtempB[i] after="<<endl;
                        //   cout<<mtempB[i]<<endl;
                        //}
                        LLT<MatrixXd> llt(mtempB[i]);
                        mtempB[i]=llt.solve(MatrixXd::Identity(mdim,mdim));
                        alphause=matrix_colsub(mualpha2,C.x.user[i].item);
                        useri=C.users.row(i);
                        tempB0[i]=C.x.user[i].rating-(alphause.transpose())*useri;
                        tempB0[i]=(movieuse.transpose())*Omegaissub*tempB0[i];
                    }
                    tempB=tempB0;
                    betasum=mubeta2.colwise().sum();
#pragma omp parallel for private(j,fdi,sdi,mtemp,atemp)
                    for (i=0; i<n; ++i) {
                        tempB[i]+=tempB1[i];
                        tempB[i]=tempB[i]+rho_beta*betasum;
                        atemp=(rt-1)*n*rho_beta*mubeta2.row(i);//inadmm, \eta=n*rho_beta*rt, checked
                        tempB[i]=tempB[i]+atemp;
                        mubeta2.row(i)=mtempB[i]*tempB[i];
                    }
                    maxbeta=absm(mubeta2-mubeta_inner).maxCoeff();
                    
                    //update mualpha
                    mtempB.resize(p);//only store matrix inverse for alpha
#pragma omp parallel for private(j,btemp)
                    for(i=0;i<p;++i){
                        mtempB[i]=(rho_alpha*(p-1)+2*c)*MatrixXd::Identity(udim,udim);
                        MatrixXd mtemp2(udim,itemmore[i].user.size());
                        MatrixXd mtemp3(udim,itemmore[i].user.size());
                        for (j=0; j<itemmore[i].user.size(); ++j) {
                            int userno=itemmore[i].user[j];
                            mtemp2.col(j)=C.users.row(userno);
                            mtemp3.col(j)=Omegais2[userno](i,i)*C.users.row(userno);
                        }
                        mtempB[i]=mtempB[i]+mtemp2*mtemp3.transpose();
                        LLT<MatrixXd> llt(mtempB[i]);
                        mtempB[i]=llt.solve(MatrixXd::Identity(udim,udim));
                    }
                    //cout<<"mtempB[0]="<<endl<<mtempB[0]<<endl;
                    maxalpha=0;
                    MatrixXd mualpha1_inner=mualpha2;
                    double maxalpha_inner=0;
                    int alpha_iter_inner=0;
                    //D.theta2=theta2;
                    //D.u2=u2;
                    while (alpha_iter_inner<C.maxIter) {//->1, was C.maxIter
                        //always unparallel for alpha
                        MatrixXd mualpha1_inner2=mualpha2;
                        for(i=0;i<p;++i){
                            temp=rho_alpha*mualpha2.rowwise().sum();
                            atemp=rho_alpha*mualpha2.col(i);
                            temp-=atemp;
                            temp+=tempA1[i];
                            //if(i==0) cout<<"temp p1="<<temp.transpose()<<endl;
                            mtemp=MatrixXd::Zero(udim,itemmore[i].user.size());
#pragma omp parallel for private(btemp)
                            for (j=0; j<itemmore[i].user.size(); ++j) {
                                int userno=itemmore[i].user[j];
                                int index=itemmore[i].numberforuser[j];
                                MatrixXd alphause=matrix_colsub(mualpha2,C.x.user[userno].item);//uses other alpha in this step, but can parallize, just the outer most cannot
                                MatrixXd movieuse=matrix_rowsub(C.movie,C.x.user[userno].item);
                                VectorXd useri=C.users.row(userno);
                                btemp=mubeta2.row(userno);
                                btemp=alphause.transpose()*useri+movieuse*btemp;
                                btemp[index]=(C.movie.row(i)).dot(mubeta2.row(userno));
                                btemp=C.x.user[userno].rating-btemp;
                                //cout<<"x.user[i].rating.size() is "<<x.user[i].rating.size()<<endl;
                                VectorXd Omega1row=Omegais2[userno].row(i);
                                VectorXd omega1row=Vec_sub(Omega1row,C.x.user[userno].item);
                                mtemp.col(j)=omega1row.dot(btemp)*C.users.row(userno);
                            }
                            //if(i==0) cout<<"mtemp="<<endl<<mtemp<<endl;
                            //if(i==0) cout<<"mtemp.rowwise().sum()="<<mtemp.rowwise().sum()<<endl;
                            temp+=mtemp.rowwise().sum();
                            //if(i==0) cout<<"temp="<<temp.transpose()<<endl;
                            mualpha2.col(i)=mtempB[i]*temp;
                            
                        }
                        //cout<<"mualpha2.block(0,0,5,5)="<<endl<<mualpha2.block(0,0,5,5)<<endl;
                        ++alpha_iter_inner;
                        maxalpha_inner=absm(mualpha2-mualpha1_inner2).maxCoeff();
                        //cout<<"maxalpha_inner="<<maxalpha_inner<<endl;
                        if (maxalpha_inner<1e-5/*C.Tol*/) {
                            //cout<<"maxalpha_inner="<<maxalpha_inner<<endl;
                            break;
                        }
                    }//? why <1
                    maxalpha=absm(mualpha2-mualpha1_inner).maxCoeff();
                }
                else{
                    maxalpha=0;
                    maxbeta=0;
                }
                //alpha, beta finished updating, next update Omega
                //Change the way Omegai is calculated, use quadratic approximation + coordinate wise change upate(two nested loops)
                //use surrogate variable for difference constraint and size constraint
                tstart1=time(0);
                MatrixXd solu=C.users*mualpha2+mubeta2*C.movie.transpose();
#pragma omp parallel for
                for (i=0; i<n; ++i) {
                    pretilde[i]=Vec_sub(solu.row(C.x.user[i].userno),C.x.user[i].item); //alphause.transpose()*useri+movieuse*btemp;
                    pretilde[i]=C.x.user[i].rating-pretilde[i];
                    Sis[i]=pretilde[i]*(pretilde[i].transpose());
                }
                
                VectorXd muAs_diag=VectorXd::Zero(p);
                VectorXd muAs_offdiag=VectorXd::Zero(rlv.size());//judge for stopping iter_between Omegasubi and OmegaAis
                VectorXd OmegaAs_diag_change=muAs_diag;//judge for stopping iter_ori
                VectorXd OmegaAs_offdiag_change=muAs_offdiag;
                cout<<"p="<<p<<", rlv.size()="<<rlv.size()<<endl;
#pragma omp parallel for private(j,k,l)
                for (i=0; i<p+rlv.size(); ++i) {
                    l=0;
                    if (i<p) {
                        vector<int> usernos=itemmore[i].user;
                        if (itemmore_idle[i]!=-1) usernos.push_back(itemmore_idle[i]);
                        for (j=0; j<usernos.size()-1; ++j) {
                            for (k=j+1; k<usernos.size(); ++k) {
                                diff_constr[i][l]=Omegais2[usernos[j]](i,i)-Omegais2[usernos[k]](i,i)-eta[i][l]+y[i][l];
                                ++l;
                            }
                        }
                    }
                    else{
                        int item1here=rlv[i-p].item1;int item2here=rlv[i-p].item2;
                        vector<int> usernos=rlv[i-p].userno;
                        if (rlv_idle[i-p]!=-1) usernos.push_back(rlv_idle[i-p]);
                        for (j=0; j<usernos.size(); ++j) {
                            for (k=j+1; k<usernos.size(); ++k) {
                                diff_constr[i][l]=Omegais2[usernos[j]](item1here,item2here)-Omegais2[usernos[k]](item1here,item2here)-eta[i][l]+y[i][l];
                                ++l;
                            }
                            size_constr[i-p][j]=Omegais2[usernos[j]](item1here,item2here)-X[i-p][j]+Z[i-p][j];
                        }
                    }
                    
                }
                int iter_between=0;
                OmegaAs_diag_change=VectorXd::Zero(p);
                OmegaAs_offdiag_change=VectorXd::Zero(rlv.size());
                while (iter_between<C.maxIter) {//for calculating Omega, two parts: part involved in likelihood, and part not involved in likelihood (As)
    #pragma omp parallel for private(j,k,sdi)
                    for (i=0; i<p; ++i) {
                        if (itemmore_idle[i]!=-1) {
                            sdi=constructsdi(size_est_diag[i], size_est_diag[i]);
                            for (k=0; k<sdi.size(); ++k) {
                                muAs_diag[i]+=diff_constr[i][sdi[k]];
                            }
                            muAs_diag[i]=muAs_diag[i]/(size_est_diag[i]-1);
                            OmegaAs_diag_change[i]+=muAs_diag[i];
                            for (k=0; k<sdi.size(); ++k) {
                                diff_constr[i][sdi[k]]-=muAs_diag[i];//update constraint involved
                            }
                        }
                    }
    #pragma omp parallel for private(j,k,sdi)
                    for (i=0; i<rlv.size(); ++i) {
                        if (rlv_idle[i]!=-1) {
                            sdi=constructsdi(size_est_offdiag[i], size_est_offdiag[i]);
                        }
                        for (k=0; k<sdi.size(); ++k) {
                            muAs_offdiag[i]+=diff_constr[i+p][sdi[k]];
                        }
                        muAs_offdiag[i]-=c6/c3*size_constr[i].back();//last element of size_constr[i]
                        muAs_offdiag[i]=muAs_offdiag[i]/(c6/c3+size_est_offdiag[i]-1);
                        OmegaAs_offdiag_change[i]+=muAs_offdiag[i];
                        int item1here=rlv[i].item1;int item2here=rlv[i].item2;
                        for (k=0; k<sdi.size(); ++k) {
                            diff_constr[i+p][sdi[k]]-=muAs_offdiag[i];//update constraint involved
                        }
                        size_constr[i].back()+=muAs_offdiag[i];
                    }
                    double change_As=max(muAs_diag.lpNorm<Infinity>(),muAs_offdiag.lpNorm<Infinity>());
                    //part related to OmegaAs are done, next do Omegaisub's with quadratic approximation
                    qua_iter=0;
                    vector<MatrixXd> Omegaissub_qua_pre=Omegaissub_qua;
                    while (qua_iter<C.maxIter) {//used to solve part involved in likelihood only
                        quasi_coor_iter=0;
#pragma omp parallel for private(j,k,l)
                        for (i=0; i<n; ++i) {
                            Dis_pre[i]=MatrixXd::Zero(sizeis[i],sizeis[i]);
                            //if(i==0) cout<<"Dis_pre[0]="<<endl<<Dis_pre[0]<<endl;
                            Ais[i]=Dis_pre[i];//Wis[i]*Dis_pre[i]*Wis[i];
                            Sis_denom[i]=4*c*MatrixXd::Ones(sizeis[i],sizeis[i]);
                            l=0;
                            for (j=0; j<sizeis[i]; ++j) {
                                Sis_denom[i](j,j)+=Wis[i](j,j)*Wis[i](j,j)+2*rho1_Omega*(itemmore[C.x.user[i].item[j]].user.size());
                                for (k=j+1; k<sizeis[i]; ++k) {
                                    Sis_denom[i](j,k)+=Wis[i](j,k)*Wis[i](j,k)+Wis[i](j,j)*Wis[i](k,k)+rho2_Omega+rho1_Omega*number_for_pair[i][l];
                                    Sis_denom[i](k,j)=Sis_denom[i](j,k);
                                    ++l;
                                }
                            }
                        }
                        change2.resize(n);
                        diff_constr_old=diff_constr;
                        size_constr_old=size_constr;
                        while (quasi_coor_iter<C.maxIter) {
//#pragma omp parallel for private(j,k), cannot parallize
                            for (i=0; i<p+rlv.size(); ++i) {
                                if (i<p) {
                                    vector<int> usernos=itemmore[i].user;
                                    for (j=0; j<usernos.size(); ++j) {
                                        int userj=usernos[j];
                                        int numberforuserj=itemmore[i].numberforuser[j];
                                        double constr_part=0;
                                        VectorXi fdj=VectorXi::LinSpaced(size_est_diag[i]-j-1,j*(size_est_diag[i]-(j+1)*0.5),j*(size_est_diag[i]-(j+1)*0.5)+size_est_diag[i]-j-2);//size, low, high
                                        VectorXi sdj;
                                        for (k=0; k<fdj.size(); ++k) {
                                            constr_part+=diff_constr[i][fdj[k]];
                                        }
                                        if(j>0){
                                            sdj=constructsdi(j+1, size_est_diag[i]);
                                            for (k=0; k<sdj.size(); ++k) {
                                                constr_part-=diff_constr[i][sdj[k]];
                                            }
                                        }
                                        muis[userj](numberforuserj,numberforuserj)=(-Sis[userj](numberforuserj,numberforuserj)+Wis[userj](numberforuserj,numberforuserj)-Ais[userj](numberforuserj,numberforuserj)-(2*rho1_Omega)*constr_part-4*c*(Omegaissub_qua[userj](numberforuserj,numberforuserj)+Dis_pre[userj](numberforuserj,numberforuserj)-Omegaisdoc_pre[userj](numberforuserj,numberforuserj)))/Sis_denom[userj](numberforuserj,numberforuserj);
                                        double temp=muis[userj](numberforuserj,numberforuserj);
                                        Dis_pre[userj](numberforuserj,numberforuserj)+=temp;
                                        Ais[userj]+=temp*Wis[userj].col(numberforuserj)*Wis[userj].row(numberforuserj);//cannot parallize because of this line
                                        for (k=0; k<fdj.size(); ++k) {
                                            diff_constr[i][fdj[k]]+=temp;
                                        }
                                        if (j>0) {
                                            for (k=0; k<sdj.size(); ++k) {
                                                diff_constr[i][sdj[k]]-=temp;
                                            }
                                        }
                                    }
                                }
                                else{//i>=p
                                    int item1here=rlv[i-p].item1;int item2here=rlv[i-p].item2;
                                    vector<int> usernos=rlv[i-p].userno;
                                    for (j=0; j<usernos.size(); ++j) {
                                        int userj=usernos[j];
                                        int in1=rlv[i-p].firstin[j];int in2=rlv[i-p].secin[j];
                                        double constr_part=0;
                                        VectorXi fdj=VectorXi::LinSpaced(size_est_offdiag[i-p]-j-1,j*(size_est_offdiag[i-p]-(j+1)*0.5),j*(size_est_offdiag[i-p]-(j+1)*0.5)+size_est_offdiag[i-p]-j-2);//size, low, high
                                        VectorXi sdj;
                                        for (k=0; k<fdj.size(); ++k) {
                                            constr_part+=diff_constr[i][fdj[k]];
                                        }
                                        if(j>0){
                                            sdj=constructsdi(j+1, size_est_offdiag[i-p]);
                                            for (k=0; k<sdi.size(); ++k) {
                                                constr_part-=diff_constr[i][sdj[k]];
                                            }
                                        }
                                        muis[userj](in1,in2)=(-Sis[userj](in1,in2)+Wis[userj](in1,in2)-Ais[userj](in1,in2)-rho1_Omega*constr_part
                                                              -4*c*(Omegaissub_qua[userj](in1,in2)+Dis_pre[userj](in1,in2)-Omegaisdoc_pre[userj](in1,in2))-rho2_Omega*size_constr[i-p][j])/Sis_denom[userj](in1,in2);
                                        double temp=muis[userj](in1,in2);
                                        muis[userj](in2,in1)=temp;
                                        Dis_pre[userj](in1,in2)+=temp;
                                        Dis_pre[userj](in2,in1)+=temp;
                                        Ais[userj]+=temp*Wis[userj].col(in1)*Wis[userj].row(in2)+temp*Wis[userj].col(in2)*Wis[userj].row(in1);
                                        for (k=0; k<fdj.size(); ++k) {
                                            diff_constr[i][fdj[k]]+=temp;
                                        }
                                        if (j>0) {
                                            for (k=0; k<sdj.size(); ++k) {
                                                diff_constr[i][sdj[k]]-=temp;
                                            }
                                        }
                                        size_constr[i-p][j]+=temp;
                                    }
                                    
                                }
                            }
#pragma omp parallel for
                            for (i=0; i<n; ++i) {
                                change2[i]=muis[i].lpNorm<Infinity>();
                            }
                            if(change2.maxCoeff()<1e-3) break;
                            ++quasi_coor_iter;
                        }
                        double constr_change_square=0, constr_change_prod=0,constr_change_square_temp=0, constr_change_prod_temp=0;
#pragma omp parallel for reduction(+:constr_change_square,constr_change_prod)
                        for (i=0; i<diff_constr.size(); ++i) {
                            vector<double> change_in_constr=vec_diff(diff_constr[i],diff_constr_old[i]);
                            constr_change_square+=vec_square_norm(change_in_constr);
                            constr_change_prod+=inner_product(change_in_constr.begin(), change_in_constr.end(), diff_constr_old[i].begin(), 0);
                        }
                        constr_change_square*=rho1_Omega;
                        constr_change_prod*=(2*rho1_Omega);
#pragma omp parallel for reduction(+:constr_change_square_temp,constr_change_prod_temp)
                        for (i=0; i<size_constr.size(); ++i) {
                            vector<double> change_in_constr=vec_diff(size_constr[i],size_constr_old[i]);
                            constr_change_square_temp+=vec_square_norm(change_in_constr);
                            constr_change_prod_temp+=inner_product(change_in_constr.begin(), change_in_constr.end(), size_constr_old[i].begin(), 0);
                        }
                        constr_change_square+=rho2_Omega*constr_change_square_temp;
                        constr_change_prod+=2*rho2_Omega*constr_change_prod_temp;
                        double part3=0,part5=0;
                        LLT<MatrixXd> llt;
                        for (i=0; i<n; ++i) {
                            llt.compute(Omegaissub_qua[i]);
                            double temp_determ=llt.matrixL().determinant();
                            part3+=2*log(temp_determ);//determinant of Omegaissub_qua[i]
                        }
                        //cout<<"Omegaissub_qua[i].rows()="<<Omegaissub_qua[i].rows()<<", Omegaissub_qua[i].cols()="<<Omegaissub_qua[i].cols()<<endl;
                        beta=1;
                        int ncheck=n;
                        while (ncheck>0) {
                            for (i=0; i<n; ++i) {
                                mtemp=Omegaissub_qua[i]+beta*Dis_pre[i];
                                llt.compute(mtemp);
                                if (llt.info()==NumericalIssue){
                                    ncheck=n;
                                    beta*=0.5;
                                    break;
                                }
                                else{
                                    ncheck-=1;
                                }
                            }
                        }
                        cout<<"here"<<endl;
                        double part1=(Dis_pre[i].array()*((1-sigma)*Sis[i]+sigma*Wis[i]).array()).sum()*beta;
                        cout<<"here"<<endl;
                        double part2=4*c*((Dis_pre[i].array()*(Omegaissub_qua[i]-Omegaisdoc_pre[i]).array()).sum());
                        double part4=Dis_pre[i].squaredNorm()*2*c;
                        part5=llt.matrixL().determinant();
                        part5=2*log(part5);
                        //double delta_t=part1/(1-sigma)-part2/sigma+part4;
                        //cout<<"delta_t="<<delta_t<<endl;
                        //cout<<"beta="<<beta<<endl;
                        //double Fdiff=beta*part1/(1-sigma)-log((Omegaissub_qua[i]+beta*Dis_pre[i]).determinant())+log((Omegaissub_qua[i]).determinant())+beta*beta*part4;
                        //cout<<"First Fdiff="<<Fdiff<<endl;
                        while (part1-(part5-part3)+(1-sigma*beta)*(part4+part2)+beta*(1-sigma)*constr_change_prod+beta*(beta-sigma)*constr_change_square>0){
                            beta=beta*0.5;
                            //Fdiff=beta*(part1/(1-sigma))-part5+part3+beta*beta*part4;
                            //cout<<"Fdiff="<<Fdiff<<endl;
                            //cout<<"beta*(part1+part2)-(part5-part3)+part4*beta*(beta-sigma)="<<beta*(part1+part2)-(part5-part3)+part4*beta*(beta-sigma)<<endl;
                            mtemp=Omegaissub_qua[i]+beta*Dis_pre[i];
                            llt.compute(mtemp);
                            part5=llt.matrixL().determinant();
                            part5=2*log(part5);
                        }
                        cout<<"here"<<endl;
                        //cout<<"beta="<<beta<<endl;
                        Omegaissub_qua[i]=mtemp;
                        Wis[i]=llt.solve(MatrixXd::Identity(sizeis[i],sizeis[i]));
                        //cout<<"Wis[i]="<<Wis[i]<<endl;
                        ++qua_iter;
                        //cout<<"qua_iter="<<qua_iter<<endl;
                        if (beta*Dis_pre[i].lpNorm<Infinity>()<1e-3) break;
                    }
#pragma omp parallel for private(mtemp)
                    for (i=0; i<n; ++i) {
                        mtemp=Omegaissub_qua_pre[i];
                        //matrix_rowcolsub_givevalue(Omegais2[i],C.x.user[i].item,Omegaissub_qua[i]);
                        change[i]=absm(mtemp-Omegaissub_qua[i]).maxCoeff();
                    }
                    tend1=time(0);
                    time1+=difftime(tend1, tstart1);
                    maxOmega=max(change_As,change.maxCoeff());
                    ++iter_between;
                    if (maxOmega<1e-3) break;
                }
                
                
#pragma omp parallel for private(mtemp)
                for (i=0; i<n; ++i) {
                    mtemp=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);
                    matrix_rowcolsub_givevalue(Omegais2[i],C.x.user[i].item,Omegaissub_qua[i]);//update Omegais2 observed part
                    change[i]=absm(mtemp-Omegaissub_qua[i]).maxCoeff();
                }
                double change_As=max(OmegaAs_diag_change.lpNorm<Infinity>(),OmegaAs_offdiag_change.lpNorm<Infinity>());
                tend1=time(0);
                time1+=difftime(tend1, tstart1);
                maxOmega=max(change_As,change.maxCoeff());
                
             //Update Omegais2 As
#pragma omp parallel for private(j,k)
                for (i=0; i<p; ++i) {
                    j=0;
                    for (k=0; k<n; ++k) {
                        if (k<usernos_diag[i][j]) {
                            Omegais2[k](i,i)+=muAs_diag[i];
                        }
                        else{
                            ++j;
                            if(j==size_est_diag[i]-2) break;
                        }
                    }
                    for (k=usernos_diag[i][size_est_diag[i]-2]+1; k<n; ++k) {
                        Omegais2[k](i,i)+=muAs_diag[i];//no need to update here
                    }
                }
#pragma omp parallel for private(j,k)
                for (i=0; i<rlv.size(); ++i) {
                    int item1here=rlv[i].item1;int item2here=rlv[i].item2;
                    j=0;
                    for (k=0; k<n; ++k) {
                        if (k<usernos_rlv[i][j]) {
                            Omegais2[k](item1here,item2here)+=muAs_offdiag[i];
                        }
                        else{
                            ++j;
                            if(j==size_est_offdiag[i]-2) break;
                        }
                    }
                    for (k=usernos_rlv[i][size_est_offdiag[i]-2]+1; k<n; ++k) {
                        Omegais2[k](item1here,item2here)+=muAs_offdiag[i];
                    }
                }
                //cout<<"Omega maximum change is at Omegai "<<pos<<endl;
                //cout<<"maxOmega="<<maxOmega<<endl;
                maxdiff_inner=max(max(maxalpha,maxbeta),maxOmega);
                ++iter_ori;
                //cout<<"iter_ori="<<iter_ori<<endl;
                if(maxdiff_inner<=1e-3) break;
            }
            
            //current Sis is available
            
            //cout<<"maxalpha="<<maxalpha<<endl;
            //cout<<"mualpha2.block(0,0,5,5)="<<endl<<mualpha2.block(0,0,5,5)<<endl;
            //cout<<"iter_ori="<<iter_ori<<endl;
            //cout<<"maxdiff_inner="<<maxdiff_inner<<endl;
            if (!a_b_converged){
                change.resize(theta.cols());
                change2.resize(change.size());
#pragma omp parallel for private(i,j,temp,atemp,btemp,a)
                for (l=0; l<(n-1)*n/2; ++l){
                    i=trin.first(l);
                    j=trin.second(l);
                    atemp=mubeta2.row(i)-mubeta2.row(j);
                    btemp=u.col(l);
                    atemp=atemp+btemp;
                    a=ST_vec(atemp,rho_ab);
                    atemp=a-theta.col(l);
                    theta.col(l)=a;
                    change[l]=atemp.lpNorm<Infinity>();
                    atemp=mubeta2.row(i)-mubeta2.row(j);
                    atemp=atemp-a;
                    change2[l]=atemp.lpNorm<Infinity>();
                    u.col(l)=u.col(l)+atemp;
                }
                maxtheta_beta=change.lpNorm<Infinity>();
                maxu_beta=change2.lpNorm<Infinity>();
                maxbeta=absm(mubeta2-betapre).maxCoeff();
                tttbeta=max(max(maxbeta,maxtheta_beta),maxu_beta);
                //cout<<"beta maxdiff="<<tttbeta<<endl;
                
                change.resize(theta2.cols());
                change2.resize(theta2.cols());
#pragma omp parallel for private(i,j,btemp,b)
                for (l=0; l<(p-1)*p/2; ++l){
                    i=trip.first(l);
                    j=trip.second(l);
                    btemp=mualpha2.col(i)-mualpha2.col(j)+u2.col(l);
                    b=ST_vec(btemp,rho_ab);
                    btemp=theta2.col(l);
                    theta2.col(l)=b;
                    btemp=btemp-b;
                    change[l]=btemp.lpNorm<Infinity>();
                    btemp=mualpha2.col(i)-mualpha2.col(j)-theta2.col(l);
                    change2[l]=btemp.lpNorm<Infinity>();
                    u2.col(l)=u2.col(l)+btemp;
                }
                maxtheta_alpha=change.lpNorm<Infinity>();
                maxu_alpha=change2.lpNorm<Infinity>();
                maxalpha=absm(mualpha2-alphapre).maxCoeff();
                //cout<<"maxalpha="<<maxalpha<<endl;
                tttalpha=max(max(maxalpha,maxtheta_alpha),maxu_alpha);
                //cout<<"alpha maxdiff is "<<tttalpha<<endl;
            }
            
            //update dual part related to Omega
            change.resize(p+rlv.size());
            change2.resize(p+rlv.size());
            VectorXd change3(rlv.size()),change4(rlv.size());
#pragma omp parallel for private(j,k,l)
            for (i=0; i<p+rlv.size(); ++i) {
                l=0;
                VectorXd change_eta(eta[i].size()),change_y(eta[i].size());
                if (i<p) {
                    vector<int> usernos=itemmore[i].user;
                    if (itemmore_idle[i]!=-1){
                        usernos.push_back(itemmore_idle[i]);
                        for (j=0; j<usernos.size()-1; ++j) {
                            for (k=j+1; k<usernos.size()-1; ++k) {
                                double temp=Omegais2[usernos[j]](i,i)-Omegais2[usernos[k]](i,i);
                                change_eta[l]=eta[i][l];
                                eta[i][l]=ST(temp+y[i][l],C.lambda3/rho2);
                                change_eta[l]=eta[i][l]-change_eta[l];
                                change_y[l]=(temp-eta[i][l]);
                                y[i][l]+=change_y[l];
                                diff_constr[i][l]+=change_y[l]-change_eta[l];
                                ++l;
                            }
                            double temp=Omegais2[usernos[j]](i,i)-Omegais2[usernos[k]](i,i);
                            change_eta[l]=eta[i][l];
                            eta[i][l]=ST(temp+y[i][l],(n-usernos.size()+1)*C.lambda3/rho2);
                            change_eta[l]=eta[i][l]-change_eta[l];
                            change_y[l]=(temp-eta[i][l]);
                            y[i][l]+=change_y[l];
                            diff_constr[i][l]+=change_y[l]-change_eta[l];
                            ++l;
                        }
                    }
                    else{
                        for (j=0; j<usernos.size()-1; ++j) {
                            for (k=j+1; k<usernos.size(); ++k) {
                                double temp=Omegais2[usernos[j]](i,i)-Omegais2[usernos[k]](i,i);
                                change_eta[l]=eta[i][l];
                                eta[i][l]=ST(temp+y[i][l],C.lambda3/rho2);
                                change_eta[l]=eta[i][l]-change_eta[l];
                                change_y[l]=temp-eta[i][l];
                                y[i][l]+=change_y[l];
                                diff_constr[i][l]+=change_y[l]-change_eta[l];
                                ++l;
                            }
                        }
                    }
                }
                else{
                    int item1here=rlv[i-p].item1;int item2here=rlv[i-p].item2;
                    vector<int> usernos=rlv[i-p].userno;
                    VectorXd change_X(usernos.size()),change_Z(usernos.size());
                    if (rlv_idle[i-p]!=-1){
                        usernos.push_back(rlv_idle[i-p]);
                        change_X.resize(usernos.size());
                        change_Z.resize(usernos.size());
                        for (j=0; j<usernos.size()-1; ++j) {
                            for (k=j+1; k<usernos.size()-1; ++k) {
                                double temp=Omegais2[usernos[j]](item1here,item2here)-Omegais2[usernos[k]](item1here,item2here);
                                change_eta[l]=eta[i][l];
                                eta[i][l]=ST(temp+y[i][l],C.lambda3/rho2);
                                change_eta[l]=eta[i][l]-change_eta[l];
                                change_y[l]=temp-eta[i][l];
                                y[i][l]+=change_y[l];
                                diff_constr[i][l]+=change_y[l]-change_eta[l];
                                ++l;
                            }
                            double temp=Omegais2[usernos[j]](item1here,item2here)-Omegais2[usernos[k]](item1here,item2here);
                            change_eta[l]=eta[i][l];
                            eta[i][l]=ST(temp+y[i][l],(n-usernos.size()+1)*C.lambda3/rho2);
                            change_eta[l]=abs(eta[i][l]-change_eta[l]);
                            change_y[l]=(temp-eta[i][l]);
                            y[i][l]+=change_y[l];
                            diff_constr[i][l]+=change_y[l]-change_eta[l];
                            ++l;
                            
                            change_X[j]=X[i][j];
                            X[i][j]=ST(Omegais2[usernos[j]](item1here,item2here)+Z[i][j],C.lambda2/rho2);
                            change_X[j]=X[i][j]-change_X[j];
                            change_Z[j]=Omegais2[usernos[j]](item1here,item2here)-X[i][j];
                            Z[i][j]+=change_Z[j];
                            size_constr[i-p][j]+=change_Z[j]-change_X[j];
                        }
                        change_X[j]=X[i][j];
                        X[i][j]=ST(Omegais2[usernos[j]](item1here,item2here)+Z[i][j],(n-usernos.size()+1)*C.lambda2/rho2);
                        change_X[j]=X[i][j]-change_X[j];
                        change_Z[j]=Omegais2[usernos[j]](item1here,item2here)-X[i][j];
                        Z[i][j]+=change_Z[j];
                        size_constr[i-p][j]+=change_Z[j]-change_X[j];
                    }
                    else{
                        for (j=0; j<usernos.size(); ++j) {
                            for (k=j+1; k<usernos.size(); ++k) {
                                double temp=Omegais2[usernos[j]](item1here,item2here)-Omegais2[usernos[k]](item1here,item2here);
                                change_eta[l]=eta[i][l];
                                eta[i][l]=ST(temp+y[i][l],C.lambda3/rho2);
                                change_eta[l]=eta[i][l]-change_eta[l];
                                change_y[l]=temp-eta[i][l];
                                y[i][l]+=change_y[l];
                                diff_constr[i][l]+=change_y[l]-change_eta[l];
                                ++l;
                            }
                            change_X[j]=X[i][j];
                            X[i][j]=ST(Omegais2[usernos[j]](item1here,item2here)+Z[i][j],C.lambda2/rho2);
                            change_X[j]=X[i][j]-change_X[j];
                            change_Z[j]=Omegais2[usernos[j]](item1here,item2here)-X[i][j];
                            Z[i][j]+=change_Z[j];
                            size_constr[i-p][j]+=change_Z[j]-change_X[j];
                        }
                    }
                    change3[i]=change_X.lpNorm<1>();
                    change4[i]=change_Z.lpNorm<1>();
                }
                change[i]=change_eta.lpNorm<1>();
                change2[i]=change_y.lpNorm<1>();
            }
            tttOmega=max(max(change.maxCoeff(),change2.maxCoeff()),max(change3.maxCoeff(),change4.maxCoeff()));//max of change1,2,3,4
            tttOmega=max(maxOmega,tttOmega);
            
            maxdiff_admm=max(max(tttbeta,tttalpha),tttOmega);
            A.mubeta=&mubeta2;
            A.mualpha=&mualpha2;
            A.Omegais=&Omegais2;
            A.off_diff=cal_offdiff_prop(trin,Omegais2);//this step is taking too much time
            
            tstart6=time(0);
            obj2=cal_obj_prop_struct_4(c,trin,trip,betadoc_pre,alphadoc_pre,Omegaisdoc_pre,Sis,A);
            tend6=time(0);
            time6+=difftime(tend6,tstart6);
            //string TF=(abs(obj1-obj2)<2e-4)?"TRUE":"FALSE";
            //cout<<"tttOmega="<<tttOmega<<", tttbeta="<<tttbeta<<", tttalpha="<<tttalpha<<", obj2="<<obj2<<", "<<TF<<endl;
            //cout<<"obj2="<<obj2<<endl;//just for admm
            ++admm_iter;
            //cout<<"mubeta2.block(0,0,5,5)="<<endl<<mubeta2.block(0,0,5,5)<<endl;
            //cout<<"iter is "<<admm_iter<<endl;
            //if(abs(obj1-obj2)<2e-4&&maxdiff_admm<1e-3/*&&iter_outer>200  5e-3 */) break;//for cold-start, this may be too strict, change to abs(obj1-obj2)/abs(obj1)<1e-4?//abs(obj1-obj2)<5e-3 still too strict? Change to abs(obj1-obj2)<0.01?or even 0.1?
            if(abs(obj1-obj2)<2e-4&&tttalpha<1e-3&&tttbeta<1e-3) break;
            obj1=obj2;
            tend5=time(0);
            time5+=difftime(tend5,tstart5);
        }
        tend3=time(0);
        time3+=difftime(tend3,tstart3);
        cout<<"Currently cost "<<time1<<" secs in eigenvalue decomp."<<endl;
        cout<<"Currently cost "<<time2<<" secs in solving Z."<<endl;
        cout<<"Current doc iter cost "<<time3<<" secs"<<endl;
        cout<<"Current alpha, beta cost "<<time4-time1-time2<<" secs"<<endl;
        cout<<"A final little part costs "<<time5<<" secs"<<endl;
        cout<<"A final little 2nd part costs "<<time6<<" secs"<<endl;
        cout<<"tttOmega="<<tttOmega<<", tttbeta="<<tttbeta<<", tttalpha="<<tttalpha<<endl;
        //current Sis is up to date
        obj_doc2=cal_obj_prop_struct_3(Sis,A);
        cout<<"admm took "<<admm_iter<<" iters"<<endl;
        cout<<"doc_iter="<<doc_iter<<", obj_doc2="<<obj_doc2<<endl;
        cout<<"Omegais[30].block(0,0,5,5)="<<endl<<Omegais2[30].block(0,0,10,10)<<endl;
        //cout<<"Omegais[30].row(1)="<<Omegais2[30].row(1)<<endl;
        //cout<<"Zis[30].row(1)="<<Zis[30].row(1)<<endl;
        //cout<<"Uis[30].row(1)="<<Uis[30].row(1)<<endl;
        cout<<"mubeta.block(0,0,5,5)="<<endl<<mubeta2.block(0,0,5,5)<<endl;
        maxbeta_doc=absm(mubeta2-betadoc_pre).maxCoeff();
        maxalpha_doc=absm(mualpha2-alphadoc_pre).maxCoeff();
        cout<<"maxdiff doc is "<<max(maxbeta_doc,maxalpha_doc)<<endl;
        if(max(maxbeta_doc,maxalpha_doc)<1e-8) a_b_converged=1;
        if (obj_doc1-obj_doc2<0.01) {//added two more conditions//&&doc_iter>10&&(obj_doc2<obj_doc1)
            break;
        }
        ++doc_iter;
        obj_doc1=obj_doc2;
        betadoc_pre=mubeta2;
        alphadoc_pre=mualpha2;
    }
    
    //second part
    //rt=1;
    //delete theta,u;delete theta2,u2; delete [] Zis;delete [] Uis;
    
    mtemp.resize(n,p);
    mtemp=C.users*mualpha2+mubeta2*C.movie.transpose();//compact multi is faster than one by one
    C.mualpha1=mualpha2;
    C.mubeta1=mubeta2;
    C.Omegais1=Omegais2;//necessary here
    C.theta=theta;
    C.theta2=theta2;
    C.u=u;
    C.u2=u2;
    C.eta=eta;
    C.y=y;
    C.Z=Z;
    C.X=X;
    result re;
    re.mualpha=mualpha2;
    re.mubeta=mubeta2;
    re.Omegais=Omegais2;
    re.solu=mtemp;
    //cout<<"MSE_train is "<<cal_MSE(mtemp,C.x)<<endl;
    re.maxdiff=maxdiff_admm;
    re.obj=obj_doc2;
    return re;
}//change C value


