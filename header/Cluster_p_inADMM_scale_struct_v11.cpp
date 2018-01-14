#include "consider_covariance.h"
//Difference from _v6, this changes the iterations inside iter_ori
result Cluster_p_inADMM_scale_struct_v11(double c,Cluster_mnl_p_ADMM_scale_para2 &C){
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
    vector<Omega_1rlv_simple> rlv=construct_rlv_simple(p,C.x.user);
    cout<<"rlv.size is "<<rlv.size()<<endl;
    vector<rated_1itemmore> itemmore=construct_itemmore(C.x);
    c6=(p+rlv.size())*n*(n-1)/2;
    
    //MatrixXi nrated=MatrixXi::Zero(p,p);
    for(int kk=0;kk<p;++kk){
        int usersize=itemmore[kk].user.size();
        cout<<"item "<<itemmore[kk].itemno<<" rated by "<<usersize<<" people \n";
    }
    for (int k=0; k<rlv.size();++k) {
        int usersize=rlv[k].userno.size();
        //cout<<"item1 "<<rlv[k].item1<<", item2 "<<rlv[k].item2<<" by "<<usersize<<endl;
        //nrated(rlv[k].item1,rlv[k].item2)=usersize;
        //nrated(rlv[k].item2,rlv[k].item1)=usersize;
    }
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
        //stl_vec_cout(itemmore[i].user);
        for (j=0; j<itemmore[i].user.size(); ++j) {
            if(j<itemmore[i].user[j]) break;
        }//result j is one user didn't rate this item
        itemmore_idle[i]=j;
        //cout<<"itemmore_idle["<<i<<"]="<<itemmore_idle[i]<<endl;
    }
#pragma omp parallel for private(l)
    for (i=0; i<rlv.size(); ++i) {
        for (l=0; l<rlv[i].userno.size(); ++l) {
            if(l<rlv[i].userno[l]) break;
        }//result l is one user didn't rate this pair
        rlv_idle[i]=l;
    }
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
    double rho_Omega=c1*rho2/c2;
    double rho_Zdiag=(c2*C.lambda3)/(2*rho2*c6);
    double rho_Z1=(c2*C.lambda2)/(2*rho2*c3);
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
    
    
    vector<MatrixXd> Zis=C.Zis;
    vector<MatrixXd> Uis=C.Uis;
    
    MatrixXd Omegais2sub;
    MatrixXd Omegasum;
    MatrixXd atempM,tempM,btempM;
    MatrixXd betadoc_pre=mubeta2,alphadoc_pre=mualpha2;
    vector<MatrixXd> Omegaisdoc_pre(n);//store sub-matrix directly
    vector<MatrixXd> Omegaissub_qua(n),Wis(n);
    vector<MatrixXd> Dis_pre(n),muis(n),Dis_diff(n),Ais(n),Zpis(n),Sis_denom(n);//Dis_pre: previous update direction for quadratic approximation; muis: change in update direction; Ais: WDW; Zpis: submatrix of Zis-Uis
    VectorXi ind1,ind2,sizeis(n);
    double beta=1,sigma=1e-2;
#pragma omp parallel for
    for (i=0; i<n; ++i) {
        Omegaissub_qua[i]=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);//Omegaissub in quadratic (t)
        sizeis[i]=Omegaissub_qua[i].rows();
        LLT<MatrixXd> llt(Omegaissub_qua[i]);
        Wis[i]=llt.solve(MatrixXd::Identity(sizeis[i],sizeis[i]));
        Zpis[i]=matrix_rowcolsub(Zis[i]-Uis[i],C.x.user[i].item);
    }
    
    //first part with rt!=1
    
    time_t tstart1,tend1,tstart2,tend2,tstart3,tend3,tstart4,tend4,tstart5,tend5,tstart6,tend6,tstart7,tend7;
    double time1=0,time2=0,time3=0,time4=0,time5=0,time6=0;
    
    obj_doc1=cal_obj_prop_struct_3(Sis,A);//the function value d.o.c. needs to minimize, d.o.c. check its decrease
    cout<<"obj_doc initial="<<obj_doc1<<endl;
    cout<<"Initial mubeta.block(0,0,5,5)="<<endl<<(*A.mubeta).block(0,0,5,5)<<endl;
    //cout<<"A.mubeta.block<2,2>(0,0)="<<(*A.mubeta).block<2,2>(0,0)<<endl;
    int a_b_converged=0;
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
                        Omegaissub=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);//C.Omegais1 changed to Omegais2, should be Omegais2
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
                            mtemp3.col(j)=Omegais2[userno](i,i)*C.users.row(userno);//C.Omegais1 changed to Omegais2, should be Omegais2
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
                                if (i==0&&j==1) {
                                    //cout<<"btemp="<<btemp.transpose()<<endl;
                                }
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
                tstart1=time(0);
                MatrixXd solu=C.users*mualpha2+mubeta2*C.movie.transpose();
#pragma omp parallel for private(j,k)
                for (i=0; i<n; ++i) {
                    pretilde[i]=Vec_sub(solu.row(C.x.user[i].userno),C.x.user[i].item); //alphause.transpose()*useri+movieuse*btemp;
                    pretilde[i]=C.x.user[i].rating-pretilde[i];
                    Sis[i]=pretilde[i]*(pretilde[i].transpose());
                    Dis_pre[i]=MatrixXd::Zero(sizeis[i],sizeis[i]);
                    Ais[i]=Dis_pre[i];//initial is 0
                    muis[i]=Dis_pre[i];
                    Sis_denom[i]=2*(rho_Omega+2*c)*MatrixXd::Ones(sizeis[i],sizeis[i]);
                    for (j=0; j<sizeis[i]; ++j) {
                        Sis_denom[i](j,j)+=Wis[i](j,j)*Wis[i](j,j);
                        for (k=j+1; k<sizeis[i]; ++k) {
                            Sis_denom[i](j,k)+=Wis[i](j,k)*Wis[i](j,k)+Wis[i](j,j)*Wis[i](k,k);
                            Sis_denom[i](k,j)=Sis_denom[i](j,k);
                        }
                    }
                }
#pragma omp parallel for private(j,k,l,change2,ind1,ind2,qua_iter, quasi_coor_iter,beta,mtemp)   //can only parallel for i, inside i are sequential
                for (i=0; i<n; ++i) {
                    tstart7=time(0);
                    qua_iter=0;
                    change2.resize(sizeis[i]*(sizeis[i]+1)/2);
                    ind1.resize(change2.size());
                    ind2.resize(ind1.size());
                    l=0;
                    for (j=0; j<sizeis[i]; ++j) {
                        for (k=j; k<sizeis[i]; ++k) {
                            ind1[l]=j;
                            ind2[l]=k;
                            ++l;
                        }
                    }
                    //cout<<"i="<<i<<endl;
                    while (qua_iter<C.maxIter) {
                        quasi_coor_iter=0;
                        if (qua_iter>0) {
                            Dis_pre[i]=MatrixXd::Zero(sizeis[i],sizeis[i]);
                            //if(i==0) cout<<"Dis_pre[0]="<<endl<<Dis_pre[0]<<endl;
                            Ais[i]=Dis_pre[i];//Wis[i]*Dis_pre[i]*Wis[i];
                        }
                        double qua_obj;
                        //cout<<"When Di=0, should be "<<rho_Omega*(Omegaissub_qua[i]-Zpis[i]).squaredNorm()+2*c*Omegaissub_qua[i].squaredNorm()-4*c*(Omegaissub_qua[i].array()*(Omegaisdoc_pre[i]).array()).sum()<<endl;
                        while(quasi_coor_iter<C.maxIter){
                            for (l=0; l<ind1.size(); ++l) {
                                //cout<<"Dis_pre[i]="<<endl<<Dis_pre[i]<<endl;
                                //qua_obj=((Sis[i]-Wis[i]).array()*Dis_pre[i].array()).sum()+0.5*(Wis[i]*Dis_pre[i]*Wis[i]*Dis_pre[i]).trace()+rho_Omega*(Omegaissub_qua[i]+Dis_pre[i]-Zpis[i]).squaredNorm()+2*c*(Omegaissub_qua[i]+Dis_pre[i]).squaredNorm()-4*c*((Omegaissub_qua[i]+Dis_pre[i]).array()*(Omegaisdoc_pre[i]).array()).sum();
                                //qua_obj_pre=0.5*(Wis[i]*Dis_pre[i]).squaredNorm();
                                j=ind1[l];k=ind2[l];
                                //cout<<"qua_obj_pre="<<qua_obj<<", will change j="<<j<<", k="<<k<<endl;
                                if (j==k) {
                                    muis[i](j,j)=(-Sis[i](j,j)+Wis[i](j,j)-Ais[i](j,j)-(2*rho_Omega+4*c)*(Omegaissub_qua[i](j,j)+Dis_pre[i](j,j))+2*rho_Omega*Zpis[i](j,j)+4*c*Omegaisdoc_pre[i](j,j))/Sis_denom[i](j,j);
                                    //muis[i](j,j)*muis[i](j,j)*0.5*Sis_denom[i](j,j)+muis[i](j,j)*(Ais[i](j,j)+Sis[i](j,j)-Wis[i](j,j)+(2*rho_Omega+4*c)*(Omegaissub_qua[i](j,j)+Dis_pre[i](j,j))-2*rho_Omega*Zpis[i](j,j)-4*c*Omegaisdoc_pre[i](j,j))<<endl;
                                    change2[l]=abs(muis[i](j,j));
                                    //cout<<"mu(j,k)="<<muis[i](j,j)<<", after change, must have obj change of "<<muis[i](j,j)*muis[i](j,j)*0.5*Sis_denom[i](j,j)+muis[i](j,j)*(Ais[i](j,j)+Sis[i](j,j)-Wis[i](j,j)+(2*rho_Omega+4*c)*(Omegaissub_qua[i](j,j)+Dis_pre[i](j,j))-2*rho_Omega*Zpis[i](j,j)-4*c*Omegaisdoc_pre[i](j,j))<<endl;
                                    Dis_pre[i](j,j)+=muis[i](j,j);
                                    Ais[i]+=muis[i](j,j)*Wis[i].col(j)*Wis[i].row(j);
                                }
                                else{
                                    muis[i](j,k)=(-Sis[i](j,k)+Wis[i](j,k)-Ais[i](j,k)-(2*rho_Omega+4*c)*(Omegaissub_qua[i](j,k)+Dis_pre[i](j,k))+2*rho_Omega*Zpis[i](j,k)+4*c*Omegaisdoc_pre[i](j,k))/Sis_denom[i](j,k);
                                    muis[i](k,j)=muis[i](j,k);
                                    change2[l]=abs(muis[i](j,k));
                                    Dis_pre[i](j,k)+=muis[i](j,k);
                                    Dis_pre[i](k,j)+=muis[i](k,j);
                                    Ais[i]+=muis[i](j,k)*Wis[i].col(j)*Wis[i].row(k)+muis[i](k,j)*Wis[i].col(k)*Wis[i].row(j);
                                }
                                //cout<<"Dis_pre[i]="<<endl<<Dis_pre[i]<<endl;
                                //cout<<"Wis[i]="<<Wis[i]<<endl;
                                //qua_obj=((Sis[i]-Wis[i]).array()*Dis_pre[i].array()).sum()+0.5*(Wis[i]*Dis_pre[i]*Wis[i]*Dis_pre[i]).trace()+rho_Omega*(Omegaissub_qua[i]+Dis_pre[i]-Zpis[i]).squaredNorm()+2*c*(Omegaissub_qua[i]+Dis_pre[i]).squaredNorm()-4*c*((Omegaissub_qua[i]+Dis_pre[i]).array()*(Omegaisdoc_pre[i]).array()).sum();
                                //((Sis[i]-Wis[i]).array()*Dis_pre[i].array()).sum()+0.5*(Wis[i]*Dis_pre[i]).squaredNorm()+rho_Omega*(Omegaissub_qua[i]+Dis_pre[i]-Zpis[i]).squaredNorm()+2*c*(Omegaissub_qua[i]+Dis_pre[i]).squaredNorm()-4*c*((Omegaissub_qua[i]+Dis_pre[i]).array()*(Omegaisdoc_pre[i]).array()).sum();
                                //cout<<"qua_obj="<<qua_obj<<", changed j="<<j<<", k="<<k<<endl;
                            }
                            //cout<<"quasi_coor_iter="<<quasi_coor_iter<<", maximum change is "<<change2.maxCoeff()<<endl;
                            ++quasi_coor_iter;
                            if (change2.maxCoeff()<1e-3) {
                                break;
                            }
                        }
                        //if(i==0) cout<<"Dis_pre[i]="<<endl<<Dis_pre[i]<<endl;
                        beta=1;
                        mtemp=Omegaissub_qua[i]+beta*Dis_pre[i];
                        LLT<MatrixXd> llt;
                        llt.compute(Omegaissub_qua[i]);
                        double part3=llt.matrixL().determinant();
                        part3=2*log(part3);//determinant of Omegaissub_qua[i]
                        llt.compute(mtemp);
                        while (llt.info() == NumericalIssue) {
                            beta=beta*0.5;
                            mtemp=Omegaissub_qua[i]+beta*Dis_pre[i];
                            llt.compute(mtemp);
                        }
                        double part1=(Dis_pre[i].array()*(Sis[i]+2*(rho_Omega+2*c)*Omegaissub_qua[i]-2*rho_Omega*Zpis[i]-4*c*Omegaisdoc_pre[i]).array()).sum()*(1-sigma);
                        double part2=(Dis_pre[i].array()*Wis[i].array()).sum()*sigma;
                        double part4=Dis_pre[i].squaredNorm()*(rho_Omega+2*c);
                        double part5=llt.matrixL().determinant();
                        part5=2*log(part5);
                        //double delta_t=part1/(1-sigma)-part2/sigma+part4;
                        //cout<<"delta_t="<<delta_t<<endl;
                        //cout<<"beta="<<beta<<endl;
                        //double Fdiff=beta*part1/(1-sigma)-log((Omegaissub_qua[i]+beta*Dis_pre[i]).determinant())+log((Omegaissub_qua[i]).determinant())+beta*beta*part4;
                        //cout<<"First Fdiff="<<Fdiff<<endl;
                        while (beta*(part1+part2)-(part5-part3)+part4*beta*(beta-sigma)>0){
                            beta=beta*0.5;
                            //Fdiff=beta*(part1/(1-sigma))-part5+part3+beta*beta*part4;
                            //cout<<"Fdiff="<<Fdiff<<endl;
                            //cout<<"beta*(part1+part2)-(part5-part3)+part4*beta*(beta-sigma)="<<beta*(part1+part2)-(part5-part3)+part4*beta*(beta-sigma)<<endl;
                            mtemp=Omegaissub_qua[i]+beta*Dis_pre[i];
                            llt.compute(mtemp);
                            part5=llt.matrixL().determinant();
                            part5=2*log(part5);
                        }
                        //cout<<"beta="<<beta<<endl;
                        Omegaissub_qua[i]=mtemp;
                        Wis[i]=llt.solve(MatrixXd::Identity(sizeis[i],sizeis[i]));
                        Sis_denom[i]=2*(rho_Omega+2*c)*MatrixXd::Ones(sizeis[i],sizeis[i]);
                        for (j=0; j<sizeis[i]; ++j) {
                            Sis_denom[i](j,j)+=Wis[i](j,j)*Wis[i](j,j);
                            for (k=j+1; k<sizeis[i]; ++k) {
                                Sis_denom[i](j,k)+=Wis[i](j,k)*Wis[i](j,k)+Wis[i](j,j)*Wis[i](k,k);
                                Sis_denom[i](k,j)=Sis_denom[i](j,k);
                            }
                        }
                        //cout<<"Wis[i]="<<Wis[i]<<endl;
                        ++qua_iter;
                        //cout<<"qua_iter="<<qua_iter<<endl;
                        if (absm(beta*Dis_pre[i]).maxCoeff()<1e-3) break;
                        //if(i==5) cout<<"For 5, absm(Dis_pre[i]).maxCoeff()="<<absm(beta*Dis_pre[i]).maxCoeff()<<endl;
                    }
                    tend7=time(0);
                }
#pragma omp parallel for private(mtemp)
                for (i=0; i<n; ++i) {
                    mtemp=Omegais2[i];
                    Omegais2[i]=Zis[i]-Uis[i];
                    matrix_rowcolsub_givevalue(Omegais2[i],C.x.user[i].item,Omegaissub_qua[i]);
                    change[i]=absm(mtemp-Omegais2[i]).maxCoeff();
                }
                tend1=time(0);
                time1+=difftime(tend1, tstart1);
                
                maxOmega=change.maxCoeff();
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
                    a=ST_vec(atemp,C.lambda1/(2*C.rho));
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
                    b=ST_vec(btemp,C.lambda1/(2*C.rho));
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
            //update Zis
            tstart2=time(0);
            off_diff=0;
            //diagonal elements, should also use solve21_ama
            change.resize(p);
#pragma omp parallel for private(j,l,atemp) reduction(+:off_diff)
            for(i=0;i<p;++i){
                int usersize=itemmore[i].user.size();//just assume no movie is rated by every person
                atemp.resize(usersize+1);
                VectorXd current(usersize+1);
                for (j=0; j<usersize; ++j) {
                    int userno=itemmore[i].user[j];
                    atemp[j]=Omegais2[userno](i,i)+Uis[userno](i,i);
                    current[j]=Zis[userno](i,i);
                }
                atemp[usersize]=Omegais2[itemmore_idle[i]](i,i)+Uis[itemmore_idle[i]](i,i);
                current[usersize]=Zis[itemmore_idle[i]](i,i);
                
                atemp=solve21_ama2_prop(atemp,n-usersize,rho_Zdiag,1000,1e-4);
                for (l=0; l<usersize; ++l) {
                    for (int tt=l+1; tt<usersize+1; ++tt) {
                        if(tt<usersize) off_diff=off_diff+abs(atemp[l]-atemp[tt]);
                        else off_diff=off_diff+(n-usersize)*abs(atemp[l]-atemp[tt]);
                    }
                }
                change[i]=(atemp-current).lpNorm<Infinity>();
                l=0;
                for (j=0; j<n; ++j) {
                    if (l<usersize) {
                        int userno=itemmore[i].user[l];
                        if (j<userno) {
                            Zis[j](i,i)=atemp[usersize];
                        }
                        else{
                            Zis[j](i,i)=atemp[l];
                            ++l;
                        }
                    }
                    else{
                        Zis[j](i,i)=atemp[usersize];
                    }
                }
            }
            maxZ=change.lpNorm<Infinity>();
            
            //off-diagonal elements
            change.resize(rlv.size());
#pragma omp parallel for private(i,j,l,ik,il,atemp,mtemp,temp) reduction(+:off_diff)
            for (int k=0; k<rlv.size();++k) {
                i=rlv[k].item1;
                j=rlv[k].item2;
                int usersize=rlv[k].userno.size();//just assume no pair is rated by every person
                atemp.resize(usersize+1);
                VectorXd current(usersize+1);
                for (l=0; l<usersize; ++l) {
                    int userno=rlv[k].userno[l];
                    atemp[l]=Omegais2[userno](i,j)+Uis[userno](i,j);
                    current[l]=Zis[userno](i,j);
                    if (l==0) {
                        //cout<<"Omegais2[userno](i,j)="<<Omegais2[userno](i,j)<<", Uis[userno](i,j)="<<Uis[userno](i,j)<<endl;
                        //cout<<"userno="<<userno<<endl;
                        //cout<<"Omegais2[userno]="<<endl<<Omegais2[userno]<<endl;
                    }
                }
                atemp[usersize]=Omegais2[rlv_idle[k]](i,j)+Uis[rlv_idle[k]](i,j);
                current[usersize]=Zis[rlv_idle[k]](i,j);
                //cout<<"atemp="<<atemp.transpose()<<endl;
                //cout<<"rho_Z1="<<rho_Z1<<endl;
                
                atemp=solve21_ama2_prop(atemp,n-usersize,rho_Zdiag/2,1000,1e-4);
                //cout<<"atemp="<<atemp.transpose()<<endl;
                atemp=ST_vec(atemp,rho_Z1);//shrink
                //cout<<"atemp="<<atemp.transpose()<<endl;
                for (l=0; l<usersize; ++l) {
                    for (int tt=l+1; tt<usersize+1; ++tt) {
                        if(tt<usersize) off_diff=off_diff+abs(atemp[l]-atemp[tt]);
                        else off_diff=off_diff+(n-usersize)*abs(atemp[l]-atemp[tt]);//calculated here
                    }
                }
                change[k]=(atemp-current).lpNorm<Infinity>();
                l=0;
                for (int tt=0; tt<n; ++tt) {
                    if (l<usersize) {
                        int userno=rlv[k].userno[l];
                        if (tt<userno) {
                            Zis[tt](i,j)=Zis[tt](j,i)=atemp[usersize];
                        }
                        else{
                            Zis[tt](i,j)=Zis[tt](j,i)=atemp[l];
                            ++l;
                        }
                    }
                    else{
                        Zis[tt](i,j)=Zis[tt](j,i)=atemp[usersize];
                    }
                }
            }
            maxZ=max(maxZ,change.lpNorm<Infinity>());
            tend2=time(0);
            time2+=difftime(tend2, tstart2);
            maxU=0;
            change.resize(n);
#pragma omp parallel for private(mtemp)
            for (i=0; i<n; ++i) {
                mtemp=Omegais2[i]-Zis[i];
                Uis[i]=Uis[i]+mtemp;
                change[i]=absm(mtemp).maxCoeff();
                Zpis[i]=matrix_rowcolsub(Zis[i]-Uis[i],C.x.user[i].item);
            }
            maxU=change.lpNorm<Infinity>();
            for (i=0; i<n; ++i) {
                change[i]=absm(Omegaispre[i]-Omegais2[i]).maxCoeff();
            }
            tend4=time(0);
            time4+=difftime(tend4,tstart4);
            
            tstart5=time(0);
            maxOmega=change.maxCoeff();
            tttOmega=max(max(maxOmega,maxZ),maxU);
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
    C.Zis=Zis;
    C.Uis=Uis;
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


