#include "consider_covariance.h"
result Cluster_TLP_scale_prop_size6(double c,Cluster_TLP_p_scale_para2 &C){//unparallel version
    int n=C.users.rows();
    int p=C.movie.rows();
    int udim=C.users.cols();
    int mdim=C.movie.cols();
    int c1=0,c2=0,c3=0,c6=0;
    int c4=n*(n-1)/2*mdim;
    int c5=p*(p-1)/2*udim;
    int i,j,l,k,kk,kkk,item1,item2,usersize,tt,userno,userno1,userno2,cstrno,index;
    for (kk=0; kk<n; ++kk) {
        kkk=C.x.user[kk].item.size();
        c1+=kkk;
        c2+=kkk*kkk;
    }
    c3=n*p*(p-1)/2;
    vector<Omega_1rlv_simple> rlv=construct_rlv_simple(p,C.x.user);
    cout<<"rlv.size is "<<rlv.size()<<endl;
    vector<rated_1itemmore> itemmore=construct_itemmore(C.x);
    c6=(p+rlv.size())*n*(n-1)/2;
    for(kk=0;kk<p;++kk){
        usersize=itemmore[kk].user.size();
    }
    for (k=0; k<rlv.size();++k) {
        usersize=rlv[k].userno.size();
    }
    vector<MatrixXd> Sis(n);  
    result re;
    MatrixXd &mualpha2=C.mualpha1;
    MatrixXd &mubeta2=C.mubeta1;
    vector<MatrixXd> &Omegais2=C.Omegais1;//C.Omegais1 is n p*p matrices, but only part of it is used.
    MatrixXd &theta=C.theta;
    MatrixXd &u=C.u;
    MatrixXd &theta2=C.theta2;
    MatrixXd &u2=C.u2;
    MatrixXd alphause, movieuse;
    VectorXd useri;
	vector<int> M;
	vector<int> user_items;
	vector<vector<Vector2i>> cstrfdi1,cstrsdi1;
	Vector2i pairtemp;
	VectorXd Omega1row,omega1row;
    
    double maxdiff_doc=1,maxdiff_admm=1,maxdiff_inner=1;
    double maxbeta=0,maxalpha=0,maxOmega=0,maxtheta_beta=0,maxu_beta=0,maxtheta_alpha=0,maxu_alpha=0,maxZ=0,maxU=0,tttalpha=0,tttbeta=0,tttOmega=0;
    
    int doc_iter=0,admm_iter=0,iter_ori=0;
    int ik,il;
    VectorXi fdi(1),sdi(1);
    VectorXd temp,atemp(2),btemp(2),a(2),b(2);
    MatrixXd mtemp(n,n),mtemp2,mtemp3;
    double obj1, obj2;
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
        for (j=0; j<itemmore[i].user.size(); ++j) {
            if(j<itemmore[i].user[j]) break;
        }
        itemmore_idle[i]=j;
    }
#pragma omp parallel for private(l)
    for (i=0; i<rlv.size(); ++i) {
        for (l=0; l<rlv[i].userno.size(); ++l){
            if(l<rlv[i].userno[l]) break;
        }//result l is one user didn't rate this pair
        rlv_idle[i]=l;
    }
#pragma omp parallel for private(alphause,movieuse,useri,btemp)
    for (i=0; i<n; ++i) {//construct pretilde
        alphause=matrix_colsub(mualpha2,C.x.user[i].item);
        movieuse=matrix_rowsub(C.movie,C.x.user[i].item);
        useri=C.users.row(i);
        btemp=mubeta2.row(i);
        pretilde[i]=alphause.transpose()*useri+movieuse*btemp;
        pretilde[i]=C.x.user[i].rating-pretilde[i];
        Sis[i]=pretilde[i]*(pretilde[i].transpose());
        if(i==1||i==25||i==70){
            cout<<"Initially error for user "<<i<<" is "<<pretilde[i].transpose()<<endl;
        }
    }
    VectorXd change,change2;
    TLP_obj_struct A,E;//A is for computing doc obj; E is for checking inner obj increasing, l2 norm
    A.mualpha=&C.mualpha1;
    A.mubeta=&C.mubeta1;
    A.Omegais=&C.Omegais1;
    A.lambda1=C.lambda1;
    A.lambda2=C.lambda2;
    A.lambda3=C.lambda3;
    A.tau=C.tau;
    A.user=&C.x.user;  
    A.c1=c1;
    A.c2=c2;
    A.c3=c3;
    A.c4=c4;
    A.c5=c5;
    A.c6=c6;
    A.log_det=VectorXd::Zero(n);
    
    doc_obj_struct B;//for abs obj
    B.c1=c1;
    B.c2=c2;
    B.c3=c3;
    B.c4=c4;
    B.c5=c5;
    B.c6=c6;
    B.lambda1=C.lambda1;
    B.lambda2=C.lambda2;
    B.lambda3=C.lambda3;//should be C.lambda
    B.tau=C.tau;
    B.user=A.user;
    B.log_det=&A.log_det;
    
    double rho2=C.rho2;//for rho_Omega
    double rho_beta=c1*C.rho/(C.tau*c4);
    cout<<"rho_beta="<<rho_beta<<endl;
    double rho_alpha=c1*C.rho/(C.tau*c5);
    double rho_Omega=c1*rho2/c2;//
    double rho_Zdiag=(c2*C.lambda3)/(2*rho2*c6*C.tau);
    double rho_Z1=(c2*C.lambda2)/(2*rho2*c3*C.tau);
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
    
    MatrixXd Omegaissub,Omegais2sub,betapre,alphapre,alphadoc, betadoc;
    vector<MatrixXd> mtempB(n),Omegaispre(n),Omegaisdoc(n);
    vector<MatrixXd> &Zis=C.Zis;
    vector<MatrixXd> &Uis=C.Uis;
    vector<VectorXd> tempB0(n),tempB(n),tempB1(n),tempA(p),tempA1(p);
    MatrixXd mtemp0;
    double obj_abs1,obj_abs2;
    double obj_inner;
    vector<vector<int>> Osize(rlv.size());//can define Osize here
    VectorXi involvedB(n),involvedA(p);
    time_t tstart1,tend1;
    double time1;
    vector<vector<int>> diffA(p*(p-1)/2),diffB(n*(n-1)/2);
    vector<vector<vector<Vector2i>>> cstrfdi_diag(p),cstrsdi_diag(p),cstrfdi_offdiag(rlv.size()),cstrsdi_offdiag(rlv.size());//for constraints, for diagonal and off
    vector<int> judge_diag(p,0), judge_offdiag(rlv.size(),0);
    double alphafuse_tau=0,betafuse_tau=0,Omegasize_tau=0,Omegafuse_tau=0;
    
    tstart1=time(0);
	involvedB=VectorXi::Zero(n);//construct diffB and calculate betafuse_tau
	for(i=0;i<n-1;++i){//diffB is in order 12,13,14,...1n; 23,24,25...2n;...(n-1)n
		for (j=i+1; j<n; ++j) {
			l=i*(n-(i+1)*0.5)+j-i-1;
			diffB[l].clear();
			vector<int> diff1;
			for (int tt=0; tt<mdim; ++tt) {
				double beta2_diff=abs(mubeta2(i,tt)-mubeta2(j,tt));
				if (beta2_diff<=C.tau) {
					diff1.push_back(tt);
					betafuse_tau+=beta2_diff;
				}
			}
			betafuse_tau+=(mdim-diff1.size())*C.tau;
			if(diff1.size()>0){
				diffB[l]=diff1;
				++involvedB[i];++involvedB[j];//cannot parallize because of this line
			}
		}
	}   
	involvedA=VectorXi::Zero(p);
	for(i=0;i<p-1;++i){
		for (j=i+1; j<p; ++j) {
			l=i*(p-(i+1)*0.5)+j-i-1;
			diffA[l].clear();
			vector<int> diff1;
			for (int tt=0; tt<udim; ++tt) {
				double alpha2_diff=abs(mualpha2(tt,i)-mualpha2(tt,j));
				if (alpha2_diff<=C.tau) {
					diff1.push_back(tt);
					alphafuse_tau+=alpha2_diff;
				}
			}
			alphafuse_tau+=(udim-diff1.size())*C.tau;
			if(diff1.size()>0){
				diffA[l]=diff1;
				++involvedA[i];++involvedA[j];
			}
		}
	}
		
	//construct Osize and calculate Omegasize_tau;
#pragma omp parallel for private(item1,item2,usersize,M,tt,userno) reduction(+:Omegasize_tau)
	for(i=0;i<rlv.size();++i){
		item1=rlv[i].item1;
		item2=rlv[i].item2;
		usersize=rlv[i].userno.size();
		M.clear();
		for (tt=0; tt<usersize; ++tt) {
			userno=rlv[i].userno[tt];
			double Osize_temp=abs(Omegais2[userno](item1,item2));
			if(Osize_temp<=C.tau){
				M.push_back(tt);//just push back index in rlv[i],tt is 第tt个user
				Omegasize_tau+=Osize_temp;
			}
		}
		Omegasize_tau+=(usersize-M.size())*C.tau;
		tt=usersize;
		userno=rlv_idle[i];
		double Osize_temp=abs(Omegais2[userno](item1,item2));
		if(Osize_temp<=C.tau){
			M.push_back(tt);
			Omegasize_tau+=Osize_temp*(n-usersize);
		}
		else{
			Omegasize_tau+=C.tau*(n-usersize);
		}
		Osize[i]=M;
	}
		
#pragma omp parallel for private(l,usersize,userno1,userno2,cstrfdi1,cstrsdi1,cstrno,tt,pairtemp) reduction(+:Omegafuse_tau)
	for (i=0; i<p; ++i) {
		cstrfdi1.clear();cstrsdi1.clear();
		judge_diag[i]=0;
		usersize=itemmore[i].user.size();
		if (usersize==1) {
			//because of the push_back(), cannot parallelize
			userno1=itemmore[i].user[0];
			userno2=itemmore_idle[i];
			double Ofuse_temp=abs(Omegais2[userno1](i,i)-Omegais2[userno2](i,i));
			if (Ofuse_temp<=C.tau){
				cstrfdi1.resize(2);cstrsdi1.resize(2);
				pairtemp(1)=0;
				pairtemp(0)=1;
				cstrfdi1[0].push_back(pairtemp);
				pairtemp(0)=0;
				cstrsdi1[1].push_back(pairtemp);
				cstrfdi_diag[i]=cstrfdi1;
				cstrsdi_diag[i]=cstrsdi1;
				judge_diag[i]=1;
				Omegafuse_tau+=(n-1)*Ofuse_temp;
			}
			else{
				Omegafuse_tau+=(n-1)*C.tau;
			}
		}
		if (usersize>1){
			cstrfdi1.resize(usersize+1),cstrsdi1.resize(usersize+1);
			cstrno=0;
			for (l=0; l<usersize; ++l) {
				userno1=itemmore[i].user[l];
				for (tt=l+1; tt<usersize; ++tt) {
					userno2=itemmore[i].user[tt];
					double Ofuse_temp=abs(Omegais2[userno1](i,i)-Omegais2[userno2](i,i));
					if (Ofuse_temp<=C.tau){
						pairtemp(1)=cstrno;
						pairtemp(0)=tt;
						cstrfdi1[l].push_back(pairtemp);//last of cstrfdi1 is never used, as of first of cstrsdi1
						pairtemp(0)=l;
						cstrsdi1[tt].push_back(pairtemp);
						++cstrno;
						Omegafuse_tau+=Ofuse_temp;
					}
					else{
						Omegafuse_tau+=C.tau;
					}
				}
				tt=usersize;
				userno2=itemmore_idle[i];
				double Ofuse_temp=abs(Omegais2[userno1](i,i)-Omegais2[userno2](i,i));
				if (Ofuse_temp<=C.tau){
					pairtemp(1)=cstrno;
					pairtemp(0)=tt;
					cstrfdi1[l].push_back(pairtemp);//last of cstrfdi1 is never used, as of first of cstrsdi1
					pairtemp(0)=l;
					cstrsdi1[tt].push_back(pairtemp);
					++cstrno;
					Omegafuse_tau+=(n-usersize)*Ofuse_temp;
				}
				else{
					Omegafuse_tau+=(n-usersize)*C.tau;
				}
			}
			cstrfdi_diag[i]=cstrfdi1;
			cstrsdi_diag[i]=cstrsdi1;
			judge_diag[i]=cstrno;
		}
	} 
		
#pragma omp parallel for private(l,usersize,i,j,userno1,userno2,cstrfdi1,cstrsdi1,cstrno,tt,pairtemp) reduction(+:Omegafuse_tau)
	for (k=0; k<rlv.size();++k) {
		usersize=rlv[k].userno.size();
		cstrfdi1.clear();cstrsdi1.clear();
		judge_offdiag[k]=0;
		i=rlv[k].item1;
		j=rlv[k].item2;
		if (usersize==1) {
			userno1=rlv[k].userno[0];
			userno2=rlv_idle[k];
			double Ofuse_temp=abs(Omegais2[userno1](i,j)-Omegais2[userno2](i,j));
			if(Ofuse_temp<=C.tau){
				cstrfdi1.resize(2);cstrsdi1.resize(2);
				Vector2i pairtemp;
				pairtemp(1)=0;
				pairtemp(0)=1;
				cstrfdi1[0].push_back(pairtemp);
				pairtemp(0)=0;
				cstrsdi1[1].push_back(pairtemp);
				cstrfdi_offdiag[k]=cstrfdi1;
				cstrsdi_offdiag[k]=cstrsdi1;
				judge_offdiag[k]=1;
				Omegafuse_tau+=(n-1)*Ofuse_temp;
			}
			else{
				Omegafuse_tau+=(n-1)*C.tau;
			}
		}
		if(usersize>1){
			cstrfdi1.resize(usersize+1);cstrsdi1.resize(usersize+1);
			cstrno=0;
			for (l=0; l<usersize; ++l) {//l should be private
				userno1=rlv[k].userno[l];
				for (tt=l+1; tt<usersize; ++tt) {
					userno2=rlv[k].userno[tt];
					double Ofuse_temp=abs(Omegais2[userno1](i,j)-Omegais2[userno2](i,j));
					if(Ofuse_temp<=C.tau){
						pairtemp(1)=cstrno;
						pairtemp(0)=tt;
						cstrfdi1[l].push_back(pairtemp);
						pairtemp(0)=l;
						cstrsdi1[tt].push_back(pairtemp);
						++cstrno;
						Omegafuse_tau+=Ofuse_temp;
					}
					else{
						Omegafuse_tau+=C.tau;
					}
				}
				tt=usersize;
				userno2=rlv_idle[k];
				double Ofuse_temp=abs(Omegais2[userno1](i,j)-Omegais2[userno2](i,j));
				if(Ofuse_temp<=C.tau){
					pairtemp(1)=cstrno;
					pairtemp(0)=tt;
					cstrfdi1[l].push_back(pairtemp);
					pairtemp(0)=l;
					cstrsdi1[tt].push_back(pairtemp);
					++cstrno;
					Omegafuse_tau+=(n-usersize)*Ofuse_temp;
				}
				else{
					Omegafuse_tau+=(n-usersize)*C.tau;
				}
			}
			cstrfdi_offdiag[k]=cstrfdi1;
			cstrsdi_offdiag[k]=cstrsdi1;
			judge_offdiag[k]=cstrno;
		}
	}
	tend1=time(0);
	time1=difftime(tend1,tstart1);
	cout<<"Constructing diffA, diffB, diffOmega cost "<<time1<<" seconds."<<endl;
	
	tstart1=time(0);
	cout<<"Omegafuse_tau="<<Omegafuse_tau<<endl;
	obj1=cal_TLP_obj_prop_size_wi(n,A.c1,A.c3,A.c4,A.c5,A.c6,A.lambda1,A.lambda2,A.lambda3,A.tau,alphafuse_tau,betafuse_tau,Omegasize_tau,Omegafuse_tau, A.log_det,
	  Sis,C.x.user,Omegais2); 
	tend1=time(0);
	time1=difftime(tend1,tstart1);
	cout<<"Calculating obj_doc cost "<<time1<<" seconds."<<endl;
    cout<<"Omegafuse_tau="<<Omegafuse_tau<<", obj initial="<<obj1<<endl;
    
    
    while (doc_iter<C.maxIter){
    	betadoc=mubeta2;
        alphadoc=mualpha2;
        Omegaisdoc=Omegais2;
        admm_iter=0;
        B.alphafuse=cal_alpha_fuse_abs(trip,mualpha2,diffA);
        B.betafuse=cal_beta_fuse_abs(trin,mubeta2,diffB); 
        B.Omegafusesize=cal_Omega_fusesize_abs(Omegais2,cstrfdi_diag,cstrsdi_diag,cstrfdi_offdiag,cstrsdi_offdiag,judge_diag,judge_offdiag,Osize,rlv,itemmore,itemmore_idle,rlv_idle);
        B.mualpha=&mualpha2;
        B.mubeta=&mubeta2;
        B.Omegais=&Omegais2;
        obj_abs1=cal_doc_obj_struct_prop_abs2(c,betadoc,alphadoc,Omegaisdoc,Sis,B);
        while (admm_iter<3000) {
            betapre=mubeta2;
            alphapre=mualpha2;
            Omegaispre=Omegais2;
#pragma omp parallel for private(j,fdi,sdi,mtemp)
            for (i=0; i<n; ++i) {
                tempB1[i]=VectorXd::Zero(mdim);
                if(i<n-1){
                    fdi=fdibeta[i];
                    mtemp=MatrixXd::Zero(mdim,fdi.size());
                    for (j=0; j<fdi.size(); ++j) {
                        if(diffB[fdi[j]].size()>0) mtemp.col(j)=theta.col(fdi[j])-u.col(fdi[j]);
                    }
                    tempB1[i]+=mtemp.rowwise().sum();
                }
                if(i>0){
                    sdi=sdibeta[i-1];
                    mtemp=MatrixXd::Zero(mdim,sdi.size());//didn't set to 0 before!!!, must set to 0, just resize doesn't know its real value
                    for (j=0; j<sdi.size(); ++j) {
                        if(diffB[sdi[j]].size()>0) mtemp.col(j)=theta.col(sdi[j])-u.col(sdi[j]);
                    }
                    tempB1[i]-=mtemp.rowwise().sum();
                }
                tempB1[i]*=rho_beta;
                tempB1[i]+=(2*c*betadoc.row(i));//can add variable here?
            }
#pragma omp parallel for private(j,fdi,sdi,mtemp)
            for (i=0; i<p; ++i) {
                tempA1[i]=VectorXd::Zero(udim);
                if(i<p-1){
                    fdi=fdialpha[i];
                    mtemp=MatrixXd::Zero(udim,fdi.size());//didn't set to 0 before!!!, must set to 0, just resize doesn't know its real value
                    for (j=0; j<fdi.size(); ++j) {
                        int tt=fdi[j];
                        if(diffA[tt].size()>0) mtemp.col(j)=theta2.col(tt)-u2.col(tt);
                    }
                    tempA1[i]+=mtemp.rowwise().sum();
                }
                if(i>0){
                    sdi=sdialpha[i-1];
                    mtemp=MatrixXd::Zero(udim,sdi.size());//didn't set to 0 before!!!, must set to 0, just resize doesn't know its real value
                    for (j=0; j<sdi.size(); ++j) {
                        int tt=sdi[j];
                        if(diffA[tt].size()>0) mtemp.col(j)=theta2.col(tt)-u2.col(tt);
                    }
                    tempA1[i]-=mtemp.rowwise().sum();
                }
                tempA1[i]*=rho_alpha;
                tempA1[i]+=(2*c*alphadoc.col(i));
            }
            iter_ori=0;//iter for alpha, beta and Omega (original variable)
            while (iter_ori<1) {//<=1 is running two iter_ori iterations.  
                mtempB.resize(n);
#pragma omp parallel for private(movieuse,Omegaissub,alphause,useri)
                for(i=0;i<n;++i){
                    movieuse=matrix_rowsub(C.movie,C.x.user[i].item);
                    Omegaissub=matrix_rowcolsub(Omegais2[i],C.x.user[i].item);//shouldn't use C.Omegais1
                    mtempB[i]=movieuse.transpose()*Omegaissub*movieuse;
                    mtempB[i]=mtempB[i]+(rho_beta*involvedB[i]+2*c)*MatrixXd::Identity(mdim,mdim);//can add variable here?
                    LLT<MatrixXd> llt(mtempB[i]);
					mtempB[i]=llt.solve(MatrixXd::Identity(mdim,mdim));
                    alphause=matrix_colsub(mualpha2,C.x.user[i].item);//shouldn't use C.mualpha1
                    useri=C.users.row(i);
                    tempB0[i]=C.x.user[i].rating-(alphause.transpose())*useri;
                    tempB0[i]=(movieuse.transpose())*Omegaissub*tempB0[i];
                    tempB0[i]+=tempB1[i];//combine adding tempB0 and tempB1,must be time-saving
                }
                double maxbeta_inner=0;
                int beta_iter_inner=0;
                MatrixXd mubeta_inner=mubeta2;
                while (beta_iter_inner<10) {//C.maxIter,unparallel
                    MatrixXd mubeta1_inner2=mubeta2;
                    for(i=0;i<n;++i){
                        mtemp0=MatrixXd::Zero(n,mdim);
#pragma omp parallel for private(l)
                        for (j=0; j<n; ++j) {
                            if(j==i) continue;
                            if (j<i) l=j*(n-(j+1)*0.5)+i-j-1;
                            else l=i*(n-(i+1)*0.5)+j-i-1;
                            if (diffB[l].size()>0) {
                                mtemp0.row(j)= mubeta2.row(j);
                            }
                        }
                        tempB[i]=mtemp0.colwise().sum();
                        tempB[i]*=rho_beta;
                        tempB[i]+=tempB0[i];//combine adding tempB0 and tempB1
                        mubeta2.row(i)=mtempB[i]*tempB[i];
                    }
                    maxbeta_inner=absm(mubeta2-mubeta1_inner2).maxCoeff();
                    ++beta_iter_inner;
                    if (maxbeta_inner<C.Tol) {
                        break;
                    }
                }
                MatrixXd::Index maxRow, maxCol;
            	maxbeta=absm(mubeta2-mubeta_inner).maxCoeff(&maxRow, &maxCol);
            	cout<<"max change of mubeta occurs at row "<<maxRow<<", col "<<maxCol<<", from "
  				 <<mubeta_inner(maxRow,maxCol)<<" to "<<mubeta2(maxRow,maxCol)<<"."<<endl;
                
                mtempB.resize(p);//only store matrix inverse for alpha
#pragma omp parallel for private(j,mtemp2,mtemp3,userno)
                for(i=0;i<p;++i){
                    mtempB[i]=(rho_alpha*involvedA[i]+2*c)*MatrixXd::Identity(udim,udim);//can add variable here?
                    mtemp2.resize(udim,itemmore[i].user.size());
                    mtemp3.resize(udim,itemmore[i].user.size());
                    for (j=0; j<itemmore[i].user.size(); ++j) {
                        userno=itemmore[i].user[j];
                        mtemp2.col(j)=C.users.row(userno);
                        mtemp3.col(j)=Omegais2[userno](i,i)*mtemp2.col(j);//shouldn't use C.Omegais1
                    }
                    mtempB[i]=mtempB[i]+mtemp2*mtemp3.transpose();
                    LLT<MatrixXd> llt(mtempB[i]);
					mtempB[i]=llt.solve(MatrixXd::Identity(udim,udim));
                }
                MatrixXd mualpha1_inner=mualpha2;  
                double maxalpha_inner=0;
                int alpha_iter_inner=0;
                while (alpha_iter_inner<10) {
                    MatrixXd mualpha1_inner2=mualpha2;
                    for(i=0;i<p;++i){
                        mtemp0=MatrixXd::Zero(udim,p);
#pragma omp parallel for private(j,l)
                        for (j=0; j<p; ++j) {
                            if(j==i) continue;
                            if (j<i) l=j*(p-(j+1)*0.5)+i-j-1;//previously here is wrong without considering l correctly
                            else l=i*(p-(i+1)*0.5)+j-i-1;
                            if (diffA[l].size()>0) mtemp0.col(j)=mualpha2.col(j);
                        }
                        temp=mtemp0.rowwise().sum();
                        temp*=rho_alpha;
                        temp+=tempA1[i];
                        mtemp=MatrixXd::Zero(udim,itemmore[i].user.size());
#pragma omp parallel for private(btemp,userno,index,user_items,alphause,movieuse,useri,Omega1row,omega1row)
                        for (j=0; j<itemmore[i].user.size(); ++j) {
                            userno=itemmore[i].user[j];
                            index=itemmore[i].numberforuser[j];
                            user_items=C.x.user[userno].item;
                            alphause=matrix_colsub(mualpha2,user_items);//uses other alpha in this step, but can parallize, just the outer most cannot
                            movieuse=matrix_rowsub(C.movie,user_items);
                            useri=C.users.row(userno);
                            btemp=mubeta2.row(userno);
                            btemp=alphause.transpose()*useri+movieuse*btemp;
                            btemp[index]=(C.movie.row(i)).dot(mubeta2.row(userno));
                            btemp=C.x.user[userno].rating-btemp;
                            Omega1row=Omegais2[userno].row(i);//shouldn't use C.Omegais1!!!
                            omega1row=Vec_sub(Omega1row,C.x.user[userno].item);
                            mtemp.col(j)=omega1row.dot(btemp)*C.users.row(userno);
                        }
                        temp+=mtemp.rowwise().sum();
                        mualpha2.col(i)=mtempB[i]*temp;
                        /*if(obj_inner>obj_inner0){
                         cout<<"Stopping here, obj is increasing!"<<endl;
                         cout<<"involvedA[i]="<<involvedA[i]<<endl;
                         cout<<"alpha_iter_inner="<<alpha_iter_inner<<endl;
                         mtemp.resize(n,p);
                         mtemp=C.users*mualpha2+mubeta2*C.movie.transpose();//compact multi is faster than one by one
                         re.mualpha=mualpha2;
                         re.mubeta=mubeta2;
                         re.Omegais=Omegais2;
                         re.solu=mtemp;
                         re.maxdiff=maxdiff_doc;//should change here
                         re.obj=obj2;
                         return re;
                         }
                         obj_inner0=obj_inner;*/
                    }
                    ++alpha_iter_inner;
                    maxalpha_inner=absm(mualpha2-mualpha1_inner2).maxCoeff();
                    if (maxalpha_inner<1e-5/*C.Tol*/) {
                        break;
                    }
                }
                maxalpha=absm(mualpha2-mualpha1_inner).maxCoeff();
                
                
#pragma omp parallel for private(btemp,alphause,movieuse,useri)
                for (i=0; i<n; ++i) {//construct WS
                    alphause=matrix_colsub(mualpha2,C.x.user[i].item);  
                    movieuse=matrix_rowsub(C.movie,C.x.user[i].item);
                    useri=C.users.row(i);
                    btemp=mubeta2.row(i);
                    pretilde[i]=alphause.transpose()*useri+movieuse*btemp;
                    pretilde[i]=C.x.user[i].rating-pretilde[i];
                    Sis[i]=pretilde[i]*(pretilde[i].transpose());
                }
                change.resize(n);
                MatrixXd Tm;
                A.log_det=VectorXd::Zero(n);
#pragma omp parallel for private(Tm,j,mtemp)
                for (i=0; i<n; ++i) {
                    mtemp=Omegais2[i];
                    Omegais2[i]=Zis[i]-Uis[i];
                    MatrixXd Zisub=matrix_rowcolsub(Zis[i],C.x.user[i].item);
                    MatrixXd Uisub=matrix_rowcolsub(Uis[i],C.x.user[i].item);
                    MatrixXd Omegaipre=matrix_rowcolsub(Omegaisdoc[i],C.x.user[i].item);
                    Tm=0.5*Sis[i]-rho_Omega*(Zisub-Uisub)-2*c*Omegaipre;
                    SelfAdjointEigenSolver<MatrixXd> es(Tm);
                    VectorXd eigenv=es.eigenvalues();
                    MatrixXd V=es.eigenvectors();
                    for(j=0;j<Tm.rows();++j){
                        eigenv[j]=(-eigenv[j]+sqrt(eigenv[j]*eigenv[j]+2*(rho_Omega+2*c)))/(2*(rho_Omega+2*c));
                        A.log_det[i]+=log(eigenv[j]);
                    }
                    MatrixXd Omegais2sub=V*(eigenv.asDiagonal())*V.transpose();
                    matrix_rowcolsub_givevalue(Omegais2[i],C.x.user[i].item,Omegais2sub);
                    change[i]=absm(mtemp-Omegais2[i]).maxCoeff();
                }
                maxOmega=change.maxCoeff();
                
                maxdiff_inner=max(max(maxalpha,maxbeta),maxOmega);
                if(maxdiff_inner<=1e-3) break;
                ++iter_ori;
            }
            cout<<"iter_ori="<<iter_ori<<", maxalpha="<<maxalpha<<", maxbeta="<<maxbeta<<", maxOmega="<<maxOmega<<endl;
            change=VectorXd::Zero(theta.cols());//must define to zero, otherwise some elements are not defined.
            change2=change;
#pragma omp parallel for private(i,j,atemp,btemp,a)
            for (l=0; l<(n-1)*n/2; ++l){
                if (diffB[l].size()>0) {//because of this row, must define change and change2 to be zero at beginning
                    i=trin.first(l);
                    j=trin.second(l);
                    atemp=mubeta2.row(i)-mubeta2.row(j);
                    atemp=atemp+u.col(l);
                    a=atemp;
                    for (int tt=0; tt<diffB[l].size(); ++tt) {
                        int index=diffB[l][tt];
                        a[index]=ST(a[index],C.lambda1/(2*C.rho));//number correct? only soft_threshold indices that are in diffB[l]
                    }
                    btemp=a-theta.col(l);
                    theta.col(l)=a;
                    change[l]=btemp.lpNorm<Infinity>();
                    btemp=atemp-a;
                    change2[l]=(btemp-u.col(l)).lpNorm<Infinity>();
                    u.col(l)=btemp;
                }
            }
            maxtheta_beta=change.lpNorm<Infinity>();
            maxu_beta=change2.lpNorm<Infinity>();
            tttbeta=max(max(maxbeta,maxtheta_beta),maxu_beta);
            
            change=VectorXd::Zero(theta2.cols());
            change2=change;
#pragma omp parallel for private(i,j,btemp,a,atemp)
            for (l=0; l<(p-1)*p/2; ++l){
                i=trip.first(l);
                j=trip.second(l);
                if (diffA[l].size()>0){
                    atemp=mualpha2.col(i)-mualpha2.col(j)+u2.col(l);
                    a=atemp;
                    for (int tt=0; tt<diffA[l].size(); ++tt) {
                        int index=diffA[l][tt];
                        a[index]=ST(atemp[index],C.lambda1/(2*C.rho));//correct, only soft-threshold indices in diffA[l]
                    }
                    btemp=a-theta2.col(l);
                    theta2.col(l)=a;
                    change[l]=btemp.lpNorm<Infinity>();
                    btemp=atemp-a;
                    change2[l]=(btemp-u2.col(l)).lpNorm<Infinity>();
                    u2.col(l)=btemp;
                }
            }
            maxtheta_alpha=change.lpNorm<Infinity>();
            maxu_alpha=change2.lpNorm<Infinity>();
            maxalpha=absm(mualpha2-alphapre).maxCoeff();
            tttalpha=max(max(maxalpha,maxtheta_alpha),maxu_alpha);
            
            off_diff=0;
            change.resize(p);
#pragma omp parallel for private(j,l,atemp)
            for(i=0;i<p;++i){//diagonal
                int usersize=itemmore[i].user.size();
                atemp.resize(usersize+1);
                VectorXd current(usersize+1);
                if (judge_diag[i]==0) {
                    for (j=0; j<usersize+1; ++j) {
                        int userno=j<usersize?itemmore[i].user[j]:itemmore_idle[i];
                        atemp[j]=Omegais2[userno](i,i)+Uis[userno](i,i);
                        current[j]=Zis[userno](i,i);
                    }
                    change[i]=(atemp-current).lpNorm<Infinity>();
                }
                else {
                    vector<vector<Vector2i>> cstrfdi1=cstrfdi_diag[i],cstrsdi1=cstrsdi_diag[i];
                    for (j=0; j<usersize+1; ++j) {
                        int userno=j<usersize?itemmore[i].user[j]:itemmore_idle[i];
                        atemp[j]=Omegais2[userno](i,i)+Uis[userno](i,i);
                        current[j]=Zis[userno](i,i);
                    }
                    atemp=solve21_ama2_ptl_prop(n-usersize,rho_Zdiag,1000,1e-4,atemp,cstrfdi1,cstrsdi1);
                    change[i]=(atemp-current).lpNorm<Infinity>();
                }
                
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
            
            change.resize(rlv.size());
#pragma omp parallel for private(i,j,l,ik,il,atemp,mtemp,temp)
            for (int k=0; k<rlv.size();++k) {
                i=rlv[k].item1;
                j=rlv[k].item2;
                int usersize=rlv[k].userno.size();
                atemp.resize(usersize+1);
                
                VectorXd current(usersize+1);
                if (judge_offdiag[k]==0) {
                    for (l=0; l<usersize+1; ++l) {
                        int userno=l<usersize?rlv[k].userno[l]:rlv_idle[k];
                        atemp[l]=Omegais2[userno](i,j)+Uis[userno](i,j);
                        current[l]=Zis[userno](i,j);
                    }
                    change[k]=(atemp-current).lpNorm<Infinity>();
                }
                
                if (judge_offdiag[k]>0) {
                    vector<vector<Vector2i>> cstrfdi1=cstrfdi_offdiag[k],cstrsdi1=cstrsdi_offdiag[k];
                    for (l=0; l<usersize+1; ++l) {
                        int userno=l<usersize?rlv[k].userno[l]:rlv_idle[k];
                        atemp[l]=Omegais2[userno](i,j)+Uis[userno](i,j);
                        current[l]=Zis[userno](i,j);
                    }
                    atemp=solve21_ama2_ptl_prop(n-usersize,rho_Zdiag/2,1000,1e-4,atemp,cstrfdi1,cstrsdi1);
                    vector<int> M=Osize[k];
                    for (int tt=0; tt<M.size(); ++tt){
                        int index=M[tt];
                        atemp[index]=ST(atemp[index],rho_Z1);//shrinkage, size constraint
                    }
                    change[k]=(atemp-current).lpNorm<Infinity>();
                }
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
            maxU=0;
            change.resize(n);
#pragma omp parallel for private(mtemp)
            for (i=0; i<n; ++i) {
                mtemp=Omegais2[i]-Zis[i];
                Uis[i]=Uis[i]+mtemp;
                change[i]=absm(mtemp).maxCoeff();
            }
            maxU=change.lpNorm<Infinity>();
            for (i=0; i<n; ++i) {
                change[i]=absm(Omegaispre[i]-Omegais2[i]).maxCoeff();
            }
            maxOmega=change.maxCoeff();
            tttOmega=max(max(maxOmega,maxZ),maxU);
            
            maxdiff_admm=max(max(tttbeta,tttalpha),tttOmega);
            B.alphafuse=cal_alpha_fuse_abs(trip,mualpha2,diffA);
            B.betafuse=cal_beta_fuse_abs(trin,mubeta2,diffB);
            B.Omegafusesize=cal_Omega_fusesize_abs(Omegais2,cstrfdi_diag,cstrsdi_diag,cstrfdi_offdiag,cstrsdi_offdiag,judge_diag,judge_offdiag,Osize,rlv,itemmore,itemmore_idle,rlv_idle);  
            obj_abs2=cal_doc_obj_struct_prop_abs2(c,betadoc,alphadoc,Omegaisdoc,Sis,B);
            ++admm_iter;
			cout<<"admm iter is "<<admm_iter<<", doc_iter="<<doc_iter<<", obj1-obj2="<<obj_abs1-obj_abs2<<", tttalpha="<<tttalpha<<", tttbeta="<<tttbeta<<endl;
            cout<<"obj1="<<obj_abs1<<", obj2="<<obj_abs2<<endl;
            cout<<"maxbeta="<<maxbeta<<", maxtheta_beta="<<maxtheta_beta<<", maxu_beta="<<maxu_beta<<endl;
            cout<<"maxalpha="<<maxalpha<<", maxtheta_alpha="<<maxtheta_alpha<<", maxu_alpha="<<maxu_alpha<<endl;
            cout<<"maxOmega="<<maxOmega<<", maxZ="<<maxZ<<", maxU="<<maxU<<endl;
            cout<<"Omegais2[0].block(0,0,9,9)="<<endl;
			cout<<Omegais2[0].block(0,0,9,9)<<endl; 
			if(obj_abs1-obj_abs2<1e-3){ 
				break;
			}
            else{
                obj_abs1=obj_abs2;
            }
        }
        cout<<"It took "<<admm_iter<<" admm_iters"<<endl;
        ++doc_iter;
        maxalpha=absm(mualpha2-alphadoc).maxCoeff();
        maxbeta=absm(mubeta2-betadoc).maxCoeff();
		alphafuse_tau=0;betafuse_tau=0;Omegasize_tau=0;Omegafuse_tau=0;
		
        involvedB=VectorXi::Zero(n);//construct diffB and calculate betafuse_tau
		for(i=0;i<n-1;++i){//diffB is in order 12,13,14,...1n; 23,24,25...2n;...(n-1)n
			for (j=i+1; j<n; ++j) {
				l=i*(n-(i+1)*0.5)+j-i-1;
				diffB[l].clear();
				vector<int> diff1;
				for (int tt=0; tt<mdim; ++tt) {
					double beta2_diff=abs(mubeta2(i,tt)-mubeta2(j,tt));
					if (beta2_diff<=C.tau) {
						diff1.push_back(tt);
						betafuse_tau+=beta2_diff;
					}
				}
				betafuse_tau+=(mdim-diff1.size())*C.tau;
				if(diff1.size()>0){
					diffB[l]=diff1;
					++involvedB[i];++involvedB[j];//cannot parallize because of this line
				}
			}
		}   
		involvedA=VectorXi::Zero(p);
		for(i=0;i<p-1;++i){
			for (j=i+1; j<p; ++j) {
				l=i*(p-(i+1)*0.5)+j-i-1;
				diffA[l].clear();
				vector<int> diff1;
				for (int tt=0; tt<udim; ++tt) {
					double alpha2_diff=abs(mualpha2(tt,i)-mualpha2(tt,j));
					if (alpha2_diff<=C.tau) {
						diff1.push_back(tt);
						alphafuse_tau+=alpha2_diff;
					}
				}
				alphafuse_tau+=(udim-diff1.size())*C.tau;
				if(diff1.size()>0){
					diffA[l]=diff1;
					++involvedA[i];++involvedA[j];
				}
			}
		}
		
		//construct Osize and calculate Omegasize_tau;
#pragma omp parallel for private(item1,item2,usersize,M,tt,userno) reduction(+:Omegasize_tau)
		for(i=0;i<rlv.size();++i){
			item1=rlv[i].item1;
			item2=rlv[i].item2;
			usersize=rlv[i].userno.size();
			M.clear();
			for (tt=0; tt<usersize; ++tt) {
				userno=rlv[i].userno[tt];
				double Osize_temp=abs(Omegais2[userno](item1,item2));
				if(Osize_temp<=C.tau){
					M.push_back(tt);//just push back index in rlv[i],tt is 第tt个user
					Omegasize_tau+=Osize_temp;
				}
			}
			Omegasize_tau+=(usersize-M.size())*C.tau;
			tt=usersize;
			userno=rlv_idle[i];
			double Osize_temp=abs(Omegais2[userno](item1,item2));
			if(Osize_temp<=C.tau){
				M.push_back(tt);
				Omegasize_tau+=Osize_temp*(n-usersize);
			}
			else{
				Omegasize_tau+=C.tau*(n-usersize);
			}
			Osize[i]=M;
		}
		
#pragma omp parallel for private(l,usersize,userno1,userno2,cstrfdi1,cstrsdi1,cstrno,tt,pairtemp) reduction(+:Omegafuse_tau)
		for (i=0; i<p; ++i) {
			cstrfdi1.clear();cstrsdi1.clear();
			judge_diag[i]=0;
			usersize=itemmore[i].user.size();
			if (usersize==1) {
				//because of the push_back(), cannot parallelize
				userno1=itemmore[i].user[0];
				userno2=itemmore_idle[i];
				double Ofuse_temp=abs(Omegais2[userno1](i,i)-Omegais2[userno2](i,i));
				if (Ofuse_temp<=C.tau){
					cstrfdi1.resize(2);cstrsdi1.resize(2);
					pairtemp(1)=0;
					pairtemp(0)=1;
					cstrfdi1[0].push_back(pairtemp);
					pairtemp(0)=0;
					cstrsdi1[1].push_back(pairtemp);
					cstrfdi_diag[i]=cstrfdi1;
					cstrsdi_diag[i]=cstrsdi1;
					judge_diag[i]=1;
					Omegafuse_tau+=(n-1)*Ofuse_temp;
				}
				else{
					Omegafuse_tau+=(n-1)*C.tau;
				}
			}
			if (usersize>1){
				cstrfdi1.resize(usersize+1),cstrsdi1.resize(usersize+1);
				cstrno=0;
				for (l=0; l<usersize; ++l) {
					userno1=itemmore[i].user[l];
					for (tt=l+1; tt<usersize; ++tt) {
						userno2=itemmore[i].user[tt];
						double Ofuse_temp=abs(Omegais2[userno1](i,i)-Omegais2[userno2](i,i));
						if (Ofuse_temp<=C.tau){
							pairtemp(1)=cstrno;
							pairtemp(0)=tt;
							cstrfdi1[l].push_back(pairtemp);//last of cstrfdi1 is never used, as of first of cstrsdi1
							pairtemp(0)=l;
							cstrsdi1[tt].push_back(pairtemp);
							++cstrno;
							Omegafuse_tau+=Ofuse_temp;
						}
						else{
							Omegafuse_tau+=C.tau;
						}
					}
					tt=usersize;
					userno2=itemmore_idle[i];
					double Ofuse_temp=abs(Omegais2[userno1](i,i)-Omegais2[userno2](i,i));
					if (Ofuse_temp<=C.tau){
						pairtemp(1)=cstrno;
						pairtemp(0)=tt;
						cstrfdi1[l].push_back(pairtemp);//last of cstrfdi1 is never used, as of first of cstrsdi1
						pairtemp(0)=l;
						cstrsdi1[tt].push_back(pairtemp);
						++cstrno;
						Omegafuse_tau+=(n-usersize)*Ofuse_temp;
					}
					else{
						Omegafuse_tau+=(n-usersize)*C.tau;
					}
				}
				cstrfdi_diag[i]=cstrfdi1;
				cstrsdi_diag[i]=cstrsdi1;
				judge_diag[i]=cstrno;
			}
		}
		
#pragma omp parallel for private(l,usersize,i,j,userno1,userno2,cstrfdi1,cstrsdi1,cstrno,tt,pairtemp) reduction(+:Omegafuse_tau)
		for (k=0; k<rlv.size();++k) {
			usersize=rlv[k].userno.size();
			judge_offdiag[k]=0;
			cstrfdi1.clear();cstrsdi1.clear();
			i=rlv[k].item1;
			j=rlv[k].item2;
			if (usersize==1) {
				userno1=rlv[k].userno[0];
				userno2=rlv_idle[k];
				double Ofuse_temp=abs(Omegais2[userno1](i,j)-Omegais2[userno2](i,j));
				if(Ofuse_temp<=C.tau){
					cstrfdi1.resize(2);cstrsdi1.resize(2);
					Vector2i pairtemp;
					pairtemp(1)=0;
					pairtemp(0)=1;
					cstrfdi1[0].push_back(pairtemp);
					pairtemp(0)=0;
					cstrsdi1[1].push_back(pairtemp);
					cstrfdi_offdiag[k]=cstrfdi1;
					cstrsdi_offdiag[k]=cstrsdi1;
					judge_offdiag[k]=1;
					Omegafuse_tau+=(n-1)*Ofuse_temp;
				}
				else{
					Omegafuse_tau+=(n-1)*C.tau;
				}
			}
			if(usersize>1){
				cstrfdi1.resize(usersize+1);cstrsdi1.resize(usersize+1);
				cstrno=0;
				for (l=0; l<usersize; ++l) {//l should be private
					userno1=rlv[k].userno[l];
					for (tt=l+1; tt<usersize; ++tt) {
						userno2=rlv[k].userno[tt];
						double Ofuse_temp=abs(Omegais2[userno1](i,j)-Omegais2[userno2](i,j));
						if(Ofuse_temp<=C.tau){
							pairtemp(1)=cstrno;
							pairtemp(0)=tt;
							cstrfdi1[l].push_back(pairtemp);
							pairtemp(0)=l;
							cstrsdi1[tt].push_back(pairtemp);
							++cstrno;
							Omegafuse_tau+=Ofuse_temp;
						}
						else{
							Omegafuse_tau+=C.tau;
						}
					}
					tt=usersize;
					userno2=rlv_idle[k];
					double Ofuse_temp=abs(Omegais2[userno1](i,j)-Omegais2[userno2](i,j));
					if(Ofuse_temp<=C.tau){
						pairtemp(1)=cstrno;
						pairtemp(0)=tt;
						cstrfdi1[l].push_back(pairtemp);
						pairtemp(0)=l;
						cstrsdi1[tt].push_back(pairtemp);
						++cstrno;
						Omegafuse_tau+=(n-usersize)*Ofuse_temp;
					}
					else{
						Omegafuse_tau+=(n-usersize)*C.tau;
					}
				}
				cstrfdi_offdiag[k]=cstrfdi1;
				cstrsdi_offdiag[k]=cstrsdi1;
				judge_offdiag[k]=cstrno;
			}
		}
		
		tstart1=time(0);
        obj2=cal_TLP_obj_prop_size_wi(n,A.c1,A.c3,A.c4,A.c5,A.c6,A.lambda1,A.lambda2,A.lambda3,A.tau,alphafuse_tau,betafuse_tau,Omegasize_tau,Omegafuse_tau, A.log_det,
          Sis,C.x.user,Omegais2); 
        tend1=time(0);
        time1=difftime(tend1,tstart1); 
        cout<<"doc_iter="<<doc_iter<<", obj_doc2="<<obj2<<", obj_doc1="<<obj1<<", obj_doc1-obj_doc2="<<obj1-obj2<<", maxalpha="<<maxalpha<<", maxbeta="<<maxbeta<<endl;
        if (obj1-obj2<0.015){//&&doc_iter>1 change 0.01->0.05->0.1 
            break;
        }
        else{
            obj1=obj2;
        }
    }
    cout<<"doc_iter="<<doc_iter<<endl;
    mtemp.resize(n,p);
    mtemp=C.users*mualpha2+mubeta2*C.movie.transpose();//compact multi is faster than one by one
    re.mualpha=mualpha2;
    re.mubeta=mubeta2;
    re.Omegais=Omegais2;
    re.solu=mtemp;
    re.maxdiff=max(maxbeta,maxalpha);//should change here  
    re.obj=obj2;
    return re;    
}





