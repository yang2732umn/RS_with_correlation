#include "consider_covariance.h"
vector<MatrixXd> l1_solve_Omega(double lambda2,double lambda3,double rho2,const MatrixXd &mualpha,const MatrixXd &mubeta,const vector<rated_user>& user, 
    const vector<rated_1itemmore> &itemmore,const vector<Omega_1rlv_simple> &rlv,const vector<MatrixXd> & Sis){
    int n=mubeta.rows();
    int p=mualpha.cols();
    int udim=mualpha.rows();
    int mdim=mubeta.cols();
    int c1=0,c2=0,c3=0,c6=0;
    int c4=n*(n-1)/2*mdim;
    int c5=p*(p-1)/2*udim;
    for (int kk=0; kk<n; ++kk) {
        int kkk=user[kk].item.size();
        c1+=kkk;
        c2+=kkk*kkk;
    }
    c3=n*p*(p-1)/2;  
    cout<<"rlv.size is "<<rlv.size()<<endl;
    cout<<"item 0 rated by "<<endl;
    stl_vec_cout(itemmore[0].user);  
    c6=(p+rlv.size())*n*(n-1)/2;
    double rho_Omega=rho2*c1/c2;
    double rho_Zdiag=(c2*lambda3)/(2*rho2*c6);
    double rho_Z1=(c2*lambda2)/(2*rho2*c3);
    cout<<"rho_Omega="<<rho_Omega<<", rho_Zdiag="<<rho_Zdiag<<", rho_Z1="<<rho_Z1<<endl;
    
    int i,j,l;
    VectorXd change,atemp;
    double max_admm=0,maxOmega,maxZ,maxU;
    vector<int> itemmore_idle(p),rlv_idle(rlv.size());
#pragma omp parallel for private(j)
    for (i=0; i<p; ++i) {
        for (j=0; j<itemmore[i].user.size(); ++j) {
            if(j<itemmore[i].user[j]) break;
        }//result j is one user didn't rate this item
        itemmore_idle[i]=j;
    }
#pragma omp parallel for private(l)
    for (i=0; i<rlv.size(); ++i) {
        for (l=0; l<rlv[i].userno.size(); ++l) {
            if(l<rlv[i].userno[l]) break;
        }//result l is one user didn't rate this pair
        rlv_idle[i]=l;
    }
    
    vector<MatrixXd> Omegais(n),Zis(n),Uis(n);
    for (int i=0; i<n; ++i) {
        Omegais[i]=MatrixXd::Identity(p,p);
        Zis[i]=MatrixXd::Identity(p,p);
        Uis[i]=MatrixXd::Zero(p,p);
    }
    time_t tstart1,tend1,tstart2,tend2,tstart3,tend3,tstart4,tend4,tstart5,tend5,tstart6,tend6;
    double time1=0,time2=0,time3=0,time4=0,time5=0,time6=0;
    int admm_iter=0, maxIter=1000;
    double obj1=0,obj2=0,temp2=0;
   
#pragma omp parallel for reduction(+:obj1)
	for (i=0;i<n;++i){
		obj1=obj1+0.5*(Sis[i]).trace()/c1;//starting from Omega=I
	}    
    while(admm_iter<maxIter){
    	obj2=0;
    	temp2=0;//forgot this
    	//update original variable Omegais
    	change.resize(n);
		MatrixXd Tm;
		tstart1=time(0);
		MatrixXd mtemp;
	#pragma omp parallel for private(Tm,j,mtemp) reduction(+:obj2,temp2)
		for (i=0; i<n; ++i) {
			mtemp=Omegais[i];
			Omegais[i]=Zis[i]-Uis[i];
			MatrixXd Zisub=matrix_rowcolsub(Zis[i],user[i].item);
			MatrixXd Uisub=matrix_rowcolsub(Uis[i],user[i].item);
			Tm=0.5*Sis[i]-rho_Omega*(Zisub-Uisub);
			SelfAdjointEigenSolver<MatrixXd> es(Tm);
			VectorXd eigenv=es.eigenvalues();
			MatrixXd V=es.eigenvectors();
			double logdet=0;
			for(j=0;j<Tm.rows();++j){
				eigenv[j]=(-eigenv[j]+sqrt(eigenv[j]*eigenv[j]+2*rho_Omega))/(2*rho_Omega);
				logdet+=log(eigenv[j]);
			}
			MatrixXd Omegais2sub=V*(eigenv.asDiagonal())*V.transpose();
			matrix_rowcolsub_givevalue(Omegais[i],user[i].item,Omegais2sub);
			change[i]=absm(mtemp-Omegais[i]).maxCoeff();
			obj2=obj2+(Sis[i]*Omegais2sub).trace()-logdet;
			temp2=temp2+(absm(Omegais[i]).sum()-absm(Omegais[i]).trace());
		}
		obj2=obj2*0.5/c1;
		obj2+=lambda2/c3*temp2;
		tend1=time(0);
		time1+=difftime(tend1, tstart1);
		//cout<<"change for Omega is "<<change.transpose()<<endl; 
		maxOmega=change.maxCoeff();
		
    	//update dual Zis and Uis
    	cout<<"calculate Z, U"<<endl;   
		//update Zis
		tstart2=time(0);
		double off_diff=0;
		//diagonal elements, should also use solve21_ama
		change.resize(p);
		for(i=0;i<p;++i){
			int usersize=itemmore[i].user.size();//just assume no movie is rated by every person
			atemp.resize(usersize+1);
			VectorXd current(usersize+1);
			#pragma omp parallel for
			for (j=0; j<usersize; ++j) {
				int userno=itemmore[i].user[j];
				atemp[j]=Omegais[userno](i,i)+Uis[userno](i,i);
				current[j]=Zis[userno](i,i);
			}
			atemp[usersize]=Omegais[itemmore_idle[i]](i,i)+Uis[itemmore_idle[i]](i,i);
			current[usersize]=Zis[itemmore_idle[i]](i,i);
			//parallel in solve21_ama2_prop_p
			atemp=solve21_ama2_prop_p(atemp,n-usersize,rho_Zdiag,2000,2e-4);//was 1e-4 previously
			//cout<<"For item "<<i<<" , atemp="<<atemp.transpose()<<endl;
			for (l=0; l<usersize; ++l) {
				#pragma omp parallel for reduction(+:off_diff)
				for (int tt=l+1; tt<usersize+1; ++tt) {
					if(tt<usersize) off_diff=off_diff+abs(atemp[l]-atemp[tt]);
					else off_diff=off_diff+(n-usersize)*abs(atemp[l]-atemp[tt]);
				}
			}
			change[i]=(atemp-current).lpNorm<Infinity>();
			#pragma omp parallel for
			for(j=0;j<n;++j){
				Zis[j](i,i)=atemp[usersize];
			}
			#pragma omp parallel for
			for(j=0;j<usersize;++j){
				int userno=itemmore[i].user[j];
				Zis[userno](i,i)=atemp[j];
			}
		}
		maxZ=change.lpNorm<Infinity>();
            
		//off-diagonal elements
		change.resize(rlv.size());
		for (int k=0; k<rlv.size();++k) {
			i=rlv[k].item1;
			j=rlv[k].item2;
			int usersize=rlv[k].userno.size();//just assume no pair is rated by every person
			atemp.resize(usersize+1);
			VectorXd current(usersize+1);
			#pragma omp parallel for
			for (l=0; l<usersize; ++l) {
				int userno=rlv[k].userno[l];
				atemp[l]=Omegais[userno](i,j)+Uis[userno](i,j);
				current[l]=Zis[userno](i,j);
			}
			atemp[usersize]=Omegais[rlv_idle[k]](i,j)+Uis[rlv_idle[k]](i,j);
			current[usersize]=Zis[rlv_idle[k]](i,j);
			
			atemp=solve21_ama2_prop_p(atemp,n-usersize,rho_Zdiag/2,2000,2e-4);//was 1e-4 previously
			atemp=ST_vec_p(atemp,rho_Z1);//shrink, function itself is parallel
			//cout<<"atemp="<<atemp.transpose()<<endl;
			//cout<<"For rlv "<<k<<" , atemp="<<atemp.transpose()<<endl;    
			
			for (l=0; l<usersize; ++l) {
				#pragma omp parallel for reduction(+:off_diff)
				for (int tt=l+1; tt<usersize+1; ++tt) {
					if(tt<usersize) off_diff=off_diff+abs(atemp[l]-atemp[tt]);
					else off_diff=off_diff+(n-usersize)*abs(atemp[l]-atemp[tt]);//calculated here
				}
			}
			change[k]=(atemp-current).lpNorm<Infinity>();
			
#pragma omp parallel for
			for(int tt=0;tt<n;++tt){
				Zis[tt](i,j)=Zis[tt](j,i)=atemp[usersize];
			}
#pragma omp parallel for
			for(int tt=0;tt<usersize;++tt){
				int userno=rlv[k].userno[tt];
				Zis[userno](i,j)=Zis[userno](j,i)=atemp[tt];
			}
		}
		cout<<"done Z calculation"<<endl;
	
		maxZ=max(maxZ,change.lpNorm<Infinity>());
		tend2=time(0);
		time2+=difftime(tend2, tstart2);
	
		maxU=0;
		change.resize(n);
#pragma omp parallel for private(mtemp)
		for (i=0; i<n; ++i) {
			mtemp=Omegais[i]-Zis[i];
			Uis[i]=Uis[i]+mtemp;
			change[i]=absm(mtemp).maxCoeff();
		}
		maxU=change.lpNorm<Infinity>();
		max_admm=max(maxOmega,max(maxZ,maxU));
		obj2+=off_diff*lambda3/(2*c6);
		cout<<"admm_iter="<<admm_iter<<", obj1="<<obj1<<", obj2="<<obj2<<", obj1-obj2="<<obj1-obj2<<endl;
		cout<<"maxOmega="<<maxOmega<<", maxZ="<<maxZ<<", maxU="<<maxU<<endl;
		cout<<"Omegais[0].block(0,0,9,9)="<<endl;
		cout<<Omegais[0].block(0,0,9,9)<<endl;
		cout<<"Zis[0].block(0,0,9,9)="<<endl;
		cout<<Zis[0].block(0,0,9,9)<<endl;
		cout<<"Omegais[1].block(0,0,9,9)="<<endl;
		cout<<Omegais[1].block(0,0,9,9)<<endl;
		cout<<"Omegais[4].block(0,0,9,9)="<<endl;
		cout<<Omegais[4].block(0,0,9,9)<<endl;
		cout<<"Omegais[14].block(0,0,9,9)="<<endl;
		cout<<Omegais[14].block(0,0,9,9)<<endl;
		++admm_iter;
		if (max_admm<1e-3||obj1-obj2<1e-4) break;   
		else {
		 obj1=obj2;   
		}  
    }
    return Omegais;  
}
    





