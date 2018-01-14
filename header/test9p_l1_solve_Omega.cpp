//this file want it to do 27 movie, 100 user, 0.8 missing, caseg6_s05012~caseg6_s05012 series of simulation for novariance, L1 and TLP


#include "consider_covariance.h"
//89,90
//c3 has 12 clusters for both users and items, 75% quantile of abs Sigma offdiag is set to 0.5
//c3 has 10 clusters for both users and items, Omega=Omega*2
//d1 has rho^(i-j) for Sigma
//d2 has variance 0.5
//d7 has 3 groups of Omega 10,1,5
//d8 5,0.5,2.5 times
//e1 all use 0.9^(i-j)
//f1 200 users, 50 movies, all use one Omega
//f2 use two Omegas
//f3 use scale /2 and /20
//f4 use scale /20 and /40
//f5 use scale /20 and /100, Omega1 and Omega2 have 446 nonzero to estimate in total(abs<10=0), but only 200*50*0.1=1000 obs, maybe too small
//f6 use scale /20 and /100, with Omega1 and Omega2 more sparse
//f7 considers more users, n=400
//f8 uses submatrices of Sigma to generate data
//f12 is train data ~ Omega, and tune test not indep from train, also normal, w_MSE for test

//200_50_0.5 Omega2[abs(Omega2)<5]=0, Omega2[abs(Omega2)>20]=0
int main(){
    //novariance first
    string trainname("../real_data/train940_1298_s05.txt"); 
    
    int position1=trainname.find("s0");
    int position2=trainname.find(".");
    string seed=trainname.substr(position1+2,position2-position1-2);
    int seedint=atoi(seed.c_str());
    
    string tunename("../real_data/tune940_1298_s05.txt");
    string testname("../real_data/test940_1298_s05.txt");
    string dname2("../real_data/data_100k_s05.txt");
    string dname3("../real_data/data_100k_TLP_s05.txt");
    
    string name3("../real_data/movieinfo3.txt");
    MatrixXd movie=readgn(name3);
    string name4("../real_data/userinfo3.txt");
    MatrixXd users=readgn(name4);
    
    MatrixXd mualpha,mubeta;
    
    int n=users.rows();
    int p=movie.rows();
    int udim=users.cols();
    int mdim=movie.cols();
    MatrixXd A1=readgn(trainname);//A is data
    rated_user_and_item train=construct_user_item(A1);
    
    
    MatrixXd A2=readgn(testname);//A is data
    rated_user_and_item test=construct_user_item(A2);
    A2=readgn(tunename);//A is data
    rated_user_and_item tune=construct_user_item(A2);
    //A1=readgn(traintunename);//A is data
    //rated_user_and_item traintune=construct_user_item(A1);
    
    
    time_t tstart, tend,tstartall,tendall;
    cout<<"users.rows()="<<users.rows()<<", users.cols()="<<users.cols()<<endl;
    cout<<"mualpha.rows()="<<mualpha.rows()<<", mualpha.cols()="<<mualpha.cols()<<endl;
    cout<<"mubeta.rows()="<<mubeta.rows()<<", mubeta.cols()="<<mubeta.cols()<<endl;
    cout<<"movie.rows()="<<movie.rows()<<", movie.cols()="<<movie.cols()<<endl;
    
    MatrixXd solu_orig=users*mualpha+mubeta*movie.transpose();
    double MSE_orig=cal_MSE(solu_orig,test);
    MSE_orig=cal_MSE(solu_orig,train);
    cout<<"MSE orig train ="<<MSE_orig<<endl;
    
    string pathAndNamenovariancetune("../real_data/real_940_1298_tune_novariance_s05.txt");
    string pathAndNamenovariancefinal("../real_data/real_940_1298_final_novariance.txt");
    //double lambda=10;
    VectorXd lambda1(9); lambda1<<100,5,2,1.5,1.3,1,0.8,0.5,0.1;//800,600,400,350,300,250,200,150,100,50,30,10,5,1,0.5,0.1,0.05;
    double bestw_MSE_tune;
    double MSE;
    VectorXd final(10);
    int bestlambda1index;
    MatrixXd bestsolu,bestalpha,bestbeta;
    MatrixXd bestu,besttheta,bestu2,besttheta2;
    vector<MatrixXd> bestZis,bestUis(n);
    double rho=2;
    vector<string> alphanames(lambda1.size()), betanames(lambda1.size());
    MatrixXd rec(lambda1.size(),7);
    
    tstartall = time(0);
    
    
    mualpha=readgn("../real_data/data_100k_s05_0.9_alpha_novariance.txt");
    mubeta=readgn("../real_data/data_100k_s05_0.9_beta_novariance.txt");
    
    
    bestsolu=users*mualpha+mubeta*movie.transpose();
    cout<<"bestsolu.block(0,0,5,5)="<<endl<<bestsolu.block(0,0,5,5)<<endl;
    cout<<"mubeta.block(0,0,5,5)="<<endl<<mubeta.block(0,0,5,5)<<endl;
    cout<<"mualpha.block(0,0,5,5)="<<endl<<mualpha.block(0,0,5,5)<<endl;
    
    
    
    //Omegais1=readgn_seq("../simulation/rep100_prop/simdata_100_100_0.8_o2_caseg6_s0501_2_5_0.1_5_Omega.txt",n);
    //current mualpha and mubeta are best from novariance
    //next is L1
    
    vector<MatrixXd> bestOmegais;
    
    solu_orig=bestsolu;
    MSE_orig=cal_MSE(solu_orig,test);
    MSE_orig=cal_MSE(solu_orig,train);
    cout<<"MSE orig train ="<<MSE_orig<<endl;
    MSE_orig=cal_MSE(solu_orig,test);
    cout<<"MSE orig test ="<<MSE_orig<<endl;
    cout<<"Solve for L1 Omega for s05."<<endl;
    
    lambda1.resize(1);lambda1<<1;//final[1];//100,10;//5,1;10;5,1;//5,1,0.1
    VectorXd lambda2(2);lambda2<<400,40;//<<1;//1,0.1;//,0.05;
    VectorXd lambda3(2);lambda3<<5,1;//5; //the same as lambda1
    double c=0.02;//20;2000;0.02;0.1 seems working for TLP; 
    
    
    vector<rated_1itemmore> itemmore=construct_itemmore(train);
    vector<Omega_1rlv_simple> rlv=construct_rlv_simple(p,train.user);
    vector<MatrixXd> Sis(n);
    VectorXd btemp;
    vector<VectorXd> pretilde(n);
#pragma omp parallel for private(btemp)
    for (int i=0; i<n; ++i) {//construct pretilde
        MatrixXd alphause=matrix_colsub(mualpha,train.user[i].item);
        MatrixXd movieuse=matrix_rowsub(movie,train.user[i].item);
        VectorXd useri=users.row(i);
        btemp=mubeta.row(i);
        pretilde[i]=alphause.transpose()*useri+movieuse*btemp;
        pretilde[i]=train.user[i].rating-pretilde[i];
        Sis[i]=pretilde[i]*(pretilde[i].transpose());
        if(i==1||i==25||i==70){
            cout<<"Initially error for user "<<i<<" is "<<(pretilde[i]).transpose()<<endl;
        }
    }
    vector<MatrixXd> Omegais=l1_solve_Omega(lambda2[0],lambda1[0],5,mualpha,mubeta,train.user,itemmore,
     rlv,Sis);//5 is rho
    get_matricestoData(Omegais,"Omegais_1_400_1_start_s05.txt");            
}  



