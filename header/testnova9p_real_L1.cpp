
#include "consider_covariance.h"

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
    
    MatrixXd solu_orig;
    double MSE_orig;
    
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
    
    string pathAndNameL1tune("../real_data/real_940_1298_tune_estvar_L1_s05.txt");//tune in last is real tune
    string pathAndNameL1final("../real_data/real_940_1298_final_estvar_L1.txt");
    
    lambda1.resize(2);lambda1<<0.9,0.5; //final[1];//100,10;//5,1;10;5,1;//5,1,0.1 
    VectorXd lambda2(2);lambda2<<500,300;//<<1;//1,0.1;//0.05; 
    // lambda1.resize(1);lambda1<<1;
//     VectorXd lambda2(1);lambda2<<300;
    VectorXd lambda3(2);lambda3<<5,1;//5; //the same as lambda1
    double c=0.02; //20;2000;0.02;0.1 seems working for TLP;    
    int count=0;
    int bestparaindex;
    double bestw_tune_MSE;
    string method;
    layers2 L;
    int nlayer=2;
    L.mualpha.resize(nlayer);
    L.mubeta.resize(nlayer);
    L.u.resize(nlayer);
    L.u2.resize(nlayer);
    L.theta.resize(nlayer);
    L.theta2.resize(nlayer);
    L.Omegais.resize(nlayer);
    L.Zis.resize(nlayer);
    L.Uis.resize(nlayer);
    L.names.resize(nlayer);
    result re3;
    alphanames.resize(lambda1.size()*lambda2.size()*lambda3.size());
    betanames.resize(lambda1.size()*lambda2.size()*lambda3.size());
    vector<string> lam1alphanames(lambda1.size()-1);
    vector<string> unames(alphanames.size()), u2names(alphanames.size()),thetanames(alphanames.size()),theta2names(alphanames.size()),Znames(alphanames.size()),Unames(alphanames.size());
    vector<string> Omeganames(lambda1.size()*lambda2.size()*lambda3.size());
    
    Cluster_mnl_p_ADMM_scale_para2_2 C;// here is start for L1
    
    C.x=train;
    C.movie=movie;
    C.users=users;
    C.Tol=1e-3;
    C.maxIter=3000;
    C.rho=2;//2;0.2
    C.rho2=15;//15;8, 1,2 is working better than previously used 8, obj is decreasing faster, and time used is less
    C.log_det=VectorXd::Zero(n);
    C.mualpha1=mualpha;
    C.mubeta1=mubeta;
    C.Zis.resize(n);
    C.Uis.resize(n);
    C.Omegais1.resize(n);
    string Omegastart="Omegais_1_400_1_start_s05.txt";// //Omegais_1_400_1_start_s05.txt
    C.Omegais1=readgn_seq(Omegastart,n);
    cout<<"Omegastart is "<<Omegastart<<". Reading finished."<<endl;
    
    //vector<MatrixXd> Omegais1(n);//=readgn_seq(Omeganame,n);
    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
        //C.Omegais1[i]=MatrixXd::Identity(p,p);
        C.Zis[i]=C.Omegais1[i];
        C.Uis[i]=MatrixXd::Zero(p,p);
    }
    cout<<"here"<<endl;
    C.u=MatrixXd::Zero(mubeta.cols(),n*(n-1)/2);
    C.theta=C.u;
    C.u2=MatrixXd::Zero(mualpha.rows(),p*(p-1)/2);
    C.theta2=C.u2;
    
    rec.resize(lambda1.size()*lambda2.size()*lambda3.size(),5);
    //vector<MatrixXd> lam1beta(lambda1.size()-1),lam1alpha(lambda1.size()- 1),lam1u(lambda1.size()-1),lam1u2(lambda1.size()-1),lam1theta(lambda1.size()-1),lam1theta2(lambda1.size()-1);
    //vector<vector<MatrixXd>> lam1Omegais(lambda1.size()-1),lam1Zis(lambda1.size()-1),lam1Uis(lambda1.size()-1);
    cout<<"here"<<endl;
    tstartall = time(0);
    for (int i=0; i<lambda1.size(); ++i) {
        C.lambda1=lambda1[i];
        for (int j=0; j<lambda2.size(); ++j){
            C.lambda2=lambda2[j];
            //if (lambda1.size()>1&&i>0&&j==0) {
            //C.mubeta1=lam1beta[i-1];
            //C.mualpha1=lam1alpha[i-1];
            //C.Omegais1=lam1Omegais[i-1];
            //C.u=lam1u[i-1];
            //C.u2=lam1u2[i-1];
            //C.theta=lam1theta[i-1];
            //C.theta2=lam1theta2[i-1];
            //C.Zis=lam1Zis[i-1];
            //C.Uis=lam1Uis[i-1];
            //cout<<"lam1beta=\n"<<lam1beta[i-1].block<2,2>(0,0)<<endl;
            //cout<<"lam1alpha=\n"<<lam1alpha[i-1].block<2,2>(0,0)<<endl;
            //cout<<"lam1u="<<endl;
            //cout<<lam1u[i-1].block<2,2>(0,0)<<endl;
            //cout<<"lam1Zis[0]="<<endl;
            //cout<<lam1Zis[i-1][0].block<2,2>(0,0)<<endl;
            //}
            for (int k=0; k<1; ++k) {//lambda3.size()
                C.lambda3=C.lambda1;//lambda3[k];can skip this 2*, not too much difference
                size_t found=dname2.find(".txt");
                string realpha=dname2;
                realpha=realpha.erase(found,4);
                realpha=realpha+"_";
                ostringstream convert;   // stream used for the conversion
                convert <<C.lambda1;      // insert the textual representation of 'Number' in the characters in the stream
                string lambdastr = convert.str();
                realpha=realpha.append(lambdastr);
                realpha=realpha+"_";
                ostringstream convert2;
                convert2 << C.lambda2;      // insert the textual representation of 'Number' in the characters in the stream
                string lambda2str = convert2.str();
                realpha=realpha.append(lambda2str);
                realpha=realpha+"_";
                ostringstream convert3;
                convert3 << C.lambda3;      // insert the textual representation of 'Number' in the characters in the stream
                string lambda3str = convert3.str();
                realpha=realpha.append(lambda3str);
                realpha=realpha+"_alpha.txt";
                cout<<"realpha is "<<realpha<<endl;
                found=realpha.find("alpha");
                string rebeta=realpha;
                rebeta.replace(found, 5, "beta");
                cout<<"rebeta is "<<rebeta<<endl;
                string reOmega=realpha;
                reOmega.replace(found, 5, "Omega");
                cout<<"reOmega is "<<reOmega<<endl;
                alphanames[count]=realpha;
                betanames[count]=rebeta;
                Omeganames[count]=reOmega;
                
                string reu=realpha;
                reu.replace(found, 5, "u");
                string reu2=realpha;
                reu2.replace(found, 5, "u2");
                string retheta=realpha;
                retheta.replace(found, 5, "theta");
                string retheta2=realpha;
                retheta2.replace(found, 5, "theta2");
                string reZ=realpha;
                reZ.replace(found, 5, "Z");
                string reU=realpha;
                reU.replace(found, 5, "U");
                unames[count]=reu;
                u2names[count]=reu2;
                thetanames[count]=retheta;
                theta2names[count]=retheta2;
                Znames[count]=reZ;
                Unames[count]=reU;
                
                cout<<"lambda1="<<C.lambda1<<", lambda2="<<C.lambda2<<", "<<"lambda3="<<C.lambda3<<", rho="<<C.rho<<endl;
                int outer_check=0;
                if(i==0) outer_check=1;  
                tstart=time(0);
                method="v9 2lambda1 s05";//rule1 is with rule lam1 series, rule0 is using bestalpha and bestbeta.
                //if(!(i==0&&j==0&&k==0)) rule2assignstarter2(2,i,j,k,0,C.mubeta1,C.mualpha1,C.u,C.u2,C.theta,C.theta2,C.Omegais1,C.Zis,C.Uis,L);//define starting value
                
                re3=Cluster_p_inADMM_scale_struct_v9_4(c,C);
                tend = time(0);
                double timecount=difftime(tend, tstart);
                cout << "It took " << timecount << " second(s)." << endl;  
                
                int clu_alpha=cal_cluster_no(C.mualpha1);
                cout<<"clusters No. in alpha is "<<clu_alpha<<endl;
                int clu_beta=cal_cluster_no(C.mubeta1.transpose());
                cout<<"clusters No. in beta is "<<clu_beta<<endl;
                string namenow=lambdastr+"_"+lambda2str+"_"+lambda3str;
                //rule2assignlayer2(2,i,j,k,0,lambda1.size(),lambda2.size(),lambda3.size(),C.mubeta1,C.mualpha1,C.u,C.u2,C.theta,C.theta2,C.Omegais1,C.Zis,C.Uis,L,namenow);//define L, rule2
                //if (lambda1.size()>1&&j==0&&(i!=lambda1.size()-1)) {
                //cout<<"i="<<i<<endl;
                //cout<<"lam1u.size()="<<lam1u.size()<<endl;
                //lam1alpha[i]=C.mualpha1;
                //lam1beta[i]=C.mubeta1;
                //lam1u[i]==C.u;
                //lam1u2[i]==C.u2;
                //lam1theta[i]==C.theta;
                //lam1theta2[i]==C.theta2;
                //lam1Omegais[i]=C.Omegais1;
                //lam1Zis[i]==C.Zis;
                //lam1Uis[i]==C.Uis;
                //cout<<"Here C.u=\n"<<lam1u[i].block<2,2>(0,0)<<endl;
                //cout<<"lam1u="<<endl;
                //cout<<lam1u[i].block<2,2>(0,0)<<endl;
                //}
                //cout<<"lam1u="<<endl;
                //cout<<lam1u[0].block<2,2>(0,0)<<endl;
                
                //initP=re3.P; // revise initP to zero, see if the same as v6
                //cout<<"initP value="<<initP.Zis[0](1,1)<<endl;
                VectorXd final(14);
                final[0]=C.lambda1;
                final[1]=C.lambda2;
                final[2]=C.lambda3;
                final[3]=cal_likelihood_prop(users,movie,tune,C.mualpha1,C.mubeta1,C.Omegais1);//this is on train and is wrong!!!
                double MSE=cal_MSE(re3.solu,tune);
                cout<<"MSE tune="<<MSE<<endl;
                final[4]=MSE;
                MSE=cal_MSE(re3.solu,train);
                cout<<"MSE train="<<MSE<<endl;
                final[5]=MSE;
                final[6]=re3.obj;
                final[7]=timecount;
                final[8]=clu_alpha;
                final[9]=clu_beta;
                final[10]=c;
                final[11]=C.rho2;
                final[12]=seedint;
                final[13]=cal_MSE(re3.solu,test);
                get_sVectortoData(method,final,pathAndNameL1tune);
                
                if (i==0&&j==0&&k==0) {
                    bestparaindex=0;
                    bestw_tune_MSE=final[3];//use MSE_tune as criterion, because likelihood on tune cannot be calculated due to lack of Omega
                    bestsolu=re3.solu;
                    bestalpha=C.mualpha1;
                    bestbeta=C.mubeta1;
                    bestOmegais=C.Omegais1;
                    bestu=C.u;
                    bestu2=C.u2;
                    besttheta=C.theta;
                    besttheta2=C.theta2;
                    bestZis=C.Zis;
                    bestUis=C.Uis;
                }
                else{
                    if (final[3]<bestw_tune_MSE) {
                        bestparaindex=count;
                        bestw_tune_MSE=final[3];//rec.row(bestindex) keep the best para combi
                        bestsolu=re3.solu;
                        bestalpha=C.mualpha1;
                        bestbeta=C.mubeta1;
                        bestOmegais=C.Omegais1;
                        bestu=C.u;
                        bestu2=C.u2;
                        besttheta=C.theta;
                        besttheta2=C.theta2;
                        bestZis=C.Zis;
                        bestUis=C.Uis;
                    }
                }
                rec(count,0)=C.lambda1;rec(count,1)=C.lambda2;rec(count,2)=C.lambda3; rec(count,3)=clu_alpha;rec(count,4)=clu_beta;
                ++count; 
                //C.mualpha1=bestalpha;
                //C.mubeta1=bestbeta;//use the current best alpha and beta for warm start               
                //do not do following, use previous as start
                //C.mualpha1=mualpha;
                //C.mubeta1=mubeta;
                //C.Omegais1=Omegais1;
                //C.Omegais1=bestOmegais;
                //C.Omegais1=Omegais1;
                //C.Omegais1=readgn_seq(Omeganames[bestparaindex],n);//use previous Omega
            }
        }
    }
    
    cout<<"bestparaindex is "<<bestparaindex<<endl;
    C.lambda1=rec(bestparaindex,0);
    C.lambda2=rec(bestparaindex,1);
    C.lambda3=rec(bestparaindex,2);
    get_MatrixtoData(bestalpha, alphanames[bestparaindex]);
    get_MatrixtoData(bestbeta, betanames[bestparaindex]);
    //get_matricestoData(bestOmegais,Omeganames[bestparaindex]);
    get_MatrixtoData(bestu,unames[bestparaindex]);
    get_MatrixtoData(bestu2,u2names[bestparaindex]);
    get_MatrixtoData(besttheta,thetanames[bestparaindex]);
    get_MatrixtoData(besttheta2,theta2names[bestparaindex]);
    //get_matricestoData(bestUis,Unames[bestparaindex]);
    //get_matricestoData(bestZis,Znames[bestparaindex]);   
    
    final.resize(14);
    final[0]=seedint;
    final[1]=C.lambda1;
    final[2]=C.lambda2;
    final[3]=C.lambda3;
    cout<<"bestbeta.block(0,0,5,5)="<<endl<<bestbeta.block(0,0,5,5)<<endl;
    cout<<"bestalpha.block(0,0,5,5)="<<endl<<bestalpha.block(0,0,5,5)<<endl;
    cout<<"bestsolu.block(0,0,5,5)="<<endl<<bestsolu.block(0,0,5,5)<<endl;
    
    final[4]=0;
    cout<<"With best lambda1="<<C.lambda1<<", best lambda2="<<C.lambda2<<", best lambda3="<<C.lambda3<<endl;
    MSE=cal_MSE(bestsolu,test);
    cout<<"L1 method, MSE test="<<MSE<<endl;
    final[5]=MSE;//unweighted MSE_test
    MSE=cal_MSE(bestsolu,train);
    cout<<"MSE train="<<MSE<<endl;
    final[6]=MSE;//unweighted MSE_train
    final[7]=rec(bestparaindex,3);//clu_alpha
    final[8]=rec(bestparaindex,4);//clu_beta
    final[9]=0;
    final[10]=0;
    
    tendall = time(0);
    cout<<"Total L1 cost "<<difftime(tendall, tstartall)<<" secs. \n";  
    final[11]=difftime(tendall, tstartall);
    final[12]=C.rho2;
    final[13]=c;
    
    get_sVectortoData(method,final, pathAndNameL1final);  
    
    /*if(remove(trainname.c_str())!=0||remove(tunename.c_str())!=0||remove(testname.c_str())!=0||remove(traintunename.c_str())!=0
     ||remove(name3.c_str())!=0||remove(name4.c_str())!=0||remove("alphasim_100_100_0.8_o2_caseg6_s0501_2.txt")!=0||remove("betasim_100_100_0.8_o2_caseg6_s0501_2.txt")!=0
     ||remove(alphatruename.c_str())!=0||remove(betatruename.c_str())!=0||remove("Sigmatune100_100_0.8_o2_caseg6_s0501_2.txt")!=0
     ||remove("Sigmatest100_100_0.8_o2_caseg6_s0501_2.txt")!=0||remove("Cortrain100_100_0.8_o2_caseg6_s0501_2.txt")!=0
     ||remove("Omega1_true100_100_0.8_o2_caseg6_s0501_2.txt")!=0||remove("Omega2_true100_100_0.8_o2_caseg6_s0501_2.txt")!=0
     ||remove("Omegais_true100_100_0.8_o2_caseg6_s0501_2.txt")!=0) perror("Error deleting file\n");*/
    
}

