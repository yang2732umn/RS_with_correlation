

#include "consider_covariance.h"

int main(){
    string trainname("../simulation/rep100_prop/trainsim100_100_0.85_o2_caseg6_s01_2.txt");
    
    int position1=trainname.find("s0");
    int position2=trainname.find(".");
    string seed=trainname.substr(position1+2,position2-position1-2);
    int seedint=atoi(seed.c_str());
    
    string tunename("../simulation/rep100_prop/tunesim100_100_0.85_o2_caseg6_s01_2.txt");
    string testname("../simulation/rep100_prop/testsim100_100_0.85_o2_caseg6_s01_2.txt");
    string traintunename("../simulation/rep100_prop/traintunesim100_100_0.85_o2_caseg6_s01_2.txt");
    string dname2("../simulation/rep100_prop/simdata_100_100_0.85_o2_caseg6_s01_2.txt");
    string dname3("../simulation/rep100_prop/simdata_TLP_100_100_0.85_o2_caseg6_s01_2.txt");
    
    string name3("../simulation/rep100_prop/moviesim100_100_0.85_o2_caseg6_s01_2.txt");
    MatrixXd movie=readgn(name3);
    string name4("../simulation/rep100_prop/usersim100_100_0.85_o2_caseg6_s01_2.txt");
    MatrixXd users=readgn(name4);
    
    string alphaname("../simulation/rep100_prop/alphasim_100_100_0.85_o2_caseg6_s01_2.txt");
    MatrixXd mualpha=readgn(alphaname);
    string betaname("../simulation/rep100_prop/betasim_100_100_0.85_o2_caseg6_s01_2.txt");
    MatrixXd mubeta=readgn(betaname);
    
    string alphatruename("../simulation/rep100_prop/alpha_true100_100_0.85_o2_caseg6_s01_2.txt");
    string betatruename("../simulation/rep100_prop/beta_true100_100_0.85_o2_caseg6_s01_2.txt");
    MatrixXd mualphatrue=readgn(alphatruename);
    MatrixXd mubetatrue=readgn(betatruename);
    
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
    
    vector<MatrixXd> testM=readgn_seq("../simulation/rep100_prop/Sigmatest100_100_0.85_o2_caseg6_s01_2.txt",test.user.size());
    for (int i=0; i<testM.size(); ++i) {
        testM[i]=testM[i].inverse();
    }
    time_t tstart, tend,tstartall,tendall;
    MatrixXd solu_orig=users*mualpha+mubeta*movie.transpose();
    double MSE_orig=cal_MSE(solu_orig,test);
    MSE_orig=cal_MSE(solu_orig,train);
    cout<<"MSE orig train ="<<MSE_orig<<endl;
    
    string pathAndNamenovariancetune("../simulation/rep100_prop/sim100_100_0.85_o2_caseg6_s01_2_40Omega_tune_novariance_p1.txt");
    string pathAndNamenovariancefinal("../simulation/rep100_prop/sim100_100_0.85_o2_caseg6_40Omega_final_nonsep_novariance_p1.txt");
    VectorXd lambda1(17); lambda1<<800,600,400,350,300,250,200,150,100,50,30,10,5,1,0.5,0.1,0.05;
    double bestw_MSE_tune;
    double MSE;
    VectorXd final(10);
    int bestlambda1index;
    MatrixXd bestsolu,bestalpha,bestbeta;
    MatrixXd bestu,besttheta,bestu2,besttheta2;
    vector<MatrixXd> bestZis,bestUis;
    double rho=2;
    vector<string> alphanames(lambda1.size()), betanames(lambda1.size());
    MatrixXd rec(lambda1.size(),7);
    
    tstartall = time(0);
     
    
     mualpha=readgn("../simulation/rep100_prop/simdata_100_100_0.85_o2_caseg6_s01_2_50_alpha_novariance.txt");
     mubeta=readgn("../simulation/rep100_prop/simdata_100_100_0.85_o2_caseg6_s01_2_50_beta_novariance.txt");
     MatrixXd bestalphano=mualpha,bestbetano=mubeta;
     
     
     bestsolu=users*mualpha+mubeta*movie.transpose();
     cout<<"bestsolu.block(0,0,5,5)="<<endl<<bestsolu.block(0,0,5,5)<<endl;
     cout<<"mubeta.block(0,0,5,5)="<<endl<<mubeta.block(0,0,5,5)<<endl;
     cout<<"mualpha.block(0,0,5,5)="<<endl<<mualpha.block(0,0,5,5)<<endl;
     
     double wMSE_no=cal_w_MSE(bestsolu,test,testM);
     cout<<"current novariance wMSE= "<<wMSE_no<<endl;
     
     
     
     vector<MatrixXd> bestOmegais;
    
     
     solu_orig=bestsolu;
     MSE_orig=cal_MSE(solu_orig,test);
     MSE_orig=cal_MSE(solu_orig,train);
     cout<<"MSE orig train ="<<MSE_orig<<endl;
     MSE_orig=cal_MSE(solu_orig,test);
     cout<<"MSE orig test ="<<MSE_orig<<endl;
     
     string pathAndNameL1tune("../simulation/rep100_prop/sim100_100_0.85_o2_caseg6_s01_2_40Omega_estvar_L1_tune_p1_3.txt");//tune in last is real tune
     string pathAndNameL1final("../simulation/rep100_prop/sim100_100_0.85_o2_caseg6_40Omega_estvar_L1_final_nonsep_p1.txt");
     
    lambda1.resize(1);lambda1<<20;//30;5,1;5,2;//5,1,0.1
    VectorXd lambda2(2);lambda2<<1,0.1;//<<1,0.1;//1,0.1;//,0.05; size penalty
    VectorXd lambda3(2);lambda3<<5,1;//1,0.5; //the same as lambda1, fused penalty
     double c=0.02;//0.1 seems working for TLP; 
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
     vector<string> lam1alphanames(lambda1.size()-1);
     vector<string> unames(alphanames.size()), u2names(alphanames.size()),thetanames(alphanames.size()),theta2names(alphanames.size()),Znames(alphanames.size()),Unames(alphanames.size());
     vector<string> Omeganames(lambda1.size()*lambda2.size()*lambda3.size());
     
     Cluster_mnl_p_ADMM_scale_para2 C;// here is start for L1
     
    
     C.x=train;
     C.movie=movie;
     C.users=users;
     C.Tol=1e-3;
     C.maxIter=3000;
     C.rho=2;//2;0.2
     C.rho2=15;//15;8，1,2 is working better than previously used 8, obj is decreasing faster, and time used is less
     C.mualpha1=mualpha;
     C.mubeta1=mubeta;
     C.Omegais1.resize(n);
     C.Zis.resize(n);
     C.Uis.resize(n);
     for (int i=0; i<n; ++i) {
     C.Omegais1[i]=MatrixXd::Identity(p,p);
     C.Zis[i]=MatrixXd::Identity(p,p);
     C.Uis[i]=MatrixXd::Identity(p,p);
     }
     C.u=MatrixXd::Zero(mubeta.cols(),n*(n-1)/2);
     C.theta=C.u;
     C.u2=MatrixXd::Zero(mualpha.rows(),p*(p-1)/2);
     C.theta2=C.u2;
     
     
     double s,s1,s2;
     vector<Omega_1rlvmore> rlv=construct_rlv2(p,C.x.user);
     MatrixXd Sc=MatrixXd::Zero(p,p);
     for (int i=0; i<rlv.size(); ++i) {
     int item1=rlv[i].item1;
     int item2=rlv[i].item2;
     int count=rlv[i].userno.size();
     s=0;s1=0;s2=0;
     for (int j=0; j<count; ++j) {
     s=s+rlv[i].firstrating[j]*rlv[i].secrating[j];
     s1+=rlv[i].firstrating[j];
     s2+=rlv[i].secrating[j];
     }
     s=(s-(double)s1*s2/count)/count;
     Sc(item1,item2)=s;Sc(item2,item1)=s;
     }
     if(train.item.size()<p) cout<<"Train data doesn't have all p movies, data division error!"<<endl;
     for (int i=0; i<p; ++i) {
     s=0;s1=0;
     int count=train.item[i].user.size();
     for (int j=0; j<train.item[i].user.size(); ++j){
     double temp=train.item[i].rating[j];
     s+=temp*temp;
     s1+=temp;
     }
     s=s-(double)s1*s1/count;
     s=s/count;
     Sc(i,i)=s;
     }
     
     get_MatrixtoData(Sc, "../simulation/rep100_prop/SampleCov.txt");//smallest eigenvalue of Sc is -2000, too small. Change Sc too much.
     
     
     rec.resize(lambda1.size()*lambda2.size()*lambda3.size(),5);
     alphanames.resize(lambda1.size()*lambda2.size()*lambda3.size());
     betanames.resize(lambda1.size()*lambda2.size()*lambda3.size());
     
     tstartall = time(0);
     for (int i=0; i<lambda1.size(); ++i) {
     C.lambda1=lambda1[i];
     for (int j=0; j<lambda2.size(); ++j){
     C.lambda2=lambda2[j];
    for (int k=0; k<lambda3.size(); ++k) {
        C.lambda3=lambda3[k];//C.lambda1;//lambda3[k];can skip this 2*, not too much difference
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
        method="v9 2lambda1";//rule1 is with rule lam1 series, rule0 is using bestalpha and bestbeta.
        if(!(i==0&&j==0&&k==0)) rule2assignstarter2(2,i,j,k,0,C.mubeta1,C.mualpha1,C.u,C.u2,C.theta,C.theta2,C.Omegais1,C.Zis,C.Uis,L);//define starting value
        
        re3=Cluster_p_inADMM_scale_struct_v9(c,C);//不同方法最后obj差很多，可能是因为L1问题它本身nonconvex，结果都是local解
        tend = time(0);
        double timecount=difftime(tend, tstart);
        cout << "It took " << timecount << " second(s)." << endl;
        
        int clu_alpha=cal_cluster_no(re3.mualpha);
        cout<<"clusters No. in alpha is "<<clu_alpha<<endl;
        int clu_beta=cal_cluster_no(re3.mubeta.transpose());
        cout<<"clusters No. in beta is "<<clu_beta<<endl;
        string namenow=lambdastr+"_"+lambda2str+"_"+lambda3str;
        rule2assignlayer2(2,i,j,k,0,lambda1.size(),lambda2.size(),lambda3.size(),C.mubeta1,C.mualpha1,C.u,C.u2,C.theta,C.theta2,C.Omegais1,C.Zis,C.Uis,L,namenow);//define L, rule2
        
        VectorXd final(13);
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
        get_sVectortoData(method,final,pathAndNameL1tune);
        
        if (i==0&&j==0&&k==0) {
            bestparaindex=0;
            bestw_tune_MSE=final[3];//use MSE_tune as criterion, because likelihood on tune cannot be calculated due to lack of Omega
            bestsolu=re3.solu;
            bestalpha=re3.mualpha;
            bestbeta=re3.mubeta;
            bestOmegais=re3.Omegais;
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
                bestalpha=re3.mualpha;
                bestbeta=re3.mubeta;
                bestOmegais=re3.Omegais;
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
        
    }
}
}

cout<<"bestparaindex is "<<bestparaindex<<endl;
C.lambda1=rec(bestparaindex,0);
C.lambda2=rec(bestparaindex,1);
C.lambda3=rec(bestparaindex,2);
get_MatrixtoData(bestalpha, alphanames[bestparaindex]);
get_MatrixtoData(bestbeta, betanames[bestparaindex]);
get_matricestoData(bestOmegais,Omeganames[bestparaindex]);
get_MatrixtoData(bestu,unames[bestparaindex]);
get_MatrixtoData(bestu2,u2names[bestparaindex]);
get_MatrixtoData(besttheta,thetanames[bestparaindex]);
get_MatrixtoData(besttheta2,theta2names[bestparaindex]);
get_matricestoData(bestUis,Unames[bestparaindex]);
get_matricestoData(bestZis,Znames[bestparaindex]);

final.resize(15);
final[0]=seedint;
final[1]=C.lambda1;
final[2]=C.lambda2;
final[3]=C.lambda3;
cout<<"bestbeta.block(0,0,5,5)="<<endl<<bestbeta.block(0,0,5,5)<<endl;
cout<<"bestalpha.block(0,0,5,5)="<<endl<<bestalpha.block(0,0,5,5)<<endl;
cout<<"bestsolu.block(0,0,5,5)="<<endl<<bestsolu.block(0,0,5,5)<<endl;

final[4]=cal_w_MSE(bestsolu,test,testM);//weighted MSE_test
cout<<"L1 best wMSE="<<final[4]<<", with best lambda1="<<C.lambda1<<", best lambda2="<<C.lambda2<<", best lambda3="<<C.lambda3<<endl;
MSE=cal_MSE(bestsolu,test);
cout<<"MSE test="<<MSE<<endl;
final[5]=MSE;//unweighted MSE_test
MSE=cal_MSE(bestsolu,train);
cout<<"MSE train="<<MSE<<endl;
final[6]=MSE;//unweighted MSE_train
final[7]=rec(bestparaindex,3);//clu_alpha
final[8]=rec(bestparaindex,4);//clu_beta
final[9]=(C.mualpha1-mualphatrue).norm()/sqrt(mualphatrue.rows()*mualphatrue.cols());
final[10]=(C.mubeta1-mubetatrue).norm()/sqrt(mubetatrue.rows()*mubetatrue.cols());

tendall = time(0);
cout<<"Total L1 cost "<<difftime(tendall, tstartall)<<" secs. \n";
final[11]=difftime(tendall, tstartall);
final[12]=C.rho2;
final[13]=C.rho;
final[14]=c;

get_sVectortoData(method,final, pathAndNameL1final);



/*if(remove(trainname.c_str())!=0||remove(tunename.c_str())!=0||remove(testname.c_str())!=0||remove(traintunename.c_str())!=0
 ||remove(name3.c_str())!=0||remove(name4.c_str())!=0||remove("alphasim_100_100_0.85_o2_caseg6_s01_2.txt")!=0||remove("betasim_100_100_0.85_o2_caseg6_s01_2.txt")!=0
 ||remove(alphatruename.c_str())!=0||remove(betatruename.c_str())!=0||remove("Sigmatune100_100_0.85_o2_caseg6_s01_2.txt")!=0
 ||remove("Sigmatest100_100_0.85_o2_caseg6_s01_2.txt")!=0||remove("Cortrain100_100_0.85_o2_caseg6_s01_2.txt")!=0
 ||remove("Omega1_true100_100_0.85_o2_caseg6_s01_2.txt")!=0||remove("Omega2_true100_100_0.85_o2_caseg6_s01_2.txt")!=0
 ||remove("Omegais_true100_100_0.85_o2_caseg6_s01_2.txt")!=0) perror("Error deleting file\n");*/

}

