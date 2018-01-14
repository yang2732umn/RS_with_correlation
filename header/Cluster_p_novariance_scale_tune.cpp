#include "consider_covariance.h"
VectorXd  Cluster_p_novariance_scale_tune(const Cluster_p_novariance_scale_tune_para &C){//include novariance and variance
    MatrixXd A1=readgn(C.trainname);//A is data
    MatrixXd A2=readgn(C.tunename);//A is data
    MatrixXd A3=readgn(C.traintunename);//A is data
    MatrixXd A4=readgn(C.testname);//A is data
    
    rated_user_and_item train=construct_user_item(A1);
    rated_user_and_item tune=construct_user_item(A2);
    rated_user_and_item traintune=construct_user_item(A3);
    rated_user_and_item test=construct_user_item(A4);
    
    VectorXd lambda1=C.lambda1;
    
    Cluster_p_novariance_scale_para C2;
    C2.x=train;
    C2.Tol=C.Tol;
    C2.rho=C.rho;
    C2.maxIter=C.maxIter;
    C2.movie=readgn(C.moviename);
    C2.users=readgn(C.usersname);
    C2.mualpha1=readgn(C.alphainitname);
    C2.mubeta1=readgn(C.betainitname);
    
    double cblambda1=lambda1[0];//current best lambda1
    double cbMSE_tune;//current best MSE_tune
    MatrixXd cbmualpha,cbmubeta;
    VectorXd MSE_tune(lambda1.size());
    VectorXd testre(10);
    time_t tstart, tend;
    
    tstart = time(0);
    for (int i=0; i<lambda1.size(); ++i) {
        C2.lambda1=lambda1[i];
        result re=LS_Lasso_Cluster_4_fan3_p_admm_std_ref(C2);
        MSE_tune[i]=cal_MSE(re.solu,tune);
        if (i==0) {
            cbMSE_tune=MSE_tune[i];
            cbmualpha=re.mualpha;
            cbmubeta=re.mubeta;
        }
        else {
            if (MSE_tune[i]<cbMSE_tune) {
                cbMSE_tune=MSE_tune[i];
                cblambda1=lambda1[i];
                cbmualpha=re.mualpha;
                cbmubeta=re.mubeta;
            }
        }
        C2.mualpha1=re.mualpha;
        C2.mubeta1=re.mubeta;
    }
    tend = time(0);
    double timecount=difftime(tend, tstart);
    cout << "Tuning took " << timecount << " second(s)." << endl;
    
    C2.x=traintune;
    C2.lambda1=cblambda1;
    C2.mualpha1=cbmualpha;
    C2.mubeta1=cbmubeta;
    
    size_t found=C.dname.find(".txt");
    string realpha=C.dname;
    realpha=realpha.erase(found,4);
    realpha=realpha+"_";
    ostringstream convert;   // stream used for the conversion
    convert << cblambda1;      // insert the textual representation of 'Number' in the characters in the stream
    string lambdastr = convert.str();
    realpha=realpha.append(lambdastr);
    realpha=realpha+"_novariance_alpha.txt";
    cout<<"realpha is "<<realpha<<endl;//
    
    found=realpha.find("alpha");
    string rebeta=realpha;
    rebeta.replace(found, 5, "beta");
    cout<<"rebeta is "<<rebeta<<endl;
    
    get_MatrixtoData(cbmualpha, realpha);
    get_MatrixtoData(cbmubeta, rebeta);
    
    MatrixXd solutemp=C2.users*cbmualpha+cbmubeta*C2.movie.transpose();
    double MSE_train=cal_MSE(solutemp,train);
    
    result re=LS_Lasso_Cluster_4_fan3_p_admm_std_ref(C2);
    double MSE_test=cal_MSE(re.solu,test);
    VectorXd final(7);
    final[0]=cblambda1;
    final[1]=MSE_train;
    final[2]=cbMSE_tune;
    final[3]=timecount;
    final[4]=MSE_test;
    int clu_alpha=cal_cluster_no(re.mualpha);
    cout<<"clusters No. in alpha is "<<clu_alpha<<endl;
    int clu_beta=cal_cluster_no(re.mubeta.transpose());
    cout<<"clusters No. in beta is "<<clu_beta<<endl;
    final[5]=clu_alpha;
    final[6]=clu_beta;
    
    cout<<"lambda  seq is "<<lambda1.transpose()<<endl;
    cout<<"MSEtune seq is "<<MSE_tune.transpose()<<endl;
    return final;
}
