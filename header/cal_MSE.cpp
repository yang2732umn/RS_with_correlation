#include "consider_covariance.h"
double cal_MSE(const MatrixXd& solu, const rated_user_and_item& test){
    //string A="train_predict.txt";
    //ofstream fichier(A.c_str(),ios_base::app);
    //fichier.precision(10);//set the number of significant digits to be 4 in the file output
    //fichier.setf(ios::fixed);
    
    double MSE=0;
    vector<rated_user> user=test.user;
    int k=0;
    
    for (int i=0; i<user.size(); ++i) {
        for(int j=0;j<user[i].item.size();++j){
            double temp=solu(user[i].userno,user[i].item[j])-user[i].rating[j];
            //fichier<<i<<" "<<user[i].rating[j]<<" "<<solu(user[i].userno,user[i].item[j])<< endl;
            MSE=MSE+temp*temp;
            k=k+1;
        }
    }
    //fichier.close();
    MSE=sqrt(MSE/k);
    return MSE;
}