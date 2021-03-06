#include "consider_covariance.h"
double cal_w_MSE(const MatrixXd &solu, const rated_user_and_item & test,const vector<MatrixXd> & Omegais){
    double MSE=0;
    vector<rated_user> user=test.user;
    int k=0,l=0;
    for (int i=0; i<user.size(); ++i) {
        VectorXd resi(user[i].item.size());
        for(int j=0;j<user[i].item.size();++j){
            resi[j]=solu(user[i].userno,user[i].item[j])-user[i].rating[j];
            ++k;
        }
        MSE=MSE+resi.transpose()*Omegais[l]*resi;
        ++l;
    }
    MSE=sqrt(MSE/k);
    return MSE;
}
