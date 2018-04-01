#include "consider_covariance.h"
double cal_MSE(const MatrixXd& solu, const rated_user_and_item& test){
    
    double MSE=0;
    vector<rated_user> user=test.user;
    int k=0;
    
    for (int i=0; i<user.size(); ++i) {
        for(int j=0;j<user[i].item.size();++j){
            double temp=solu(user[i].userno,user[i].item[j])-user[i].rating[j];
            MSE=MSE+temp*temp;
            k=k+1;
        }
    }
    MSE=sqrt(MSE/k);
    return MSE;
}
