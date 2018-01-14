#include "consider_covariance.h"
double cal_likelihood_prop(const Cluster_mnl_p_ADMM_scale_para &C){//negative log likelihood, the smaller the better
    int n=C.users.rows();
    int p=C.movie.rows();
    MatrixXd solu=C.users*C.mualpha1+C.mubeta1*C.movie.transpose();
    double temp3=0;
    VectorXd resid;
    # pragma omp parallel private (resid)
    {
        # pragma omp for reduction(+:temp3)
        for (int i=0; i<C.x.user.size(); ++i) {
            resid.resize(C.x.user[i].item.size());
            for(int j=0;j<C.x.user[i].item.size();++j){
                resid[j]=solu(C.x.user[i].userno,C.x.user[i].item[j])-C.x.user[i].rating[j];
            }
            MatrixXd Si=resid*resid.transpose();
            MatrixXd Omegai=matrix_rowcolsub(C.Omegais1[i],C.x.user[i].item);
            //cout<<"On tune set, resid for user "<<i<<" is "<<resid.transpose()<<endl;
            //cout<<"Omegai="<<endl<<Omegai<<endl;
            if(i==75){
                cout<<"C.x.user[75].item is "<<endl;
                stl_vec_cout(C.x.user[75].item);
                cout<<"Omegai[75]="<<endl;
                cout<<Omegai<<endl;
            }
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
            if (Omegai.determinant()<0) {
                cout<<"Omega "<<i<<" negative determined!"<<endl;
            }
            if (Omegai.determinant()==0) {
                cout<<"Omega "<<i<<" undetermined!"<<endl;
            }
            temp3=temp3+(Si*Omegai).trace()-log((Omegai).determinant());
        }
        //can use temp3=temps/C.x.user.size(); to let the quantity has a smaller scale
    }
    return temp3;
    
}


double cal_likelihood_prop(const MatrixXd &users,const MatrixXd &movie, const rated_user_and_item &data,const MatrixXd &mualpha,const MatrixXd &mubeta,const vector<MatrixXd> &Omegais){//negative log likelihood, the smaller the better//data is in form of tune
    int n=users.rows();
    int p=movie.rows();
    MatrixXd solu=users*mualpha+mubeta*movie.transpose();
    double temp3=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:temp3)
        for (int i=0; i<data.user.size(); ++i) {
            resid.resize(data.user[i].item.size());
            for(int j=0;j<data.user[i].item.size();++j){
                resid[j]=solu(data.user[i].userno,data.user[i].item[j])-data.user[i].rating[j];
            }
            MatrixXd Si=resid*resid.transpose();
            MatrixXd Omegai=matrix_rowcolsub(Omegais[i],data.user[i].item);
            //cout<<"On tune set, resid for user "<<i<<" is "<<resid.transpose()<<endl;
            //cout<<"Omegai="<<endl<<Omegai<<endl;
            if(i==75){
                cout<<"data.user[75].item is "<<endl;
                stl_vec_cout(data.user[75].item);
                cout<<"Omegai[75]="<<endl;
                cout<<Omegai<<endl;
            }
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
            if (Omegai.determinant()<0) {
                cout<<"Omega "<<i<<" negative determined!"<<endl;
            }
            if (Omegai.determinant()==0) {
                cout<<"Omega "<<i<<" undetermined!"<<endl;
            }
            temp3=temp3+(Si*Omegai).trace()-log((Omegai).determinant());
        }
        //can use temp3=temps/data.user.size(); to let the quantity has a smaller scale
    }
    return temp3;
}


double cal_likelihood_prop(const Cluster_TLP_p_scale_para &C){
    int n=C.users.rows();
    int p=C.movie.rows();
    MatrixXd solu=C.users*C.mualpha1+C.mubeta1*C.movie.transpose();
    double temp3=0;
    VectorXd resid;
# pragma omp parallel private (resid)
    {
# pragma omp for reduction(+:temp3)
        for (int i=0; i<C.x.user.size(); ++i) {
            resid.resize(C.x.user[i].item.size());
            for(int j=0;j<C.x.user[i].item.size();++j){
                resid[j]=solu(C.x.user[i].userno,C.x.user[i].item[j])-C.x.user[i].rating[j];
            }
            MatrixXd Si=resid*resid.transpose();
            MatrixXd Omegai=matrix_rowcolsub(C.Omegais1[i],C.x.user[i].item);
            if(isnan(Omegai)) {
                cout<<"NaN exists in Omegais[i]"<<endl;
                cout<<"Omegais["<<i<<"] is "<<endl;
                cout<<Omegai<<endl;
            }
            if (Omegai.determinant()<0) {
                cout<<"Omega "<<i<<" negative determined!"<<endl;
            }
            if (Omegai.determinant()==0) {
                cout<<"Omega "<<i<<" undetermined!"<<endl;
            }
            if (i==32) {
                cout<<"Omega 32="<<endl<<Omegai<<endl;
            }
            temp3=temp3+(Si*Omegai).trace()-log((Omegai).determinant());
        }
    }
    return temp3;
}