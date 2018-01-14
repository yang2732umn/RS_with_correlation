#include "consider_covariance.h"
pairdouble cal_Omega_fusesize_abs(const vector<MatrixXd> &Omegais,const vector<vector<vector<Vector2i>>> &cstrfdi_diag,const vector<vector<vector<Vector2i>>> &cstrsdi_diag, const vector<vector<vector<Vector2i>>> &cstrfdi_offdiag,const vector<vector<vector<Vector2i>>> &cstrsdi_offdiag,const vector<int> &judge_diag,const vector<int> &judge_offdiag,const vector<vector<int>> &Osize,const  vector<Omega_1rlv_simple>& rlv,const vector<rated_1itemmore> &itemmore, const vector<int> &itemmore_idle,const vector<int> &rlv_idle){
    pairdouble M;
    double Omegafuse=0,Omegasize=0;
    int n=Omegais.size();
    int p=judge_diag.size();
    int i,j,usersize,userno1,userno2,l,k,item1,item2,userno,tt,index;
    vector<vector<Vector2i>> cstrfdi1;
#pragma omp parallel for private(usersize,userno1,userno2,cstrfdi1,l,tt,index) reduction(+:Omegafuse)
    for (i=0; i<p; ++i) {//diagonal first
        if (judge_diag[i]>0) {
            usersize=itemmore[i].user.size();
            if (usersize==1) {
                userno1=itemmore[i].user[0];
                userno2=itemmore_idle[i];
                Omegafuse+=(n-1)*abs(Omegais[userno1](i,i)-Omegais[userno2](i,i));
            }
            if (usersize>1) {
                cstrfdi1=cstrfdi_diag[i];
                for (l=0; l<usersize; ++l) {//l should be private
                    userno1=itemmore[i].user[l];
                    for (tt=0; tt<cstrfdi1[l].size(); ++tt) {//only fdi, shouldn't be any repetition
                        index=cstrfdi1[l][tt](0);
                        if (index<usersize) {
                            userno2=itemmore[i].user[index];
                            Omegafuse+=abs(Omegais[userno1](i,i)-Omegais[userno2](i,i));
                        }
                        else{
                            userno2=itemmore_idle[i];
                            Omegafuse+=(n-usersize)*abs(Omegais[userno1](i,i)-Omegais[userno2](i,i));
                        } 
                    }
                }
            }
        } 
    }
#pragma omp parallel for private(usersize,i,j,userno1,userno2,cstrfdi1,l,tt,index) reduction(+:Omegafuse)
    for (k=0; k<rlv.size(); ++k) {//off-diagonal
        if (judge_offdiag[k]>0) {
            usersize=rlv[k].userno.size();
            i=rlv[k].item1;
            j=rlv[k].item2;
            if (usersize==1) {
                userno1=rlv[k].userno[0];
                userno2=rlv_idle[k];
                Omegafuse+=(n-1)*abs(Omegais[userno1](i,j)-Omegais[userno2](i,j));
            }
            if (usersize>1) {
                cstrfdi1=cstrfdi_offdiag[k];
                for (l=0; l<usersize; ++l) {
                    userno1=rlv[k].userno[l];
                    for (tt=0; tt<cstrfdi1[l].size(); ++tt) {
                        index=cstrfdi1[l][tt](0);
                        if (index<usersize) {
                            userno2=rlv[k].userno[index];
                            Omegafuse+=abs(Omegais[userno1](i,j)-Omegais[userno2](i,j));
                        }
                        else{
                            userno2=rlv_idle[k];
                            Omegafuse+=(n-usersize)*abs(Omegais[userno1](i,j)-Omegais[userno2](i,j));
                        }
                        
                    }
                }
            }
        }
    }
    
#pragma omp parallel for private(item1,item2,usersize,k,index,userno) reduction(+:Omegasize)
    for (i=0; i<rlv.size(); ++i) {
        item1=rlv[i].item1;
        item2=rlv[i].item2;
        usersize=rlv[i].userno.size();
        for (k=0; k<Osize[i].size(); ++k) {
            index=Osize[i][k];
            if (index<usersize) {
                userno=rlv[i].userno[index];
                Omegasize+=abs(Omegais[userno](item1,item2));
            }
            else{
                userno=rlv_idle[i];
                Omegasize+=(n-usersize)*abs(Omegais[userno](item1,item2));
            }
        }
    }
    M.first=Omegafuse;
    M.second=Omegasize;
    return M;
}




