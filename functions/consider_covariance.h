//  consider_covariance.h
//  consider_covariance
//
//  Created by Fan Yang on 7/22/14.
//  Copyright (c) 2014 Bonnie. All rights reserved.


#ifndef consider_covariance_h
#define consider_covariance_h


#include <iostream>
#include <fstream>
#include <sstream>
#include<iomanip>
#include <string>
#include <vector>
#include<math.h>//In C++, abs() and fabs() works the same, also sqrt() in math.h
#include <ctime>
#include <cstdlib>
#include<set>
//#include<random>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
//#include <Eigen/Eigenvalues>  //seems no effect
#include <chrono>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

/*struct Edge
 {
 int from,  weight;
 friend bool operator<(Edge a, Edge b)
 {
 return a.weight < b.weight;
 }
 };*/



struct rated_user{//user are dominant
    int userno;
    vector<int> item;
    VectorXd rating;//the two should have the same length
};

struct rated_item{//item dominant, for one item
    int itemno;
    vector<int> user;
    VectorXd rating;//the two should have the same length
};


struct rated_1itemmore{//for one item
    int itemno;
    vector<int> user;
    vector<int> numberforuser;
};



struct Omega_1rlv_simple{//pair info
    int item1;//No. of item
    int item2;
    vector<int> userno;//user No.'s, all users who rated these two simultaneously
};


struct Omega_1rlv{//pair info
    int item1;//No. of item
    int item2;
    vector<int> userno;//user No.'s, all users who rated these two simultaneously
    vector<int> firstin;//first item's ordering for each user
    vector<int> secin;//second item's ordering for each user
};


struct Omega_1rlvmore{//pair info
    int item1;//No. of item
    int item2;
    vector<int> userno;//user No.'s, all users who rated these two simultaneously
    vector<int> firstin;//first item's ordering for each user
    vector<int> secin;//second item's ordering for each user
    vector<double> firstrating;
    vector<double> secrating;
};



struct rated_user_and_item{
    vector<rated_user> user;
    vector<rated_item> item;
};


struct tri{
    VectorXi first;//wi=1/sigma_i^2
    VectorXi second;
};

struct Cluster_mnl_p_ADMM_scale_para{
    double lambda1,lambda2,lambda3,Tol,rho,rho2;
    int maxIter;
    rated_user_and_item x;
    MatrixXd movie,users,mualpha1,mubeta1;
    vector<MatrixXd> Omegais1;
};

struct Cluster_mnl_p_ADMM_scale_para2{//add components for ADMM proxy variables
    double lambda1,lambda2,lambda3,Tol,rho,rho2;
    int maxIter;
    rated_user_and_item x;
    MatrixXd movie,users,mualpha1,mubeta1,theta,u,theta2,u2;
    vector<MatrixXd> Omegais1,Zis,Uis;
};

struct Cluster_mnl_p_ADMM_scale_para2_2{//add log determinant of each Omegaisub[i]
    double lambda1,lambda2,lambda3,Tol,rho,rho2;
    int maxIter;
    rated_user_and_item x;
    VectorXd log_det;
    MatrixXd movie,users,mualpha1,mubeta1,theta,u,theta2,u2;
    vector<MatrixXd> Omegais1,Zis,Uis;
};

struct Cluster_mnl_p_ADMM_scale_para3{//for v12, use surrogate variable for w difference penalty
    double lambda1,lambda2,lambda3,Tol,rho,rho2;
    int maxIter;
    rated_user_and_item x;
    vector<int> itemmore_idle; vector<int> rlv_idle;
    VectorXi size_est_diag;
    VectorXi size_est_offdiag;
    vector<rated_1itemmore> itemmore;
    vector<Omega_1rlv> rlv;
    vector<MatrixXd> Omegais1;
    vector<vector<double>> X,Z,eta,y;
    MatrixXd movie,users,mualpha1,mubeta1,theta,u,theta2,u2;
};



struct proxy_var{
    MatrixXd *theta;
    MatrixXd *u;
    MatrixXd *theta2;
    MatrixXd *u2;
    MatrixXd *Zis;
    MatrixXd *Uis;
};

void delete_proxy(proxy_var P);

int existindex(const vector<int> & a, int b);

double cal_vec_pairdiff(const VectorXd& X);

vector<rated_1itemmore> construct_itemmore(const rated_user_and_item &A);

vector<Omega_1rlv_simple> construct_rlv_simple(int p,const vector<rated_user> &user);

vector<Omega_1rlv> construct_rlv(int p,const vector<rated_user> &user);

vector<Omega_1rlvmore> construct_rlv2(int p,const vector<rated_user> &user);

struct train_tune_test{
    rated_user_and_item train;
    rated_user_and_item tune;
    rated_user_and_item test;
};

struct layers{//layers of alpha, beta, Omega
    vector<MatrixXd> mualpha;
    vector<MatrixXd> mubeta;
    vector<vector<MatrixXd>> Omegais;
};

struct layers2{//layers of alpha, beta, Omega
    vector<MatrixXd> mualpha;
    vector<MatrixXd> mubeta;
    vector<MatrixXd> u;
    vector<MatrixXd> u2;
    vector<MatrixXd> theta;
    vector<MatrixXd> theta2;
    vector<vector<MatrixXd>> Omegais, Zis,Uis;
    vector<string> names;
};


struct layers3{//layers of alpha, beta, Omega
    vector<MatrixXd> mualpha;
    vector<MatrixXd> mubeta;
    vector<MatrixXd> u;
    vector<MatrixXd> u2;
    vector<MatrixXd> theta;
    vector<MatrixXd> theta2;
    vector<vector<MatrixXd>> Omegais;
    vector<vector<vector<double>>> X,Z,eta,y;
    vector<string> names;
};

void rule1assignstarter(int nlayer, int i,int j,MatrixXd &betastart,MatrixXd &alphastart, vector<MatrixXd>& Omegaisstart,const layers &L);//

void rule1assignlayer(int nlayer, int i,int j,int lam1size,const MatrixXd &betanow,const MatrixXd &alphanow ,const vector<MatrixXd> & Omegaisnow,layers &L);

void rule2assignstarter(int nlayer, int i,int j,int k,int l,MatrixXd &betastart,MatrixXd &alphastart, vector<MatrixXd>& Omegaisstart,const layers &L);

void rule2assignlayer(int nlayer, int i,int j,int k,int l,int lam1size,int lam2size,int lam3size,const MatrixXd &betanow,const MatrixXd &alphanow ,const vector<MatrixXd> & Omegaisnow,layers &L);
void rule2assignstarter2(int nlayer, int i,int j,int k,int l,MatrixXd &betastart,MatrixXd &alphastart, MatrixXd& ustart,MatrixXd& u2start, MatrixXd& thetastart,MatrixXd& theta2start, vector<MatrixXd>& Omegaisstart, vector<MatrixXd>& Zisstart,vector<MatrixXd>& Uisstart,const layers2 &L);
void rule2assignlayer2(int nlayer, int i,int j,int k,int l,int lam1size,int lam2size,int lam3size,const MatrixXd &betanow,const MatrixXd &alphanow,const MatrixXd& unow,const MatrixXd& u2now,const MatrixXd& thetanow,const MatrixXd& theta2now,const vector<MatrixXd> & Omegaisnow,const vector<MatrixXd> & Zisnow,const vector<MatrixXd> & Uisnow,layers2 &L, string &namenow);
void rule2assignstarter3(int nlayer, int i,int j,int k,int l,MatrixXd &betastart,MatrixXd &alphastart, MatrixXd& ustart,MatrixXd& u2start, MatrixXd& thetastart,MatrixXd& theta2start, vector<MatrixXd>& Omegaisstart, vector<vector<double>>& Xstart,vector<vector<double>>& Zstart,vector<vector<double>>& eta_start,vector<vector<double>>& y_start,const layers3 &L);
void rule2assignlayer3(int nlayer, int i,int j,int k,int l,int lam1size,int lam2size,int lam3size,const MatrixXd &betanow,const MatrixXd &alphanow,const MatrixXd& unow,const MatrixXd& u2now,const MatrixXd& thetanow,const MatrixXd& theta2now,const vector<MatrixXd> & Omegaisnow,const vector<vector<double>>& Xnow,const vector<vector<double>>& Znow,const vector<vector<double>>& eta_now,const vector<vector<double>>& y_now,layers3 &L,string &namenow);

struct result{
    double obj;
    double maxdiff;
    MatrixXd mualpha;
    MatrixXd mubeta;
    vector<MatrixXd> Omegais;
    MatrixXd solu;
    int normalstatus;//only used in TLP, =0 means normal, =1 means wrong
};

struct result2{
    double obj;
    double maxdiff;
    MatrixXd mualpha;
    MatrixXd mubeta;
    vector<MatrixXd> Omegais;
    MatrixXd solu;
    int normalstatus;//only used in TLP, =0 means normal, =1 means wrong
    proxy_var P;
};


struct dresult{
    double obj;
    double maxdiff;
    MatrixXd mualpha;
    MatrixXd mubeta;
    VectorXd wis;
    MatrixXd solu;
    int normalstatus;//only used in TLP, =0 means normal, =1 means wrong
};//double clustering result


inline int sign(double a){
    double b=0;
    if(a>0) b=1;
    if(a==0) b=0;
    if(a<0) b=-1;
    return b;
}
double ST(double a,double b);

VectorXd ST_vec(const VectorXd &a,double b);

VectorXd ST_vec_p(const VectorXd &a,double b);

MatrixXd ST_mat(const MatrixXd &a,double b);


template<class TYPE>//function template for stl:vector print
void stl_vec_cout(const vector<TYPE> &a){
    typename vector<TYPE>::const_iterator it;
    for(it=a.begin();it!=a.end();++it){
        cout<<*it<<" ";
    }
    cout<<endl;
}

template<class TYPE>//function template for stl:vector quick delete, this function doesn't maintain the orders of the remaining elements;this function is not used
vector<TYPE> stl_vec_quickDelete(int idx, vector<TYPE> vec){
    vec[idx] = vec.back();
    vec.pop_back();
    return vec;
}


MatrixXd absm(const MatrixXd &x);

VectorXd Vec_sub(const VectorXd & a, const vector<int> &b);

MatrixXd matrix_colsub(const MatrixXd &a, const vector<int> &b);

MatrixXd matrix_rowsub(const MatrixXd & a, const vector<int> &b);

MatrixXd matrix_rowcolsub(const MatrixXd &a, const vector<int> &b);

//子矩阵赋值
void matrix_rowcolsub_givevalue(MatrixXd &a, const vector<int> &b,const MatrixXd &c);


void get_MatrixtoData(const MatrixXd &X,const string& pathAndName);

void get_intMatrixtoData(const MatrixXi &X,const string& pathAndName);

void get_matricestoData(const vector<MatrixXd> &matrices,const string& pathAndName);

void get_vectoData(const VectorXd &X,const string& pathAndName);

VectorXd readv(const string& filename);

MatrixXd readgn(const string& filename);

vector<MatrixXd> readgn_seq(const string& filename,int nmat);//read a sequence of nmat

MatrixXi readgni(const string& filename);

template<typename T>
bool isnan(T value)
{
    return value != value;
    
}

VectorXi constructsdi(int i, int n);

//parameter variable format, const reference???
struct Cluster_TLP_p_scale_para{
    double lambda1,lambda2,lambda3,Tol,rho,rho2,tau;
    int maxIter;
    string realpha;
    rated_user_and_item x;
    MatrixXd movie,users,mualpha1,mubeta1;
    vector<MatrixXd> Omegais1;
};

//Cluster_TLP_p_scale_para_more is the same as Cluster_TLP_p_scale_para if ever used

struct Cluster_TLP_p_scale_para2{
    double lambda1,lambda2,lambda3,Tol,rho,rho2,tau;
    int maxIter;
    string realpha;
    rated_user_and_item x;
    MatrixXd movie,users,mualpha1,mubeta1,theta,u,theta2,u2;
    vector<MatrixXd> Omegais1,Zis,Uis;
    
};

inline double J_tau(double x,double tau){//min(|x|,tau)
    // double a=tau;
//     if (abs(x)<tau) {
//         a=abs(x);
//     }
	// return a;
	return min(abs(x),tau);
}

struct pairint{
    int first;
    int second;
};

struct pairdouble{
    double first;
    double second;
};


struct doc_obj_struct{//for inside doc
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double c6;
    double lambda1;
    double lambda2;
    double lambda3;
    double tau;
    double alphafuse;
    double betafuse;
    pairdouble Omegafusesize;
    MatrixXd *mualpha;
    MatrixXd *mubeta;
    vector<MatrixXd> *Omegais;
    vector<rated_user> *user;
    VectorXd *log_det;
};

struct TLP_obj_struct{
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double c6;
    double lambda1;
    double lambda2;
    double lambda3;
    double tau;
    double off_diff;
    VectorXd log_det;
    MatrixXd *mualpha;
    MatrixXd *mubeta;
    vector<MatrixXd> *Omegais;
    vector<rated_user> *user;
};

struct obj_struct{
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double c6;
    double lambda1;
    double lambda2;
    double lambda3;
    double Osize;
    double off_diff;
    MatrixXd *mualpha;
    MatrixXd *mubeta;
    VectorXd *log_det;
    vector<MatrixXd>* Omegais;
    vector<rated_user> *user;
};



struct TLP_obj_Omega_struct{
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double c6;
    double off_diff;
    double lambda2;
    double lambda3;
    double tau;
    MatrixXd movie;
    MatrixXd users;
    vector<rated_user> user;
    vector<MatrixXd> Sis;
    vector<MatrixXd> Omegais;
};

double cal_TLP_obj_Omega_struct(const TLP_obj_Omega_struct &A);//use double off_diff

double cal_TLP_offdiff(double tau,const vector<MatrixXd>& Omegais,const vector<Omega_1rlv> &rlv,const vector<rated_1itemmore>& itemmore);
    //this calculates TLP  diff terms for Omega
    
double matrix_Jtaunorm(double tau,const MatrixXd& X);

double po_matrix_Jtaunorm(double tau,const MatrixXd& X);

double matrix_diff_Jtaunorm(double tau,const MatrixXd& X,const MatrixXd& Y); 

double Vec_Jtaunorm(double tau,const VectorXd& X);

double vec_linfnorm(const vector<double> & x);//vector l-infinity norm

double vec_square_norm(const vector<double> & x);//vector square norm

vector<double> vec_diff(const vector<double> & x,const vector<double> & y);

double cal_TLP_offdiff_prop(double tau,const vector<MatrixXd>& Omegais);
    //this calculates TLP  diff terms for Omega
    
double cal_TLP_offdiff_prop2(double tau,vector<MatrixXd>* Omegais);
//use pointer to MatrixXd seq

double cal_seq_mat_maxdiff(const vector<MatrixXd> matseq1,const vector<MatrixXd> matseq2);
    //calculates the maxmimum difference between two seqs of matrices

VectorXd solve21(const VectorXd &y,double lambda, int maxIter, double Tol);// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)

VectorXd solve21_ama(const VectorXd &y,double lambda, int maxIter, double Tol);// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)

double D_gamma(const VectorXd &y,const VectorXd &lambda,double gamma);

double F_gamma(const VectorXd &y,const VectorXd &beta,double gamma);


VectorXd solve21_ama2(const VectorXd &y,double gamma, int maxIter, double Tol);// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
    //ama2 is similar to acc_ama, but just without acceleration
    //all connected


VectorXd solve21_acc_ama(const VectorXd &y,double gamma, int maxIter, double Tol);
// use ADMM to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
//all connected
//for each sample yi, it's a scalar




VectorXd solve21_ama2_prop(const VectorXd &y,int m,double gamma, int maxIter, double Tol);
// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
//this is for propagation version v4_3
//m is the number of u_i's whose y_i is 0.
//y is real y + last idle value(may not be 0)
//ama2 is similar to acc_ama, but just without acceleration
//all connected

VectorXd solve21_ama2_prop_p(const VectorXd &y,int m,double gamma, int maxIter, double Tol);


VectorXd solve21_acc_ama_ptl(int judge,double gamma, int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi);
// use acc ama to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
//judge is whether there is constraint or not, if >0, there is, if =0 there isn't
//partially connected
//for each sample yi, it's a scalar
//Vector2i first coordinate is index of the other, second coordinate is number of constraint
//also solves the case of gamma=0 correctly


VectorXd solve21_ama2_ptl(int judge,double gamma,int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi);
//not use acc
// use ama to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
//judge is whether there is constraint or not, if >0, there is, if =0 there isn't
//partially connected
//for each sample yi, it's a scalar
//Vector2i first coordinate is index of the other, second coordinate is number of constraint
//also solves the case of gamma=0 correctly


VectorXd solve21_ama2_ptl_prop(int m,double gamma, int maxIter, double Tol,const VectorXd &y,const vector<vector<Vector2i>>&cstrfdi,const vector<vector<Vector2i>>&cstrsdi);
//not use acc, last element is special case
//use ama to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
//used only if there is constraint
//partially connected
//for each sample yi, it's a scalar
//Vector2i first coordinate is index of the other, second coordinate is number of constraint
//also solves the case of gamma=0 correctly


inline void proj_linf(VectorXd &x, double gamma){
    for (int i=0; i<x.size(); i++) {
        x[i]=fmin(fmax(x[i],-gamma),gamma);
    }
}


double F_gamma_vec(const MatrixXd &y,const MatrixXd &beta,double gamma);
//vec version
//y is a matrix with samples in columns


MatrixXd solve21_acc_ama_vec(const MatrixXd &y,double gamma, int maxIter, double Tol);
// use ADMM to solve min1/2||y-beta||^2+lambda\sum_{i<k}|beta_i-beta_k|  (equation 21)
//all connected
//samples are in cols!!p is sample size
//need parallize within this function


MatrixXd solve21_acc_ama_vec_ptl(const MatrixXd &y,double gamma, int maxIter, double Tol, const vector<vector<int>> diffB);
// use acc ama to solve min1/2||y-beta||^2+gamma\sum_{i<k}|beta_i-beta_k|  (equation 21)
//for any two pair, difference of different subvector is penalized
//diffB is size p(p-1)/2, gives for each pair, which differences are penalized
//samples are in cols of y!!p is sample size
//need parallize within this function


//MatrixXi  constructpair(vector<Omega_1rlv> rlv,){}

//consider no ties in A
VectorXd solve21_hock2(VectorXd &A,double lambda);
//faster than hock when K is large


VectorXd solve21_hock3(VectorXd &A,double lambda);
//faster than hock when K is large

VectorXd solve21_hock(VectorXd &A,double lambda);


VectorXd solve21_ama_ptl(const VectorXd &y,const MatrixXi &pair,double lambda, int maxIter, double Tol);
// use ADMM solve min1/2||y-beta||^2+lambda||D beta||_1  (equation 21)
//partially connected


double g(const MatrixXd& Omega,const vector<rated_user>& user,double lambda3,const vector<MatrixXd>& Sis);
//not used


ArrayXd Arraysign(const ArrayXd &x);
//not used

void get_VectortoData(const VectorXd &X,const string& pathAndName);

void get_sVectortoData(const string &s,const VectorXd &X,const string& pathAndName);

VectorXi unique_veci(const VectorXi &a);
//get unique values of a VectorXi



rated_user_and_item construct_user_item(const MatrixXi &data);
//this function gives the user item observed pairs from the ...*3 data matrix
//user and item are all ordered

//reload of the above function for double matrix
rated_user_and_item construct_user_item(const MatrixXd &data);

vector<rated_item> construct_item_from_user(const vector<rated_user> &user);

inline int myrandom (int i){
    return rand()%i;
}


double cal_obj_prop_struct_3(const vector<MatrixXd>& Sis,const obj_struct &A);
//lambda3=lambda1, used for with L1 penalty on Omega 

double cal_obj_prop_struct_5(const vector<MatrixXd>& Sis,const obj_struct &A);


double cal_obj_prop_struct_4(double c,const tri &trin,const tri &trip,const MatrixXd& betapre,const MatrixXd& alphapre,const vector<MatrixXd>& Omegaispre,const vector<MatrixXd>& Sis,const obj_struct &A);
//lambda3=lambda1
//use for inner doc(admm) objective function for L1 version with L1 penalty on Omega (Cluster_p_inADMM_scale_struct_v6)
//c is the constant used for quadratic term, how to determine/solve for c
double cal_obj_prop_struct_6(double c,const tri &trin,const tri &trip,const MatrixXd& betapre,const MatrixXd& alphapre,const vector<MatrixXd>& Omegaispre,const vector<MatrixXd>& Sis,const obj_struct &A);


double cal_offdiff(const vector<MatrixXd>& Omegais,const vector<Omega_1rlv>& rlv,const vector<rated_1itemmore>& itemmore);
//this calculates diff terms for Omega

double cal_offdiff_prop(const tri &trin,const vector<MatrixXd>& Omegais);
//this calculates diff terms for Omega
//Omegais is n p*p, _prop is for version with propogation


double cal_TLP_obj_struct(const TLP_obj_struct &A);

double cal_TLP_obj_struct_prop(const TLP_obj_struct &A);

struct obj_Omega2_struct{
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double c6;
    double off_diff;
    double lambda2;
    double lambda3;
    MatrixXd movie;
    MatrixXd users;
    vector<rated_user> user;
    vector<MatrixXd> Sis;
    vector<MatrixXd> Omegais;
    
};


double cal_obj_Omega2_struct(const obj_Omega2_struct &A);
//use double off_diff


struct two_lambda_struct{
    double lambda1;
    double lambda2;
    MatrixXd mualpha;
    MatrixXd mubeta;
    MatrixXd movie;
    MatrixXd users;
    vector<rated_user> user;
};

double cal_obj2_2lambda_struct(const two_lambda_struct &A);

double cal_novariance_likelihood(MatrixXd mualpha, MatrixXd mubeta, vector<rated_user> user, MatrixXd movie, MatrixXd users);

double cal_obj2_2lambda(const MatrixXd& mualpha,const MatrixXd& mubeta, double lambda1, double lambda2,const vector<rated_user>& user,const MatrixXd& movie, const MatrixXd& users);

struct obj_beta_struct{//beta and alpha has the same struct
    double lambda1;
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double c6;
    MatrixXd mualpha;
    MatrixXd mubeta;
    MatrixXd movie;
    MatrixXd users;
    vector<MatrixXd> Omegais;
    vector<rated_user> user;
};

struct obj_alpha_inner_struct{//beta and alpha has the same struct
    double rho_alpha;
    MatrixXd mualpha;
    MatrixXd mubeta;
    MatrixXd movie;
    MatrixXd users;
    MatrixXd* theta2;
    MatrixXd* u2;
    vector<MatrixXd> Omegais;
    vector<rated_user> user;
};

double cal_obj_beta_struct(const obj_beta_struct &A);
//modified to l2 norm for TLP
//user should be train.user



/*double cal_obj_beta(const doc_obj_struct &B,const vector<vector<int>>&diffB,MatrixXd* theta,MatrixXd* u){
 //user should be train.user
 int n=B.users.rows();
 int p=B.movie.rows();
 MatrixXd solu=B.users*B.mualpha+B.mubeta*B.movie.transpose();
 double obj=0,temp=0;
 VectorXd resid;
 # pragma omp parallel private (resid)
 {
 # pragma omp for reduction(+:obj)
 for (int i=0; i<mubeta.rows()-1; ++i) {
 for(int j=i+1;j<mubeta.rows();++j){
 l=i*(n-(i+1)*0.5)+j-i-1;
 if(diffB[l].size()>0) obj=obj+(mubeta.row(i)-mubeta.row(j)-(*theta).col(l)+(*u).col(l)).squaredNorm();
 }
 }
 # pragma omp single
 {
 obj=obj*lambda1/(2*c4);//
 }
 
 # pragma omp for reduction(+:temp)
 for (int i=0; i<user.size(); ++i) {
 resid.resize(user[i].item.size());
 for(int j=0;j<user[i].item.size();++j){
 resid[j]=solu(user[i].userno,user[i].item[j])-user[i].rating[j];
 }
 MatrixXd Si=resid*resid.transpose();
 MatrixXd Omegai=Omegais[i];
 if(isnan(Omegai)) {
 cout<<"NaN exists in Omegais[i]"<<endl;
 cout<<"Omegais["<<i<<"] is "<<endl;
 cout<<Omegai<<endl;
 }
 temp=temp+((Si*Omegai).trace());
 }
 }
 obj=obj+temp*0.5/c1;
 obj=obj*100;
 return obj;
 }*/


double cal_obj_alpha_inner_struct(const obj_alpha_inner_struct &A,const vector<vector<int>>& diffA);//modified for TLP


double cal_obj_alpha_struct(const obj_beta_struct &A);


int cal_cluster_no(const MatrixXd &X);
//X is a matrix with n columns

int cal_cluster_no2(const MatrixXd &X);
//X is a matrix with n*(n-1)/2 columns

double cal_MSE(const MatrixXd& solu, const rated_user_and_item& test);

double cal_w_MSE(const MatrixXd &solu, const rated_user_and_item & test,const vector<MatrixXd> & Omegais);

double cal_w_d_MSE(const MatrixXd &solu, const rated_user_and_item& test,const VectorXd& wis);
//diagnonal Omega=wis


double cal_w_MSE2(const MatrixXd &solu, const rated_user_and_item & test,const vector<MatrixXd> & Omegais);
//only weight by diagonal of Omega

MatrixXd constructDiT(int m, int mi,const vector<int> &a);




vector<VectorXd> cal_resid(const MatrixXd &mualpha,const MatrixXd &mubeta,const vector<rated_user> &user,const MatrixXd& movie,const MatrixXd& users);
//calculate residuals for each user

double cal_likelihood(const Cluster_mnl_p_ADMM_scale_para &C);

double cal_likelihood(const Cluster_TLP_p_scale_para &C);

double cal_likelihood_prop(const Cluster_mnl_p_ADMM_scale_para &C);

double cal_likelihood_prop(const MatrixXd &users,const MatrixXd &movie, const rated_user_and_item &data,const MatrixXd &mualpha,const MatrixXd &mubeta,const vector<MatrixXd> &Omegais);

double cal_likelihood_prop(const Cluster_TLP_p_scale_para &C);

struct dcluster_var_l1_para{
    int maxIter;
    int udim2;
    int mdim2;
    double lambda1,lambda2,lambda3,Tol,rho,rho2;//lambda2 is about sparsity penalty, always set 0
    VectorXd wis1;//wi=1/sigma_i^2
    rated_user_and_item x;
    MatrixXd movie,users,mualpha1,mubeta1;
};

double cal_likelihood_d(const dcluster_var_l1_para &C);
 //diagonal precision matrix, each person different sigma

result Cluster_p_inADMM_scale_struct_v9(double c,Cluster_mnl_p_ADMM_scale_para2 &C);

result Cluster_p_inADMM_scale_struct_v9_2(double c,Cluster_mnl_p_ADMM_scale_para2 &C);

result Cluster_p_inADMM_scale_struct_v9_3(double c,Cluster_mnl_p_ADMM_scale_para2_2 &C);

result Cluster_p_inADMM_scale_struct_v9_4(double c,Cluster_mnl_p_ADMM_scale_para2_2 &C);//A.off_diff calculated faster, differently

result Cluster_p_inADMM_scale_struct_v11(double c,Cluster_mnl_p_ADMM_scale_para2 &C);

result Cluster_p_inADMM_scale_struct_v12(double c,Cluster_mnl_p_ADMM_scale_para3 &C);

double cal_TLP_docbeta_obj_struct(const TLP_obj_struct &A,const vector<vector<int>>& diffB);

double cal_TLP_docalpha_obj_struct(const TLP_obj_struct &A,const vector<vector<int>>& diffA);

double cal_TLP_docbeta_obj_struct_prop(const TLP_obj_struct &A, const vector<vector<int>>& diffB);

double cal_TLP_docalpha_obj_struct_prop(const TLP_obj_struct &A,const vector<vector<int>>& diffA);


double cal_alpha_fuse_abs(const tri &trip,const MatrixXd &mualpha,const vector<vector<int>> &diffA);

double cal_beta_fuse_abs(const tri &trin,const MatrixXd &mubeta,const vector<vector<int>> &diffB);

pairdouble cal_Omega_fusesize_abs(const vector<MatrixXd> &Omegais,const vector<vector<vector<Vector2i>>> &cstrfdi_diag,const vector<vector<vector<Vector2i>>> &cstrsdi_diag, const vector<vector<vector<Vector2i>>> &cstrfdi_offdiag,const vector<vector<vector<Vector2i>>> &cstrsdi_offdiag,const vector<int> &judge_diag,const vector<int> &judge_offdiag,const vector<vector<int>> &Osize,const  vector<Omega_1rlv_simple>& rlv,const vector<rated_1itemmore> &itemmore, const vector<int> &itemmore_idle,const vector<int> &rlv_idle);
//both fuse and size on Omega abs

double cal_doc_obj_struct_prop_abs(const doc_obj_struct &A);  
//user should be train.user
//with L1 penalty on Omega size also
//within doc, using previous alpha, beta and Omega to determine set
//diffA is for alpha, diffB for beta, diffO is for Omega


double cal_doc_obj_struct_prop_abs2(double c,const MatrixXd& betapre,const MatrixXd& alphapre,const vector<MatrixXd>& Omegaispre,const vector<MatrixXd>& Sis,const doc_obj_struct &A);
//inside doc, abs obj value, but with quadratic adjustment for non-convexity
//used in Cluster_TLP_scale_prop_size3
//user should be train.user
//with L1 penalty on Omega size also
//within doc, using previous alpha, beta and Omega to determine set
//diffA is for alpha, diffB for beta, diffO is for Omega


double cal_TLP_obj_prop_size(const vector<MatrixXd>& Sis,const TLP_obj_struct &A);  
//also with size contraint on Omega
  
double cal_TLP_obj_prop_size_wi(int n,int c1,int c3,int c4,int c5,int c6,double lambda1,double lambda2,double lambda3,double tau,
	double alphafuse_tau,double betafuse_tau,double Omegasize_tau,double Omegafuse_tau, const VectorXd &log_det,
	const vector<MatrixXd>& Sis,const vector<rated_user> &user,const vector<MatrixXd>& Omegais);//with info

double cal_obj_abo(double rho,double rho2,const TLP_obj_struct &A,const vector<vector<int>>& diffA,const vector<vector<int>>& diffB,MatrixXd* theta2,MatrixXd* u2,MatrixXd* theta,MatrixXd* u,const vector<MatrixXd>& Zis,const vector<MatrixXd>& Uis);
//this calculates the part for alpha, beta and Omega


result Cluster_TLP_p_scale_struct_v2(Cluster_TLP_p_scale_para &C/*,int check*/);
//this function is first version of TLP
//v2 has no size constraint on off-diag of Omega



result Cluster_TLP_p_scale_struct_v2_prop(Cluster_TLP_p_scale_para &C/*,int check*/);
//this function is first version of TLP
//v2 has no size constraint on off-diag of Omega


result Cluster_TLP_p_scale_struct_v2_prop_2(Cluster_TLP_p_scale_para &C/*,int check*/);
//prop_2 does lambda/2c5(sum(J_tau(x,tau))), without dividing tau, parameter close to L1
//this function is first version of TLP
//v2 has no size constraint on off-diag of Omega


result Cluster_TLP_p_scale_prop_size(Cluster_TLP_p_scale_para &C/*,int check*/);
//prop_2 does lambda/2c5(sum(J_tau(x,tau))), without dividing tau, parameter close to L1
//v2 has no size constraint on off-diag of Omega
//x should be train data
//this has J_tau penalty on Omega size as well
//put d.o.c at the outermost iteration
//use maximum block inside doc


result Cluster_TLP_scale_prop_size2(Cluster_TLP_p_scale_para &C);
//unparallel version
//does propagation of Omega
//with L1 penalty on Omega
//TLP, Apply ADMM together for alpha, beta and Omega
//C.x should be train data


result Cluster_TLP_scale_prop_size3(double c,Cluster_TLP_p_scale_para &C);
//unparallel version
//_size3 is using quadratic terms to adjust for non-convexity, c is the constant that times the quadratic terms(in scaled loss function, the constant is c/c1)
//does propagation of Omega
//with L1 penalty on Omega
//TLP, Apply ADMM together for alpha, beta and Omega
//C.x should be train data

result Cluster_TLP_scale_prop_size4(double c,Cluster_TLP_p_scale_para &C);
//unparallel version
//_size3 is using quadratic terms to adjust for non-convexity, c is the constant that times the quadratic terms(in scaled loss function, the constant is c/c1)
//does propagation of Omega
//with L1 penalty on Omega
//TLP, Apply ADMM together for alpha, beta and Omega
//C.x should be train data
//difference from _size3: inside iter_ori, only run once for alpha_inner and beta_inner

result Cluster_TLP_scale_prop_size5(double c,Cluster_TLP_p_scale_para2 &C);
//difference from _size4: use initialization for theta,u, theta2,u2,Zis, Uis

result Cluster_TLP_scale_prop_size6(double c,Cluster_TLP_p_scale_para2 &C);


result LS_Lasso_Cluster_4_fan3_p_admm_2lambda(rated_user_and_item x, MatrixXd movie, MatrixXd users, MatrixXd mualpha1, MatrixXd mubeta1, double lambda1, double lambda2, double rho,double Tol, int maxIter);
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part
//lambda1 for beta, and lambda2 for alpha

result LS_Lasso_Cluster_4_fan3_p_admm_std(rated_user_and_item x, MatrixXd movie, MatrixXd users, MatrixXd mualpha1, MatrixXd mubeta1, double lambda1, double rho,double Tol, int maxIter);
//this is novariance
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood,
//also deal with if parallel computing of alpha doesn't work


result LS_Lasso_Cluster_p_inadmm_std(const rated_user_and_item & x, const MatrixXd & movie, const MatrixXd & users, MatrixXd mualpha1, MatrixXd mubeta1, double & lambda1, double & rho,double  Tol, int maxIter);
//this is novariance
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood
//also deal with if parallel computing of alpha doesn't work
//use inexact admm


result LS_Lasso_Cluster_p_inadmm_std2(const rated_user_and_item & x, const MatrixXd & movie, const MatrixXd & users, MatrixXd mualpha1, MatrixXd mubeta1, double & lambda1, double & rho,double  Tol, int maxIter);
//this is novariance
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood
//also deal with if parallel computing of alpha doesn't work
//use inexact admm

result LS_Lasso_Cluster_p_inadmm_std3(const rated_user_and_item & x, const MatrixXd & movie, const MatrixXd & users, MatrixXd& mualpha1, MatrixXd& mubeta1, double & lambda1, double & rho,double  Tol, int maxIter);
//this is novariance
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood
//also deal with if parallel computing of alpha doesn't work
//use inexact admm
//first use r=0.8, after converge, use r=1


result LS_Lasso_Cluster_4_fan3_p_admm_std_2(rated_user_and_item x, MatrixXd movie, MatrixXd users, MatrixXd mualpha1, MatrixXd mubeta1, double lambda1, double rho,double Tol, int maxIter,int factor);
//this is novariance
//factor means lambda_alpha=lambda1, lambda_beta=lambda1/factor, changed 3 places
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood,
//also deal with if parallel computing of alpha doesn't work



result LS_Lasso_Cluster_4_fan3_p_admm_std_3(rated_user_and_item x, MatrixXd movie, MatrixXd users, MatrixXd mualpha1, MatrixXd mubeta1, double lambda1, double lambda2, double rho,double Tol, int maxIter);
//this is novariance
//factor means lambda_alpha=lambda1, lambda_beta=lambda2, changed 3 places
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood,
//also deal with if parallel computing of alpha doesn't work


struct Cluster_p_novariance_scale_para{
    double lambda1,Tol,rho;
    int maxIter;
    rated_user_and_item x;
    MatrixXd movie,users,mualpha1,mubeta1;
};


result LS_Lasso_Cluster_4_fan3_p_admm_std_ref(const Cluster_p_novariance_scale_para &C);
//this is novariance ref version //
//fix Omega at I
//the same result as Cluster_mnl_p_ADMM function comment out Omega update part, standardize penalty and likelihood,




struct Cluster_p_novariance_scale_tune_para{
    int maxIter;
    double Tol,rho;
    string trainname,tunename,testname,traintunename,moviename,usersname,alphainitname,betainitname,dname;
    VectorXd lambda1;
};


VectorXd  Cluster_p_novariance_scale_tune(const Cluster_p_novariance_scale_tune_para &C);



struct Cluster_p_var_scale_tune_para{
    int maxIter;
    double Tol,rho,rho2;
    string trainname,tunename,testname,traintunename,moviename,usersname,alphainitname,betainitname;
    VectorXd lambda1,lambda2,lambda3;
};


VectorXd  Cluster_p_var_scale_tune(const Cluster_p_var_scale_tune_para &C);





struct dclust_var_l1_obj_struct{
    int udim2,mdim2;
    double c1;
    double c3;
    double c4;
    double c5;
    double c6;
    double c7;
    double lambda1;
    double lambda2;
    double lambda3;
    MatrixXd mualpha;
    MatrixXd mubeta;
    VectorXd wis;
    vector<rated_user> user;
    MatrixXd movie;
    MatrixXd users;
    vector<rated_1itemmore> itemmore;
};



double cal_dcluster_l1_obj_struct(const dclust_var_l1_obj_struct& A);


void construct_fdisdi(const int& n,vector<VectorXi> &fdi,vector<VectorXi> &sdi);
//for all i from 1 to n
//fdi and sdi should be of size n-1 already

void construct_Zinv(const int& n,const int& mdim,const int& mdim2,MatrixXd& Zinv, MatrixXd& A2);
//matrix references should be in right size already
//Zinv(n*mdim,n*mdim); A1(n*(n-1)/2*mdim,n*mdim); A2(n*mdim2*(mdim2-1)/2,n*mdim)
//use parallel inside, because there will be no further parallism outside


void construct_t1_A1t2_A2t3(const int& n,const int& mdim,const int& mdim2,const vector<VectorXi>& fdibeta,const vector<VectorXi>& sdibeta,const vector<VectorXi>& fdibetainner,const vector<VectorXi>& sdibetainner,const MatrixXd& mubeta2, MatrixXd* gamma,MatrixXd* theta,MatrixXd* u,MatrixXd* eta,MatrixXd* v, VectorXd &t1,VectorXd& A1t2,VectorXd& A2t3);


dresult dcluster_scale_struct_l1_v1(const int outer_first, dcluster_var_l1_para &C);
//only estimates different wi=1/sigma_i^2, use double clustering on alpha and beta, L1 penalty
//for alpha,beta, ADMM is multiblock(convex), linear convergence
//for wi, ADMM is 2 block
//double clustering is only for part of alpha and beta, last udim2 and mdim2 elements
//reference as para, change C value


dresult dcluster_scale_struct_l1_v2(const int outer_first, dcluster_var_l1_para &C);
//only estimates different wi=1/sigma_i^2, use double clustering on alpha and beta, L1 penalty
//for alpha,beta, ADMM is multiblock(convex), linear convergence
//for wi, ADMM is 2 block
//double clustering is only for part of alpha and beta, last udim2 and mdim2 elements

vector<MatrixXd> l1_solve_Omega(double lambda2,double lambda3,double rho2,const MatrixXd &mualpha,const MatrixXd &mubeta,const vector<rated_user>& user, 
    const vector<rated_1itemmore> &itemmore,const vector<Omega_1rlv_simple> &rlv,const vector<MatrixXd> & Sis);


#endif
