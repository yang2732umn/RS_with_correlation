#include "consider_covariance.h"
void delete_proxy(proxy_var P){
    delete P.theta;
    delete P.u;
    delete P.theta2;
    delete P.u2;
    delete [] P.Zis;
    delete [] P.Uis;
}
