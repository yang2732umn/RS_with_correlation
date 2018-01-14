#include "consider_covariance.h"
VectorXd readv(const string& filename){
    int n;
    ifstream in(filename.c_str());
    if (!in) {
        cout<<"file wrong!"<<endl;
    }
    
    in >> n ;//first line gives n and m of data
    VectorXd data(n);
    for(int loop=0;loop <n&& (in >> data(loop));)
    {
        ++loop;
    }
    in.close();
    return data;
}