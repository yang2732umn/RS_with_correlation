#include "consider_covariance.h"
MatrixXd readgn(const string& filename){//in the first row there are the dimension of the data
    int n;//designed for general data.
    int m;
    
    ifstream in(filename.c_str());//
    if (!in) {
        cout<<"file wrong!"<<endl;
    }
    in >> n >> m;//first line gives n and m of data
    MatrixXd data(n,m);
    for(int loop=0, x=0, y=0;loop < (m*n) && (in >> data(x,y));)
    {
        ++loop;
        x = loop / m;
        y = loop % m;
    }
    in.close();
    return data;
}
