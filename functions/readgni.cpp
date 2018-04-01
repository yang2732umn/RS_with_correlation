#include "consider_covariance.h"
MatrixXi readgni(const string& filename){//in the first row there are the dimension of the data
    int n;//designed for general data.
    int m;
    
    ifstream in(filename.c_str());
    
    in >> n >> m;//first line gives n and m of data
    MatrixXi data(n,m);
    for(int loop=0, x=0, y=0;loop < (m*n) && (in >> data(x,y));)
    {
        ++loop;
        x = loop / m;
        y = loop % m;
    }
    in.close();
    return data;
} 
