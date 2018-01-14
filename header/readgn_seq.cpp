#include "consider_covariance.h"
vector<MatrixXd> readgn_seq(const string& filename,int nmat){//read a sequence of nmat matrices from a file
    //in the first row there are the dimension of the data
    int n;//designed for general data.
    int m;
    vector<MatrixXd> matrices;
    ifstream in(filename.c_str());
    if (!in) {
        cout<<"readgn_seq file wrong!"<<endl;
    }
    
    for (int i=0; i<nmat; ++i) {
        in >> n >> m;//first line gives n and m of data
        MatrixXd data(n,m);
        for(int loop=0, x=0, y=0;loop < (m*n) && (in >> data(x,y));)
        {
            ++loop;
            x = loop / m;
            y = loop % m;
        }
        matrices.push_back(data);
    }
    
    in.close();
    return matrices;
}