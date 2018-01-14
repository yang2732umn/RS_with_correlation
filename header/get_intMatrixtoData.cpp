#include "consider_covariance.h"
void get_intMatrixtoData(const MatrixXi &X,const string& pathAndName){
    ofstream fichier(pathAndName.c_str());
    if(fichier.is_open())
    {
        fichier.precision(10);//set the number of significant digits to be 4 in the file output
        fichier.setf(ios::fixed);
        fichier.setf(ios::showpoint);
        fichier<<X.rows()<<" "<<X.cols()<< endl;
        fichier<<X << endl;
        fichier.close();
    }
    else  // sinon
    {
        cerr << "Cannot write in the file!" << endl;
    }
}  
