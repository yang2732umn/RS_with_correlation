#include "consider_covariance.h"
void get_vectoData(const VectorXd &X,const string& pathAndName){
    ofstream fichier(pathAndName.c_str());
    if(fichier.is_open())
    {
        fichier.precision(12);//set the number of significant digits to be 12 in the file output
        fichier.setf(ios::fixed);
        fichier.setf(ios::showpoint);
        fichier<<X.size()<<endl;
        fichier<<X << endl;
        fichier.close();
    }
    else
    {
        cerr << "Cannot write in the file!" << endl;
    }
} 