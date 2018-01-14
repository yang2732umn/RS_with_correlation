#include "consider_covariance.h"
void get_sVectortoData(const string &s,const VectorXd &X,const string& pathAndName){
    ofstream fichier(pathAndName.c_str(),ios_base::app);
    if(fichier.is_open())  // si l'ouverture a réussi
    {
        fichier.precision(10);//set the number of significant digits to be 4 in the file output
        fichier.setf(ios::fixed);
        fichier<<s<<"  "<<X.transpose()<< endl;
        fichier.close();  // on referme le fichier
    }
    else  // sinon
    {
        cerr << "Cannot write in the file!" << endl;
    }
}
