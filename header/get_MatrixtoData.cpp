#include "consider_covariance.h"
void get_MatrixtoData(const MatrixXd &X,const string& pathAndName){
    ofstream fichier(pathAndName.c_str());
    if(fichier.is_open())  // si l'ouverture a réussi
    {
        fichier.precision(12);//set the number of significant digits to be 4 in the file output
        fichier.setf(ios::fixed);//display using normal notation (as opposed to scientific)
        fichier.setf(ios::showpoint);//Display a decimal and extra zeros, even when not needed.
        fichier<<X.rows()<<" "<<X.cols()<< endl;
        fichier<<X << endl;
        fichier.close();
    }
    else  // sinon
    {
        cerr << "Cannot write in the file!" << endl;
    }
}