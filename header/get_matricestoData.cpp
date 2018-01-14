#include "consider_covariance.h"
void get_matricestoData(const vector<MatrixXd> &matrices,const string& pathAndName){
    ofstream fichier(pathAndName.c_str());
    if(fichier.is_open())  // si l'ouverture a réussi
    {
        fichier.precision(6);//set the number of significant digits to be 6 in the file output, previously was 10, reduced to 6 to save some space
        fichier.setf(ios::fixed);
        fichier.setf(ios::showpoint);
        for (int i=0; i<matrices.size(); ++i) {
            MatrixXd X=matrices[i];
            fichier<<X.rows()<<" "<<X.cols()<< endl;
            fichier<<X << endl;
        }
        fichier.close();  // on referme le fichier
    }
    else  // sinon
    {
        cerr << "Cannot write in the file!" << endl;
    }
}