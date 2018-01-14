#include "consider_covariance.h"
rated_user_and_item construct_user_item(const MatrixXi &data){
    //this function gives the user item observed pairs from the ...*3 data matrix
    //user and item are all ordered
    VectorXi datacol1=data.col(0);
    VectorXi datacol2=data.col(1);
    VectorXi uniqueuser=unique_veci(datacol1);
    VectorXi uniqueitem=unique_veci(datacol2);//unique gives ordered numbers
    vector<rated_user> user;
    vector<rated_item> item;
    int j=0;
    for (int i=0; i<uniqueuser.size(); ++i) {
        rated_user user_crt;
        int k=0;
        user_crt.userno=uniqueuser(i);
        //data should be the first user, then second user, then third....
        while (j<data.rows()&&data(j, 0)==user_crt.userno) {
            (user_crt.item).push_back(data(j, 1));
            k=k+1;
            (user_crt.rating).conservativeResize(k);
            (user_crt.rating)[k-1]=data(j,2);
            ++j;
        }
        user.push_back(user_crt);
    }
    for (int i=0; i<uniqueitem.size(); ++i) {
        j=0;
        rated_item item_crt;
        item_crt.itemno=uniqueitem(i);
        int k=0;
        while (j<data.rows()) {
            if(data(j, 1)==item_crt.itemno){
                (item_crt.user).push_back(data(j, 0));
                k=k+1;
                (item_crt.rating).conservativeResize(k);
                (item_crt.rating)[k-1]=data(j, 2);
            }
            ++j;
        }
        item.push_back(item_crt);
    }
    rated_user_and_item A;
    A.user=user;
    A.item=item;
    return A;
}



//reload of the above function for double matrix
rated_user_and_item construct_user_item(const MatrixXd &data){
    //this function gives the user item observed pairs from the ...*3 data matrix
    //user and item are all ordered
    VectorXi datacol1=data.col(0).cast<int>();
    VectorXi datacol2=data.col(1).cast<int>();
    VectorXi uniqueuser=unique_veci(datacol1);
    VectorXi uniqueitem=unique_veci(datacol2);
    vector<rated_user> user;
    vector<rated_item> item;
    int j=0;
    for (int i=0; i<uniqueuser.size(); ++i) {
        rated_user user_crt;
        int k=0;
        user_crt.userno=uniqueuser(i);
        //data should be the first user, then second user, then third....
        while (j<data.rows()&&data(j, 0)==user_crt.userno) {
            (user_crt.item).push_back(data(j, 1));
            k=k+1;
            (user_crt.rating).conservativeResize(k);
            (user_crt.rating)[k-1]=data(j,2);
            ++j;
        }
        user.push_back(user_crt);
    }
    for (int i=0; i<uniqueitem.size(); ++i) {
        j=0;
        rated_item item_crt;
        item_crt.itemno=uniqueitem(i);
        int k=0;
        while (j<data.rows()) {
            if(data(j, 1)==item_crt.itemno){
                (item_crt.user).push_back(data(j, 0));
                k=k+1;
                (item_crt.rating).conservativeResize(k);
                (item_crt.rating)[k-1]=data(j, 2);
            }
            ++j;
        }
        item.push_back(item_crt);
    }
    rated_user_and_item A;
    A.user=user;
    A.item=item;
    return A;
}


