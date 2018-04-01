#include "consider_covariance.h"
vector<rated_1itemmore> construct_itemmore(const rated_user_and_item &A){
    vector<rated_user> user=A.user;
    vector<rated_item> item=A.item;
    vector<rated_1itemmore> itemmore(item.size());
#pragma omp parallel for
    for (int i=0; i<item.size(); ++i) {
        rated_1itemmore itemmorec;
        itemmorec.itemno=(item[i]).itemno;
        itemmorec.user=(item[i]).user;
        itemmorec.numberforuser=itemmorec.user;//just define it to give it the right size
        for (int j=0; j<(itemmorec.user).size(); ++j) {
            int userno=(itemmorec.user)[j];
            for (int k=0; k<user[userno].item.size(); ++k) {
                if (user[userno].item[k]==itemmorec.itemno) {
                    itemmorec.numberforuser[j]=k;
                    break;
                }
            }
        }
        itemmore[i]=itemmorec;
    }
    return itemmore;
}
