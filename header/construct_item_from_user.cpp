#include "consider_covariance.h"
vector<rated_item> construct_item_from_user(const vector<rated_user> &user){
    vector<rated_item> item;
    set<size_t> s;
    for( size_t i = 0; i < user.size(); ++i){
        for (size_t j=0; j<user[i].item.size(); ++j) {
            s.insert(user[i].item[j]);
        }
    }//s contains sorted itemno
    
    
    set<size_t>::iterator it=s.begin();
    
    for(size_t t=0;t<s.size();++t){
        rated_item this_item;
        this_item.itemno=*it;
        int k=0;
        for( size_t i = 0; i < user.size(); ++i){
            for (size_t j=0; j<user[i].item.size(); ++j) {
                if(user[i].item[j]==*it){
                    this_item.user.push_back(user[i].userno);
                    k=k+1;
                    (this_item.rating).conservativeResize(k);
                    (this_item.rating)[k-1]=user[i].rating[j];
                    break;
                }
            }
        }
        item.push_back(this_item);
        ++it;
    }
    return item;
}
