#include "oct_tree.h"
#include <iostream>
#include <vector>
void Get_all_valid(OCT_TREE::Oct_Node** nodes, OCT_TREE::Oct_Node* root, int valid_leaf_num){
    int valid_num=0;
    int to_search_index=0;
    std::vector<OCT_TREE::Oct_Node*> to_search;
    to_search.clear();
    to_search.push_back(root);
    root=nullptr;
    OCT_TREE::Oct_Node* cur_node=nullptr;
    while(!to_search.empty()){
        cur_node=to_search[to_search.size()-1];
        to_search.pop_back();
        if (cur_node->is_leaf) {
            nodes[valid_num]=cur_node;
            valid_num++;
            cur_node=nullptr;
            continue;
        }
        for (int i=0;i<8;i++){
            if (cur_node->child[i]) to_search.push_back(cur_node->child[i]);
        }
        cur_node=nullptr;
    }
    std::cout<<valid_num<<' '<<valid_leaf_num<<std::endl;
    if (valid_num!=valid_leaf_num) {
        
        throw "not matched node number";
    }
};

int main(){
    float reso,x0,x1,y0,y1,z0,z1;
    OCT_TREE::Point* points= new Point[10];
    for (int i=0;i<10;i++){
        points[i].x= float(i);
        points[i].y=points[i].x;
        points[i].z=points[i].y;
    }
    // reso=-1.0;
    // int b[2]; b[0]=0;b[1]=1;
    // char a=((((char*)&reso)[sizeof(float)-1]>>7) & (0x01));
    // std::cout<<b[a]<<"nimeide"<<std::endl;

    x0=0.0f; y0=0.0f; z0=0.0f;
    x1=9.0f; y1=9.0f; z1=9.0f;
    reso=0.9;

    OCT_TREE::Oct_Tree oct_tree(reso,x0,x1,y0,y1,z0,z1);
    int point_labels[10];
    for (int i=0;i<10;i++){
        point_labels[i]=oct_tree.Creat_Tree(points[i]);
        std::cout<<point_labels[i]<<' '<<points[i].x<<' '<<points[i].y<<' '<<points[i].z<<std::endl;
    }
    int valid_leaf_num=oct_tree.valid_node_num;
    OCT_TREE::Point* centers=new Point[valid_leaf_num];
    OCT_TREE::Oct_Node** valid_nodes=new Oct_Node*[valid_leaf_num];
    OCT_TREE::Oct_Node* cur_node;
    oct_tree.Get_all_valid(valid_nodes,valid_leaf_num);
    for (int i=0;i<valid_leaf_num;i++){
        cur_node=valid_nodes[i];
        centers[i].x=(cur_node->max_range.x+cur_node->min_range.x)/2.0f;
        centers[i].y=(cur_node->max_range.y+cur_node->min_range.y)/2.0f;
        centers[i].z=(cur_node->max_range.z+cur_node->min_range.z)/2.0f;
        std::cout<<centers[i].x<<' '<<centers[i].y<<' '<<centers[i].z<<std::endl;
    }
    cur_node=nullptr;

    OCT_TREE::Point a;
    a.x=1.1;a.y=1.1;a.z=1.1;
    std::cout<<oct_tree.Get_position_label(a)<<std::endl;

    delete []points;
    delete []centers;
    delete []valid_nodes;

    return 0;
}