#ifndef OCT_TREE
#define OCT_TREE

#include <iostream>
#include <vector>

namespace OCT_TREE
{

struct Point{
    float x;
    float y;
    float z;
};

struct Oct_Node{
    int point_num;
    bool is_leaf;
    int label;
    Point max_range,min_range;
    Oct_Node* child[8];
};

class Oct_Tree{
    public:
    Oct_Node* Root=new Oct_Node;
    float resolution;
    float x_max,x_min,y_max,y_min,z_max,z_min;
    int valid_node_num;

    Oct_Tree(float reso,float x0, float x1,float y0,float y1,float z0,float z1){
        for (int i=0;i<8;i++) Root->child[i]=nullptr;
        Root->point_num=0;
        Root->is_leaf=false;
        Root->label=-1;
        resolution=reso;
        Root->max_range.x=x1; Root->max_range.y=y1; Root->max_range.z=z1;
        Root->min_range.x=x0; Root->min_range.y=y0; Root->min_range.z=z0;
        this->valid_node_num=0;
        //std::cout<<this->Root->is_leaf<<std::endl;
        //std::cout<<"&Root->point_num   &Root->is_leaf   &Root->label   &(this->resolution)   &(this->valid_node_num)"<<std::endl;
        //std::cout<<&(this->Root->point_num)<<' '<<&(this->Root->is_leaf)<<' '<<&(this->Root->label)<<' '<<&(this->resolution)<<' '<<&(this->valid_node_num)<<std::endl;
    };

    void Create_Node(Oct_Node* father, Point point){
        float dist_x=father->max_range.x-father->min_range.x;
        float dist_y=father->max_range.y-father->min_range.y;
        float dist_z=father->max_range.z-father->min_range.z;
        float diff_x=this->resolution-dist_x;
        float diff_y=this->resolution-dist_y;
        float diff_z=this->resolution-dist_z;
        char divide_x= ((((char*)&diff_x)[sizeof(float)-1]>>7) & (0x01));
        char divide_y= ((((char*)&diff_y)[sizeof(float)-1]>>7) & (0x01));
        char divide_z= ((((char*)&diff_z)[sizeof(float)-1]>>7) & (0x01)); // minus = 1, reso < dist

        if (divide_x|divide_y|divide_z){
            //father->is_leaf=false;
            float center_x=(father->max_range.x+father->min_range.x)/2.0f;
            float center_y=(father->max_range.y+father->min_range.y)/2.0f;
            float center_z=(father->max_range.z+father->min_range.z)/2.0f;
            diff_x=center_x-point.x;
            diff_y=center_y-point.y;
            diff_z=center_z-point.z;
            char child_x=((((char*)&diff_x)[sizeof(float)-1]>>7) & (0x01));
            char child_y=((((char*)&diff_y)[sizeof(float)-1]>>7) & (0x01));
            char child_z=((((char*)&diff_z)[sizeof(float)-1]>>7) & (0x01)); // minus = 1, center < point
            char child_index;
            child_index=((child_x&divide_x)<<2)|((child_y&divide_y)<<1)|(child_z&divide_z);

            Oct_Node* new_child=new Oct_Node;
            new_child->min_range.x=(divide_x&child_x)*dist_x/2.0f+father->min_range.x;
            new_child->min_range.y=(divide_y&child_y)*dist_y/2.0f+father->min_range.y;
            new_child->min_range.z=(divide_z&child_z)*dist_z/2.0f+father->min_range.z;
            new_child->max_range.x=father->max_range.x-(divide_x&(1-child_x))*dist_x/2.0f;
            new_child->max_range.y=father->max_range.y-(divide_y&(1-child_y))*dist_y/2.0f;
            new_child->max_range.z=father->max_range.z-(divide_z&(1-child_z))*dist_z/2.0f;
            new_child->point_num=0;
            new_child->is_leaf=false;
            for (int i=0;i<8;i++) new_child->child[i]=nullptr;
            father->child[child_index]=new_child;
            new_child=nullptr;
            Create_Node(father->child[child_index],point);
            //std::cout<<this->Root->is_leaf<<"dd"<<std::endl;
            father=nullptr;
        }
        else{// can not divide any more
            father->is_leaf=true;
            father->label=this->valid_node_num;
            father->point_num++;
            //std::cout<<dist_x<<' '<<dist_y<<' '<<dist_z<<' '<<this->Root->is_leaf<<std::endl;
            //std::cout<<father->min_range.x<<' '<<father->min_range.y<<' '<<father->min_range.z<<std::endl;
            //std::cout<<father->max_range.x<<' '<<father->max_range.y<<' '<<father->max_range.z<<std::endl;
            father=nullptr;
            return;
        }
    };

    int Creat_Tree(Point point){
        Oct_Node* cur_node= this->Root;
        float to_center_x,to_center_y,to_center_z,
                diff_x,diff_y,diff_z;
        char pos_x,pos_y,pos_z,div_x,div_y,div_z;
        char child_index;
        int label;
        while (!cur_node->is_leaf){
            diff_x=this->resolution-cur_node->max_range.x+cur_node->min_range.x;
            diff_y=this->resolution-cur_node->max_range.y+cur_node->min_range.y;
            diff_z=this->resolution-cur_node->max_range.z+cur_node->min_range.z;
            div_x=((((char*)&diff_x)[sizeof(float)-1]>>7) &(0x01));
            div_y=((((char*)&diff_y)[sizeof(float)-1]>>7) &(0x01));
            div_z=((((char*)&diff_z)[sizeof(float)-1]>>7) &(0x01));
            to_center_x=(cur_node->max_range.x+cur_node->min_range.x)/2.0f-point.x;
            to_center_y=(cur_node->max_range.y+cur_node->min_range.y)/2.0f-point.y;
            to_center_z=(cur_node->max_range.z+cur_node->min_range.z)/2.0f-point.z;
            pos_x=((((char*)&to_center_x)[sizeof(float)-1]>>7) & (0x01));
            pos_y=((((char*)&to_center_y)[sizeof(float)-1]>>7) & (0x01));
            pos_z=((((char*)&to_center_z)[sizeof(float)-1]>>7) & (0x01));
            child_index=((div_x&pos_x)<<2)|((div_y&pos_y)<<1)|(div_z&pos_z);
            if(cur_node->child[child_index]==nullptr) {
                Create_Node(cur_node,point);
                //std::cout<<this->Root->is_leaf<<'a'<<std::endl;
                //std::cout<<&(this->valid_node_num)<<' '<<&(this->Root->is_leaf)<<' '<<&(this->resolution)<<std::endl;
                //cur_node->label=this->valid_node_num;
                //label=cur_node->label;
                label=this->valid_node_num;
                this->valid_node_num++;
                cur_node=nullptr;
                //std::cout<<this->Root->is_leaf<<std::endl;
                return label;
            }
            cur_node=cur_node->child[child_index];
        }
        cur_node->point_num++;
        label=cur_node->label;
        
        //std::cout<<"existed"<<point.x<<' '<<point.y<<' '<<point.z<<' '<<this->Root->is_leaf<<std::endl;
        cur_node=nullptr;
        return label;
    };

    int Get_position_label(Point point){
        Oct_Node* cur_node=this->Root;
        float to_center_x,to_center_y,to_center_z,
                diff_x,diff_y,diff_z;
        char pos_x,pos_y,pos_z,div_x,div_y,div_z;
        char child_index;
        while (!cur_node->is_leaf){
            diff_x=this->resolution-cur_node->max_range.x+cur_node->min_range.x;
            diff_y=this->resolution-cur_node->max_range.y+cur_node->min_range.y;
            diff_z=this->resolution-cur_node->max_range.z+cur_node->min_range.z;
            div_x=((((char*)&diff_x)[sizeof(float)-1]>>7) &(0x01));
            div_y=((((char*)&diff_y)[sizeof(float)-1]>>7) &(0x01));
            div_z=((((char*)&diff_z)[sizeof(float)-1]>>7) &(0x01));
            to_center_x=(cur_node->max_range.x+cur_node->min_range.x)/2.0f-point.x;
            to_center_y=(cur_node->max_range.y+cur_node->min_range.y)/2.0f-point.y;
            to_center_z=(cur_node->max_range.z+cur_node->min_range.z)/2.0f-point.z;
            pos_x=((((char*)&to_center_x)[sizeof(float)-1]>>7) & (0x01));
            pos_y=((((char*)&to_center_y)[sizeof(float)-1]>>7) & (0x01));
            pos_z=((((char*)&to_center_z)[sizeof(float)-1]>>7) & (0x01));
            child_index=((div_x&pos_x)<<2)|((div_y&pos_y)<<1)|(div_z&pos_z);
            if(cur_node->child[child_index]==nullptr) {
                cur_node=nullptr;
                return -1;
            }
            cur_node=cur_node->child[child_index];
        }
        int label=cur_node->label;
        cur_node=nullptr;
        return label;
    };

    void Get_all_valid(OCT_TREE::Oct_Node** nodes, int valid_leaf_num){
        int valid_num=0;
        int to_search_index=0;
        std::vector<OCT_TREE::Oct_Node*> to_search;
        to_search.clear();
        to_search.push_back(this->Root);
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
        if (valid_num!=valid_leaf_num) {
            std::cout<<valid_num<<' '<<valid_leaf_num<<std::endl;
            throw "not matched node number";
        }
    };

    void Delete_Node(Oct_Node* root){
        if (root->is_leaf){
            return;
        }
        else{
            for (int i=0;i<8;i++){
                if (root->child[i]!=nullptr){
                    Delete_Node(root->child[i]);
                    delete root->child[i];
                    root->child[i]=nullptr;
                }
            }
        }
    };

    ~Oct_Tree(){
        Delete_Node(Root);
        delete Root;
        Root=nullptr;
    };

};

};

#endif