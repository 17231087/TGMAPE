import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from models.graph_models import *
import pandas as pd

    
class TGMAPE(nn.Module):
    def __init__(self,config) -> None:
        super(TGMAPE, self).__init__()
        self.config = config
        self.device =  config.device
        
        emb_dim = config.emb_dim
        
        
        initializer = nn.init.xavier_uniform_

        for i in range(config.tag_level_num):
            setattr(self, 'user_embedding_{}'.format(i), nn.Parameter(initializer(torch.empty(config.count_dict['user_id'], emb_dim))))
            setattr(self, 'video_embedding_{}'.format(i), nn.Parameter(initializer(torch.empty(config.count_dict['photo_id'], emb_dim))))
            setattr(self, 'publisher_embedding_{}'.format(i), nn.Parameter(initializer(torch.empty(config.count_dict['publishers'], emb_dim))))
            setattr(self, 'tag_embedding_{}'.format(i), nn.Parameter(initializer(torch.empty(config.count_dict['tag{}'.format(i)], emb_dim))))
        
        self.domain_embedding = nn.Parameter(initializer(torch.empty(config.count_dict['domain_id'], emb_dim)))
        self.graph_aggr_model = IntraAggr(config).to(config.device)
        
        
        self.key_attn_linear = nn.Linear(2*emb_dim, emb_dim)

        self.video_aware_pub_attn = Attn(config.publisher_attn_method,emb_dim)
        
        
        self.chd_mlp = nn.Sequential(
                nn.Linear((2)*emb_dim,int(emb_dim)),
                nn.Dropout(p=0.2),
                nn.LeakyReLU(),
                nn.Linear(int(emb_dim),1),
                
            )
        #----- for inter-level and intra-level constrast learning
        self.intra_cst_lambda = config.intra_cst_lambda 
        self.inter_cst_lambda = config.inter_cst_lambda
        self.inter_is_par_mlp = nn.Sequential(
            nn.Linear((2)*emb_dim,int(emb_dim)),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(int(emb_dim),1)
            )
        
        self.intra_is_bro_mlp = nn.Sequential(
            nn.Linear((2)*emb_dim,int(emb_dim)),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(int(emb_dim),1),
           
        )
        
        
        
        #---- for prediction
        task_dnn = nn.ModuleList()
        hid_dim = [4*emb_dim,2*emb_dim,emb_dim]

        for j in range(len(hid_dim) - 1):
            task_dnn.add_module('ctr_hidden_{}'.format(j),nn.Linear(hid_dim[j], hid_dim[j + 1]))
            task_dnn.add_module('ctr_leakrelu_{}'.format(j),nn.LeakyReLU())                                                   
        task_dnn.add_module('task_last_layer',nn.Linear(hid_dim[-1], 1))
        self.task_dnn = task_dnn
        
        
        self.loss_func = nn.BCELoss() #WithLogits


    def graph_aggr(self,train_graph):


        
        self.graph_list = {'user':[],'video':[],'publishers':[],'tags':[]}
        for t in range(self.config.tag_level_num):
                
                
                
            user_emb,video_emb,publisher_emb,tag_emb = self.graph_aggr_model(train_graph[t],\
            getattr(self, 'user_embedding_{}'.format(t)),getattr(self, 'video_embedding_{}'.format(t)),getattr(self, 'publisher_embedding_{}'.format(t)),getattr(self, 'tag_embedding_{}'.format(t)))
            
            self.graph_list['user'].append(user_emb)
            self.graph_list['video'].append(video_emb)
            self.graph_list['publishers'].append(publisher_emb)
            self.graph_list['tags'].append(tag_emb)
    
            
            
                
                
        

    def get_inter_cst_loss(self,data_dict):
        video_emb_list = []
        neg_video_emb_list = []

        vi = data_dict['video']
        neg_par_vis = data_dict['neg_par_video']
        

        pos_par_video = None
        neg_par_video = None
        
        intra_cst_loss = 0
        batch_size = vi.shape[0]
        for i in range(self.config.tag_level_num):
            video_embedding = self.graph_list['video'][i]
            
            neg_par_vi = neg_par_vis[:,i]
            pos_video = video_embedding[vi]
            
            if pos_par_video!=None:
                pos_par_score = self.inter_is_par_mlp(torch.concat([pos_par_video,pos_video],dim=1))
                neg_par_score = self.inter_is_par_mlp(torch.concat([neg_par_video,pos_video],dim=1))

                score_gap = (pos_par_score - neg_par_score)
                intra_cst_loss += -self.inter_cst_lambda*torch.mean(F.logsigmoid(score_gap))
            pos_par_video = pos_video
            neg_par_video =video_embedding[neg_par_vi]
        return intra_cst_loss/(self.config.tag_level_num-1)
    def get_intra_cst_loss(self,data_dict):
        
        bro_loss = 0
        vi = data_dict['video']
        neg_vi= data_dict['pos_bro_video']#'neg_video']
        pos_vi = data_dict['neg_bro_video']

        for i in range(1,self.config.tag_level_num):
            
            video = vi
            neg_video = neg_vi[:,i]
            pos_video = pos_vi[:,i]
            #pos_video and video has the same parent tag node
            video_embedding = self.graph_list['video'][i]#[video]
            video_emb = video_embedding[video]
            pos_video_emb = video_embedding[pos_video]
            pos_score = self.intra_is_bro_mlp(torch.cat([video_emb,pos_video_emb],dim=1))
            neg_video_emb = video_embedding[neg_video]
            neg_score = self.intra_is_bro_mlp(torch.cat([video_emb,neg_video_emb],dim=1))
            bro_loss += -self.intra_cst_lambda*torch.mean(F.logsigmoid(pos_score-neg_score))
        
        
            
        return bro_loss/(self.config.tag_level_num-1)
    
    def get_hiretical_emb(self,data_dict,domain_emb):
        input_list = []
        fea_name_list = ['user', 'video','publishers']
        

        for fea_name in fea_name_list:
                
            feat = data_dict[fea_name] #[N,1]
            feat_embs = []
            for i in range(self.config.tag_level_num): 
                

                feat_emb = self.graph_list[fea_name][i][feat]

                
                if fea_name =='publishers': #Video-aware publisher representation
                    
                    mask = torch.gt(feat, 0).int()
                    tag_emb = self.graph_list['tags'][i][data_dict['tags'][:,i]]
                    video_emb = self.graph_list['video'][i][data_dict['video']]
                    
                    key_emb=self.key_attn_linear(torch.cat([video_emb,tag_emb],dim=1))#,user_emb],dim=1)) #torch.cat([video_emb,tag_emb],dim=1)#torch.cat([video_emb,tag_emb,domain_emb],dim=1)#
                    
                    
                    feat_emb,attn = self.video_aware_pub_attn(key_emb,feat_emb,mask)
                    

                feat_embs.append(feat_emb)
            feat_emb = torch.stack(feat_embs,1).mean(dim = 1) # Inter-Level Fusion: mean
            input_list.append(feat_emb)
        return input_list
    def forward(self,data_dict,global_graph=None,train=True):

        if train:
            self.graph_aggr(global_graph)

        domain_id = data_dict['domain_id']
        
        domain_emb = self.domain_embedding[domain_id]
        emb_dict={}

        input_list = self.get_hiretical_emb(data_dict,domain_emb)
        
        '''inter_level and intra_level cst_loss'''        
        
        inter_cst_loss = self.get_inter_cst_loss(data_dict)  
        intra_cst_loss = self.get_intra_cst_loss(data_dict)

        
        

        
        input_list.append(domain_emb)
        x = torch.cat(input_list,dim=1)

        for mod in self.task_dnn:
            
            x = mod(x)
        
        task_outputs = torch.sigmoid(x.squeeze(1))
        
        
        labels = data_dict['long_view']
        loss = self.loss_func(task_outputs,labels.float())
        
        return {"bpr_loss":loss,"inter_cst_loss":inter_cst_loss,"intra_cst_loss":intra_cst_loss}, task_outputs
     