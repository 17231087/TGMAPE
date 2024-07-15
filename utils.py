
import torch
import pandas as pd
import math
import random
import numpy as np
import dgl
import pickle as pkl
import datetime
import copy
import math


from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch
import os
from tqdm import tqdm
import pickle
miss_tag_count=0
miss_follow_record = 0
total_follow_len =0
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=5, verbose=False, delta=0):
        """
        Args:
            save_path : the save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
        

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        path = self.save_path
        torch.save(model.state_dict(), path)	# The parameters of the current optimal model will be stored here.
        
        self.val_loss_min = val_loss

class MyDataset:
    def __init__(self, config,data,batch_size,feat_dict) -> None:
        self.batch_size = batch_size
        
        self.feat_dict =feat_dict
        self.step = 0
        self.data = data

        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.device = config.device
    
    def sample_intra_pair(self,cur_index_list,tag_index,video_index):
        #-----for bro_cst_loss 
        #1.get the child tags and videos of each node
        tag01_dict,tag12_dict ={},{}
        for index in cur_index_list:
            tags_list= self.data[index][tag_index]
            video = self.data[index][video_index]
            if tags_list[0]!=0:
                #tag = 0 denotes the tag information is unknown
                if tags_list[0] not in tag01_dict.keys():
                    tag01_dict[tags_list[0]] =[]
                tag01_dict[tags_list[0]].append((tags_list[1],video))
            if tags_list[1]!=0:
                if tags_list[1] not in tag12_dict.keys():
                    tag12_dict[tags_list[1]]= []
                tag12_dict[tags_list[1]].append((tags_list[2],video))
        #2.sample postive and negative pair
        pos_video_batch =[]
        neg_video_batch =[]

        batch_num = len(cur_index_list)
        for index in cur_index_list:
            
            
            tags_list= self.data[index][tag_index]
            video = self.data[index][video_index]
            
            pos_tags_list = tags_list[:]
            pos_video_list =[video,video,video]

            neg_tags_list =[0,0,0] #Default all 0s as negative samples.
            neg_video_list = [0,0,0]
            

            for i in range(1,3):
                
                pre_tag = tags_list[i-1]
                
                if pre_tag ==0 or tags_list[i]==0:  #ignored for pre_tag or tag = 0.
                    break

                inter_tag_dict = tag01_dict if i==1 else tag12_dict
                #Negative sampling non-sibling node
                pre_tag_list = list(inter_tag_dict.keys())
                sample_count = 0
                while sample_count<batch_num:
                    neg_pre_tag = random.choice(pre_tag_list)
                    if neg_pre_tag!=pre_tag:  #non-sibling node
                        tv = random.choice(inter_tag_dict[neg_pre_tag])
                        neg_tags_list[i] = tv[0]
                        neg_video_list[i] =tv[1]
                        if neg_tags_list[i]!=0:
                            break
                    sample_count+=1

                
                #postive sampling sibling node
                sample_list = inter_tag_dict[pre_tag]

                num =len(sample_list) 
                sample_count = 0
                while sample_count<num:
                    t_v = random.choice(sample_list) #Choose a child tag at random
                    if t_v[0]!=tags_list[i]: #Choose a tag that is not the current  tag.
                        pos_tags_list[i]= t_v[0]
                        pos_video_list[i]=t_v[1]
                        
                        if t_v[0]!=0:  #Try not to choose 0.
                            
                            break
                    sample_count+=1
                
            
            
            pos_video_batch.append(pos_video_list[:])
            neg_video_batch.append(neg_video_list[:])
            

        return pos_video_batch,neg_video_batch
    def sample_inter_pair(self,cur_index_list,tag_index,video_index):
        neg_tags_batch = []
        neg_video_batch = []
        batch_num = len(cur_index_list)
        for index in cur_index_list:
            
            
            tags_list= self.data[index][tag_index]
            video = self.data[index][video_index]
            neg_tags_list =[0,0,0] #Default all 0s as negative samples.
            neg_video_list = [0,0,0]
            for i in range(1,3):
                sample_count =0
                while sample_count<batch_num:
                    neg_i= random.choice(cur_index_list)
                    sample_tags_list = self.data[neg_i][tag_index]
                    
                    if sample_tags_list[i-1]!=tags_list[i-1]: # The parent node of negative sample is different from this sample
                        neg_video_list[i-1] = self.data[neg_i][video_index] 
                        break
                    sample_count += 1
        
            
            neg_video_batch.append(neg_video_list[:])
        
        return neg_video_batch


    def next_batch(self,tag_cst_loss = False):
        
        
        if self.step == self.total_step:
            self.step = 0
            np.random.seed(0)
            np.random.shuffle(self.index_list)
        start = self.step * self.batch_size
        end = min(start + self.batch_size, self.sample_num)
        self.step += 1
        data_dict = {f: torch.tensor([self.data[i][j] for i in self.index_list[start:end]], dtype=torch.int64).to(self.device) for f,j in self.feat_dict.items()}

        #batch_num = end-start

        tag_index = self.feat_dict['tags']
        video_index = self.feat_dict['video']
        cur_index_list = self.index_list[start:end]
        
        pos_bro_video,neg_bro_video=  self.sample_intra_pair(cur_index_list,tag_index,video_index)
        
        neg_par_video = self.sample_inter_pair(cur_index_list,tag_index,video_index)
        data_dict['pos_bro_video'] = torch.tensor(pos_bro_video, dtype=torch.int64).to(self.device)
        data_dict['neg_bro_video']= torch.tensor(neg_bro_video, dtype=torch.int64).to(self.device)
        data_dict['neg_par_video']=torch.tensor(neg_par_video,dtype = torch.int64).to(self.device)
  
        return data_dict
    
        
                
def test(config,model,test_data,feat_dict,early_stopping =None):
    device = config.device
    test_batch_size = config.test_batch_size
    
    
    domain_num  = config.domain_num
    model.eval()

    domain_x_index = feat_dict['domain_id']
    
    
    y_train_true = []
    y_train_predict =[]
    domains_true =[[],[],[]]
    domains_pred = [[],[],[]]
    
    test_sample = 0
    
    
    count =0
    test_loss_sum_dict={}
    
    
    with torch.no_grad ():
        
        while True:
            
            data_dict = test_data.next_batch()
            
            
            sample_num = len(data_dict['user'])
            
                
            losses,pred= model(data_dict,train=False)
            loss = sum(losses.values())
            for k,v in losses.items():
                test_loss_sum_dict[k]= test_loss_sum_dict.get(k,0)+v.item()*sample_num


            batch_label = data_dict['long_view']#equal to the effective view label
            y_train_true += batch_label.tolist()
            y_train_predict += pred.cpu().tolist()
            domain_ids = data_dict['domain_id']
            
            for d in range(domain_num):
                index = (domain_ids == d).nonzero(as_tuple=True)[0]
                domains_true[d]+=batch_label[index].tolist()
                domains_pred[d]+=pred[index].tolist()
                

            
            test_sample+=sample_num
            if test_data.step>=test_data.total_step:
                break

        long_view_auc = roc_auc_score(y_train_true, y_train_predict)

        

        print("{} the sample number:{}".format(now_time(),test_sample),end=",")
        for k,v in test_loss_sum_dict.items():
            print("{}:{:.4f}".format(k,v/test_sample),end = ",")
        print("auc:{:.4f}".format(long_view_auc))#,end=" ")
        
        
        for d in range(domain_num):
            print("{}:{:.4f}".format(config.domain_map[d],roc_auc_score(domains_true[d], domains_pred[d])),end="; ")
                
        print()
        
            
        
        if  early_stopping:
            
            early_stopping(-long_view_auc, model)
            #Early_stop is set to True when the early stop condition is reached.
            if early_stopping.early_stop:
                print("Early stopping")
                return True
        return False

        
def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '




def load_uv_graph(dataset,domain_num):
    train_graph_tuple =[]
    total_num = len(dataset)
    for i in range(total_num):
        user,photo,label = dataset['user_id'][i],dataset['photo_id'][i],dataset['long_view'][i]
        
        if label==0:  #only the positive interaction to construct the graph
            continue
        
        
        train_graph_tuple.append([user,photo])

    return train_graph_tuple
    

        









    
def load_data(config,list_graph=False):
    
    def get_follows(user_id):
        global total_follow_len
        
        if user_id not in user2pub_dict.keys():
            return [0]*10 #0 for padding or denotes the following information is unknown
        else:
            follows = user2pub_dict[user_id]
            total_follow_len +=len(follows)
            follows = follows[:10]+[0]*(10-len(follows)) #tag = 0 denotes the tag information is unknown
            return follows
    def get_tags(photo_id):        
        return item2tag_dict.get(photo_id,[0]*3)
    read_column = ['user_id', 'photo_id', 'domain_id','long_view', 'plvtr']
  
    nrows = 300 if config.small_test else None
    train_data = pd.read_csv(config.train_data_path,usecols=read_column,nrows=nrows)
    valid_data = pd.read_csv(config.valid_data_path,usecols=read_column,nrows=nrows)
    test_data = pd.read_csv(config.test_data_path,usecols=read_column,nrows=nrows)
    
    #Load other information: user-follow-publisher, video-correspond-tag, publisher-release-tag.
    user2pub_dict  = pickle.load(open(config.user2pub_path,'rb'))
    item2tag_dict = pickle.load(open(config.item2tag_path,'rb'))
    # frequently releases videos for each publisher
    pub2tag_dict = pickle.load(open(config.pub2tag_path,'rb'))
    #1.publishers: user u following publisher list
    train_data['publishers'] = train_data['user_id'].apply(get_follows)
    valid_data['publishers'] = valid_data['user_id'].apply(get_follows)
    test_data['publishers'] = test_data['user_id'].apply(get_follows)
    

    #2.tag: video v corresponding to tag list
    train_data['tags'] = train_data['photo_id'].apply(get_tags)
    valid_data['tags'] = valid_data['photo_id'].apply(get_tags)
    test_data['tags'] = test_data['photo_id'].apply(get_tags)

    
        
    feat_dict={}
    
    
    i=0
    new_name_dict = {'user_id':'user','photo_id':'video'}
    for index,f in enumerate(train_data.columns):
        if f in new_name_dict.keys():
            f = new_name_dict[f]
        feat_dict[f]=index
        


    #---------construct graph----------
    train_graph={}
    # user-interact-video and user-follow publisher relations are shared among sub-graphs of different domains
    train_graph['uv'] = load_uv_graph(train_data,domain_num=config.domain_num)  
    follow_graph = [(u,f)  for u,fs in user2pub_dict.items() for f in fs] 
    train_graph['up']=follow_graph
    
        
    for i in range(config.tag_level_num):
        #tag = 0 denotes the tag information is unknown, so we don't add the tag 0 in the graph
        train_graph['tp{}'.format(i)] =[(int(t[i]),f) for f,t in pub2tag_dict.items() if int(t[i])!=0] 
    for i in range(config.tag_level_num):
        train_graph['tv{}'.format(i)] = [(int(t[i]),v)  for v,t in item2tag_dict.items() if int(t[i])!=0]
                
    

    return train_data.iloc[:, :].values, valid_data.iloc[:, :].values,test_data.iloc[:, :].values,feat_dict,train_graph







def build_multi_tag_graph_heter(config,train_graph_tuple,device=None):
    '''
    construct graph
    '''
    
    graph_dict ={}
    uv_u,uv_v = np.array(train_graph_tuple['uv']).transpose()
    num_nodes_dict={
        'user':config.count_dict['user_id'],
        'video':config.count_dict['photo_id']
        
    }
    
    graph_data = {
        ('user', 'uv', 'video'): (uv_u,uv_v),
        ('video', 'vu', 'user'): (uv_v,uv_u)
    }
    
        
    up_u,up_f = np.array(train_graph_tuple['up']).transpose()

    num_nodes_dict['publisher']=config.count_dict['publishers']
    graph_data[('publisher','pu','user')] = (up_f,up_u)
    graph_data[('user','up','publisher')] = (up_u,up_f)
    

    for t in range(config.tag_level_num):
        
        tv_t,tv_v = np.array(train_graph_tuple['tv{}'.format(t)]).transpose()
        num_nodes_dict['tag'] = config.count_dict['tag{}'.format(t)]
        
        graph_data[('tag','tv','video')]=(tv_t,tv_v)
        graph_data[('video','vt','tag')]=(tv_v,tv_t)
        
        
        tp_t,tp_f = np.array(train_graph_tuple['tp{}'.format(t)]).transpose()
        num_nodes_dict['tag'] = config.count_dict['tag{}'.format(t)]
        num_nodes_dict['publisher']=config.count_dict['publishers']
        graph_data[('tag','tp','publisher')]=(tp_t,tp_f)
        graph_data[('publisher','pt','tag')] = (tp_f,tp_t)
        print("num_nodes_dict",num_nodes_dict)
        g = dgl.heterograph(graph_data,num_nodes_dict=num_nodes_dict)
        if device:
            g=g.to(device)
        graph_dict[t]=g
    #print("graph_dict",graph_dict)
    return graph_dict




    