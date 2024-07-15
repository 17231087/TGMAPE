import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size,q_size=None):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','mlp']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            
            self.linear = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            q_size = 2*hidden_size
            initscale=0.05
            self.linear = torch.nn.Linear(self.hidden_size+q_size, hidden_size)
            self.v = nn.Parameter(nn.init.uniform_(torch.FloatTensor(hidden_size),a=-initscale,b=initscale))
        elif self.method == 'mlp':
            #print("aaaa")
            q_size = 2*hidden_size
            input_size = q_size+hidden_size
            self.mlp= nn.Sequential(
                nn.Linear(input_size,input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2,input_size//4),
                nn.LeakyReLU(),
                nn.Linear(input_size//4,1),
                
            )
            

    def dot_score(self, hidden, encoder_output):
        #print("aaaa")
        return torch.sum(hidden * encoder_output, dim=2)  #[N,dim]*[T,N,dim]

    def general_score(self, hidden, encoder_output):
        energy = self.linear(encoder_output) #[T,N,dim]
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.linear(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        
        return torch.sum(self.v * energy, dim=2)  #[T,N]
    def mlp_score(self,hidden,encoder_output):
        # hidden: [N,q_size]
        # encoder_output:[T,N,dim]
        T = encoder_output.shape[0]
        dim = encoder_output.shape[-1]
        
        hidden = hidden.repeat([T,1])  #[T*N,q_size]
        encoder_output = encoder_output.reshape(-1,dim)  #[T*N,dim]
        
        x=torch.cat([hidden,encoder_output],dim=1)
        
        
        return self.mlp(x).reshape(T,-1)  #T,N
        


    def forward(self, hidden, ori_encoder_outputs,mask=None):
        # 根据给定的方法计算注意力权重（能量）
        '''
        输入为： hidden: [N,dim]
        encoder_output: [N,T,dim]
        mask: [N,T]
        输出:N,T
        '''
        encoder_outputs = ori_encoder_outputs.transpose(0,1)  #[T,N,dim]
        #print(encoder_outputs.shape)
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)#[T,N]
        elif self.method == 'mlp':
            attn_energies = self.mlp_score(hidden, encoder_outputs)#[T,N]
        #return attn_energies
        
        
        attn_energies = attn_energies.t()  #[N,T]
        
        if mask!=None:
            A=F.softmax(attn_energies, dim=1) #N,T
            #print("ttttt",A[:10])
            A = A*mask #N,T
            A_sum=torch.sum(A, dim=1) #N
            threshold=torch.ones_like(A_sum)*1e-5 
            A_sum = torch.max(A_sum, threshold).unsqueeze(1) #[N,1]
            
            A = A / A_sum #[N,T]
            attn_energies =A.unsqueeze(1) #[N,1,T]

        # 返回softmax归一化概率分数（增加维度）
        else:
            attn_energies=F.softmax(attn_energies, dim=1).unsqueeze(1)  #[N,1,T]
        
        context = attn_energies.bmm(ori_encoder_outputs) #[N,1,T]*[N,T,dim]
        
        return context.squeeze(1),attn_energies.squeeze(1)

class IntraAggr(nn.Module): # Intra-Level Graph Aggregation
    def __init__(self,config):
        super(IntraAggr, self).__init__()
        self.graphlayers = nn.ModuleList()
        
        
        emb_dim = config.emb_dim
        
        self.layer_num = config.layer_num
        self.graphlayer = SingleLayerAggr(config) 
        
    
    def forward(self,train_graph,user_emb,video_emb,publisher_emb,tag_emb):
        

        prev_user_embedding,prev_video_embedding = user_emb, video_emb
        prev_publisher_emb,prev_tag_emb = publisher_emb,tag_emb


        #2. aggregate
        
        for i in range(self.layer_num):
            
            user_emb, video_emb,publisher_emb,tag_emb = self.graphlayer(train_graph,user_emb,video_emb,publisher_emb,tag_emb)  
            prev_user_embedding = prev_user_embedding+ user_emb*(1/(i+2))
            prev_video_embedding = prev_video_embedding+ video_emb*(1/(i+2))
            prev_publisher_emb = prev_publisher_emb + publisher_emb*(1/(i+2))
            prev_tag_emb = prev_tag_emb + tag_emb*(1/(i+2))
        
        return prev_user_embedding,prev_video_embedding,prev_publisher_emb,prev_tag_emb
class SingleLayerAggr(nn.Module):
    def __init__(self,config):
        super(SingleLayerAggr, self).__init__()
        self.in_dict = {
            'user':['vu','pu'],
            'video':['uv','tv'],
            'publisher':['up','tp'],
            'tag':['vt','pt']
        }
        
        self.emb_dim = config.emb_dim
        self.half_emb_dim = config.emb_dim//2
       

    
    

    def forward(self, graph, user_emb,video_emb,publisher_emb=None,tag_emb=None): 
        with graph.local_scope():
            edge_dic = {
                    'uv':user_emb[:,:self.half_emb_dim],
                    'up':user_emb[:,self.half_emb_dim:],
                    'vu':video_emb[:,:self.half_emb_dim],
                    'vt':video_emb[:,self.half_emb_dim:],
                    'pu':publisher_emb[:,:self.half_emb_dim],
                    'pt':publisher_emb[:,self.half_emb_dim:],
                    'tv':tag_emb[:,:self.half_emb_dim],
                    'tp':tag_emb[:,self.half_emb_dim:]
            }

            out_edge_dic = {}
                    
            aggr_dict= {}
            
            for etype in graph.canonical_etypes:
                #print(etype)
                src, e, dst = etype
                
                graph.nodes[src].data['{}_h'.format(e)] = edge_dic[e]
                
                aggr_dict[e] = (fn.copy_u('{}_h'.format(e),'m'),fn.mean('m','{}_h'.format(e)))

            graph.multi_update_all( 
                aggr_dict,
                'mean'
            )
            
            
            emb_dic= {}
            for dst in self.in_dict.keys():
                
                emb_dic[dst]=torch.concat([graph.nodes[dst].data['{}_h'.format(e)] for e in self.in_dict[dst]],dim=1)
            return emb_dic['user'],emb_dic['video'],emb_dic['publisher'],emb_dic['tag'] 


 

