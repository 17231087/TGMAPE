
from utils import * 
from models.TGMAPE import TGMAPE
import torch
from conf import Config
import numpy as np
import random,os
from sklearn.metrics import roc_auc_score


def main(config):
    
    print(now_time()+" start loading data")
    train_dataset,valid_dataset,test_dataset,feat_dict,train_graph_triplets = load_data(config)
    print(now_time()+" start building global graph")
    global_graph = build_multi_tag_graph_heter(config,train_graph_triplets, config.device)
    
    
    
    model = TGMAPE(config).to(config.device)
    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)
    model_path = config.save_model_dir +"main"+datetime.datetime.now().strftime('%M:%S')+".pt"#
    print("-------------Save path----------",model_path)
    
    train_dataset = MyDataset(config,train_dataset,config.batch_size,feat_dict)
    valid_dataset = MyDataset(config,valid_dataset,config.test_batch_size,feat_dict)
    test_dataset = MyDataset(config,test_dataset,config.test_batch_size,feat_dict)

    
    early_stopping = EarlyStopping(model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    
    print(now_time()+" start training")    
    for epoch in range(1,config.epochs+1):
        model.train()
        
        y_train_true = []
        y_train_predict = []
        
        train_sample=0
        train_loss_sum_dict ={}
        
        while True:
            data_dict = train_dataset.next_batch()
            sample_num = len(data_dict['user'])
            losses,pred= model(data_dict,global_graph)
            
            loss = sum(losses.values())

            batch_label = data_dict['long_view']
            
            

            y_train_true += batch_label.tolist()
            y_train_predict += pred.tolist()
            
            optimizer.zero_grad()
            loss.backward()
            for k,v in losses.items():
                train_loss_sum_dict[k]= train_loss_sum_dict.get(k,0)+v.item()*sample_num
            
            train_sample+=sample_num
            optimizer.step()
            if train_dataset.step>=train_dataset.total_step:
                break
        
        long_view_auc = roc_auc_score(y_train_true, y_train_predict)
        
        print("---------------------epoch:{}------------------------".format(epoch))
        
        print("{},训练样本:{}".format(now_time(),train_sample),end=",")
        for k,v in train_loss_sum_dict.items():
            print("{}:{:.4f}".format(k,v/train_sample),end=",")
        print("auc:{:.4f}".format(long_view_auc))#,end=" ")
        
        print("验证",end=" ")
            
        model.graph_aggr(global_graph)
        flag=test(config,model,valid_dataset,feat_dict,early_stopping=early_stopping)
        
        
    
        print("测试",end=" ")
        test(config,model,test_dataset,feat_dict)

        
        if flag:
            break
        
        
            
    
    pre_state_dict=torch.load(model_path)
    model.load_state_dict(pre_state_dict)
    model.graph_aggr(global_graph)
    print("---------------------best!!!")
    
    test(config,model,test_dataset,feat_dict)
    os.remove(model_path)


if __name__=='__main__':
    config= Config()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("------------the parameter setting-------------")
    print(config)
    main(config)
    


        