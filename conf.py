import argparse
import inspect
import torch



class Config:
    publisher_attn_method = 'dot'
    
    small_test = False
    epochs = 20
    random_seed=0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device ==torch.device('cpu'):
        print("you use cpu now!!!")
    batch_size = 8196
    test_batch_size= 20000 #speed up the inference speed

    #hy-parameter
    intra_cst_lambda = 0.6
    inter_cst_lambda =0.4
    lr=1e-4
    layer_num=2
    emb_dim=32

    #dataset information
    base_dir ="../"
    save_model_dir = base_dir+"save_model/"
    data_name = "data/MDSVR-small/"   
    data_dir = base_dir+data_name
    tag_level_num = 3
    count_dict = {'domain_id':3,'user_id': 116181, 'photo_id': 304532, 'publishers': 5435, 'tag0': 39, 'tag1': 176, 'tag2': 414}
    domain_num=3
    domain_map ={0:'Featured-Video',1:'Double-Columned Discovery',2:'Single-Columned Swift Slide'}
    train_data_path =data_dir+'train_data.csv'
    valid_data_path = data_dir+'valid_data.csv'
    test_data_path = data_dir+'test_data.csv'
    user2pub_path = data_dir+"user_follow_dict.pkl"
    item2tag_path = data_dir+"item_tag_dict.pkl"
    pub2tag_path = data_dir+"follow_tag_dict_all.pkl"

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str