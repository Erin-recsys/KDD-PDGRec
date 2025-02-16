# from dataloader_steam import Dataloader_steam_filtered as dataloader
import torch
import pickle
import os
import logging
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
path = os.path.join(base_dir, "data_exist")
user_num = 60742
game_num = 7726
type_num = 20

def get_ut_preference():
    path_tensor = path+"/tensor_user_game.pth"
    path_dic = path+"/dic_user_game.pkl"
    logging.info('111')

    
    if os.path.exists(path_tensor) and os.path.exists(path_dic):
        tensor_user_game = torch.load(path_tensor)
        with open(path_dic,"rb") as f:
            dic_user_game = pickle.load(f)
        print('tensor_user_game example:',tensor_user_game[:5])
        # return tensor_user_game, dic_user_game
    else: 
        print('user_game data path not exists!')
        
    user_game = tensor_user_game.numpy()

   
    path_game_genre_mapping = os.path.join(base_dir, "data_exist/game_genre_mapping.pkl")
    if os.path.exists(path_game_genre_mapping):
        with open(path_game_genre_mapping, 'rb') as f:
            game_type_mapping = pickle.load(f)
    else: 
        print('game_type data path not exists!')
        

    shape = (user_num, type_num)

    user_type_pct_sum = np.zeros(shape, dtype = float)

    user_type_pct_cnt = np.zeros(shape, dtype = int)
    
    for instance in user_game: #[[user,game,time,percentile]]
        user = instance[0]
        game = int(instance[1])
        percentile = instance[3]

        for tp in game_type_mapping[game]:
            # print(type(user),type(tp))
            user = int(user)
            user_type_pct_cnt[user][tp] += 1
            user_type_pct_sum[user][tp] += percentile

    user_type_pct_avg = np.divide(user_type_pct_sum, user_type_pct_cnt, where=(user_type_pct_cnt != 0))

    cnt_path = os.path.join(base_dir, "data_exist/user_type_pct_cnt.npy")
    avg_path = os.path.join(base_dir, "data_exist/user_type_pct_avg.npy")
    np.save(cnt_path, user_type_pct_cnt)
    print(f"saved in {cnt_path}")
    np.save(avg_path, user_type_pct_avg)
    print(f"saved in {avg_path}")

get_ut_preference()

