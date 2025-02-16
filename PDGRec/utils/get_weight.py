import torch
import pickle
import os
import logging
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
data_path = os.path.join(base_dir, "data_exist")
user_num = 60742
game_num = 7726
type_num = 20

def get_wi():
    cnt_path = os.path.join(base_dir, "data_exist/user_type_pct_cnt.npy")
    avg_path = os.path.join(base_dir, "data_exist/user_type_pct_avg.npy")
    user_type_pct_cnt = np.load(cnt_path)
    user_type_pct_avg = np.load(avg_path)

    game_type_inter = []
    path_game_genre_inter = os.path.join(base_dir, "data_exist/game_genre_inter.pkl")
    if os.path.exists(path_game_genre_inter):
        with open(path_game_genre_inter, 'rb') as f:
            game_type_inter = pickle.load(f)
    else: print('game_genre_inter data not exists!')

    # wc
    genre_game_cnt = np.zeros(type_num)
    wc_array = np.zeros(type_num)
    for i in game_type_inter:
        genre_game_cnt[i[1]] += 1
        
    for i in range(type_num):
        wc_array[i] = 1 - genre_game_cnt[i]/sum(genre_game_cnt)
    
    # ws
    ws_array = np.zeros(type_num)
    type_sale_sum = np.sum(user_type_pct_cnt, axis = 0)
    sale_sum = np.sum(type_sale_sum)
    cnt_matrix = np.where(user_type_pct_cnt != 0, 1, user_type_pct_cnt)
    cnt_sum = np.sum(cnt_matrix, axis=0)
    type_sale_avg = np.divide(type_sale_sum, cnt_sum)
    max_sale_avg = max(type_sale_avg)
    for i in range(type_num):
        ws_array[i] = 1 - type_sale_avg[i]/max_sale_avg

    #wm
    wm_array = np.zeros(type_num)
    for i in range(type_num):
        wm_array[i] = 1 - type_sale_sum[i]/sale_sum

    temp = np.multiply(wc_array, ws_array)
    wi_array = np.multiply(temp, wm_array)

    wi_array[wi_array<0.1] = 0.1

    return wi_array

wi_array = get_wi()
wi_path = os.path.join(base_dir, "data_exist/wi_array.npy")
np.save(wi_path, wi_array)
print(f"saved in {wi_path}")

