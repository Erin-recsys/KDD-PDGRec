import pickle
import os
import logging
import numpy as np
import time
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data_exist")

user_num = 60742
game_num = 7726
type_num = 20

cnt_path = os.path.join(base_dir, "data_exist/user_type_pct_cnt.npy")
avg_path = os.path.join(base_dir, "data_exist/user_type_pct_avg.npy")
user_type_pct_cnt = np.load(cnt_path)
user_type_pct_avg = np.load(avg_path)

wi_path = os.path.join(base_dir, "data_exist/wi_array.npy")
wi_array = np.load(wi_path)
# wi_array = np.ones(type_num)

# CI_matrix = np.zeros((100, user_num))
user_type_bool = np.where(user_type_pct_cnt != 0, 1, user_type_pct_cnt)

def cal_social_score(k:int):
    social_score_mat = []
    path_mat = os.path.join(base_dir, "data_exist/social_score_wi_ci_0.75/social_score_20.pkl")
    for i in tqdm(user_num):
        user_a_array = user_type_bool[i]
        ci_score = np.zeros(user_num)
        di_score = np.zeros(user_num)

        for j in range(user_num):
            if j == i:
                continue

            user_b_array = user_type_bool[j]
            # intersection
            intersection = np.logical_and(user_a_array, user_b_array)
            inter_wi = np.multiply(intersection, wi_array)

            # ci
            difference_c = np.logical_and(user_a_array, np.logical_not(user_b_array))     
            diff_wi_c = np.multiply(difference_c, wi_array)

            # di
            difference = np.logical_and(user_b_array, np.logical_not(user_a_array)) #b-a
            diff_wi = np.multiply(difference, wi_array)

            min_vector = np.minimum(user_type_pct_avg[i], user_type_pct_avg[j])

            max_vector = np.maximum(user_type_pct_avg[i], user_type_pct_avg[j])
            
            diff_sum = np.sum(np.multiply(diff_wi, user_type_pct_avg[j]))
            if diff_sum == 0:
                continue
            inter_max = np.sum(np.multiply(inter_wi, max_vector))
            inter_min = np.sum(np.multiply(inter_wi, min_vector))
            # print('diff_sum,inter_max:', diff_sum, inter_max)
            value = diff_sum/(diff_sum + inter_max)
            if value >= 0.2:
                di_score[j] = value

            value = inter_min / (
                inter_max + np.sum(diff_wi_c * user_type_pct_avg[i]))
            if value >= 0.75:
                ci_score[j] = value

        di_score[ci_score==0] = 0
        # print('after:',np.count_nonzero(di_score))
        print('max:',np.max(di_score))

        if np.count_nonzero(di_score) == 0: continue
        if np.count_nonzero(di_score) <= k:
            print('di num:',np.count_nonzero(di_score))
            nonzero_indices = np.nonzero(di_score)
            for tp in nonzero_indices:
                for j in tp:
                    social_score_mat.append([i,j,ci_score[j], di_score[j]])

        else:
            # top_k_values = np.partition(DI_array, -k)[-k:]
            top_k_indices = np.argpartition(di_score, -k)[-k:]
            for j in top_k_indices:
                social_score_mat.append([i,j,ci_score[j], di_score[j]])
    
    # return social_score_mat   
    with open(path_mat, 'wb') as f:
        pickle.dump(social_score_mat, f)
    print(f"file saved {path_mat}")

cal_social_score(k=20)
# print(mat[20:40])