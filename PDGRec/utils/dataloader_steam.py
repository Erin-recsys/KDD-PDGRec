import os
import sys
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats
import pdb
import torch
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
import pickle

import dgl.function as fn
import pandas as pd

game_num = 7726

class Dataloader_steam_filtered(DGLDataset):
    def __init__(self, args, path, user_id_path, app_id_path,  genre_path, device = 'cpu', name = 'steam'):
        
        logging.info("steam dataloader init...")

        self.args = args
        self.path = path
        self.user_id_path = self.path+"/users.txt"


        self.app_id_path = self.path+"/app_id.txt"

        self.genre_path = self.path+"/Games_Genres.txt"


        self.train_game_path = self.path+"/train_game.txt"
        self.valid_game_path = self.path+"/valid_data/valid_game.txt"
        self.test_game_path = self.path+"/test_data/test_game.txt"
        self.train_time_path = self.path+"/train_time.txt"
        self.device=device
        self.graph_path = self.path + "/graph.bin"
        
        '''get user id mapping and app id mapping'''
        logging.info("reading user id mapping:")
        self.user_id_mapping = self.read_user_id_mapping(self.user_id_path)
        logging.info("reading app id mapping:")
        self.app_id_mapping = self.read_app_id_mapping(self.app_id_path)
        
        


        '''build valid and test data'''

        logging.info("build valid data:")
        self.valid_data = self.build_valid_data(self.valid_game_path)
        logging.info("build test data:")
        self.test_data = self.build_test_data(self.test_game_path)

        self.process()
        dgl.save_graphs(self.graph_path, self.graph)

    def generate_percentile(self, ls):
        dic = {}
        for ls_i in ls:  
            if ls_i[1] in dic:
                dic[ls_i[1]].append(ls_i[2])  
            else:
                dic[ls_i[1]] = [ls_i[2]]
        

        for key in tqdm(dic):
            dic[key] = sorted([time for time in dic[key] if time is not None and time != -1])
        

        dic_percentile = {}
        for key in tqdm(dic):
            dic_percentile[key] = {}
            length = len(dic[key])  
            for i in range(length):
                time = dic[key][i]
                dic_percentile[key][time] = (i + 1) / length  
        user_percentiles = {}
        for ls_i in ls:
            user, game, time = ls_i[0], ls_i[1], ls_i[2]
            if time is not None and time != -1:  
                if user not in user_percentiles:
                    user_percentiles[user] = []
                user_percentiles[user].append(dic_percentile[game][time])
        
        
        user_mean_percentile = {
            user: np.mean(percentiles) if percentiles else None
            for user, percentiles in user_percentiles.items()
        }
        
        
        for i in tqdm(range(len(ls))):
            user, game, time = ls[i][0], ls[i][1], ls[i][2]
            if time is not None and time != -1:
                ls[i].append(dic_percentile[game][time])  
            else:
                
                ls[i].append(user_mean_percentile[user] if user_mean_percentile[user] is not None else 0)
        
        return ls  

    def read_user_id_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_user_id_mapping = os.path.join(base_dir, "data_exist/user_id_mapping.pkl")
        if os.path.exists(path_user_id_mapping):
            with open(path_user_id_mapping, 'rb') as f:
                mapping = pickle.load(f)

        else:
            count = int(0)
            with open(path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line not in mapping.keys():
                        mapping[line] = int(count)
                        count += 1
            with open(path_user_id_mapping, 'wb') as f:
                pickle.dump(mapping, f)
        return mapping



    def read_app_id_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_app_id_mapping = os.path.join(base_dir, "data_exist/app_id_mapping.pkl")
        if os.path.exists(path_app_id_mapping):
            with open(path_app_id_mapping, 'rb') as f:
                mapping = pickle.load(f)

        else:
            count = int(0)
            with open(path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line not in mapping.keys():
                        mapping[line] = int(count)
                        count += 1
            with open(path_app_id_mapping, 'wb') as f:
                pickle.dump(mapping, f)
        return mapping


    def build_valid_data(self, path):
        intr = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
        path_valid_data = os.path.join(base_dir, "data_exist/valid_data.pkl")
        if os.path.exists(path_valid_data):
            with open(path_valid_data, 'rb') as f:
                intr = pickle.load(f)
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')
                    user = self.user_id_mapping[line[0]]

                    if user not in intr:
                        intr[user] = [self.app_id_mapping[game] for game in line[1:]]
            with open(path_valid_data, 'wb') as f:
                pickle.dump(intr, f)
        return intr



    def build_test_data(self, path):
        intr = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_valid_data = os.path.join(base_dir, "data_exist/test_data.pkl")
        if os.path.exists(path_valid_data):
            with open(path_valid_data, 'rb') as f:
                intr = pickle.load(f)
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')
                    user = self.user_id_mapping[line[0]]
                    if user not in intr:
                        intr[user] = [self.app_id_mapping[game] for game in line[1:]]
            with open(path_valid_data, 'wb') as f:
                pickle.dump(intr, f)
        return intr



    def read_game_genre_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_type_mapping = os.path.join(base_dir, "data_exist/game_genre_mapping.pkl")
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])


                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]] 

                mapping_sort = {}
                for key in range(game_num):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)  

            return mapping



    def read_game_dev_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_type_mapping = os.path.join(base_dir, "data_exist/game_dev_mapping.pkl")
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])


                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]]

                mapping_sort = {}
                for key in range(len(lines)):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)


            return mapping_sort





    def read_game_pub_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_type_mapping = os.path.join(base_dir, "data_exist/game_pub_mapping.pkl")
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])


                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]]

                mapping_sort = {}
                for key in range(len(lines)):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)


            return mapping_sort




    def read_play_time_rank(self, game_path, time_path):  
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path = os.path.join(base_dir, "data_exist")
        path_tensor = path + "/tensor_user_game.pth"
        path_dic = path + "/dic_user_game.pkl"

        if os.path.exists(path_tensor) and os.path.exists(path_dic):
            tensor_user_game = torch.load(path_tensor)
            with open(path_dic, "rb") as f:
                dic_user_game = pickle.load(f)
            return tensor_user_game, dic_user_game

        else:
            ls = []
            dic_game = {}
            with open(game_path, 'r') as f_game:
                with open(time_path, 'r') as f_time:
                    lines_game = f_game.readlines()
                    lines_time = f_time.readlines()
                    for i in tqdm(range(len(lines_game))):
                        line_game = lines_game[i].strip().split(',')
                        line_time = lines_time[i].strip().split(',')
                        user = self.user_id_mapping[line_game[0]]  
                      
                        if user not in dic_game:
                            dic_game[user] = []

                       
                        idx_time_filtered = [j for j in range(1, len(line_time)) if line_time[j] != r'\N']
                        line_time_filtered = [float(line_time[j]) for j in idx_time_filtered]

                        if len(line_time_filtered) > 0:
                            ar_time = np.array(line_time_filtered)
                            time_mean = np.mean(ar_time)  
                        else:
                            continue

                        for j in range(1, len(line_game)):  
                            game = self.app_id_mapping[line_game[j]]  
                            dic_game[user].append(game)  

                            if line_time[j] == r'\N':
                                ls.append([user, game, None])
                                continue  

                            time = float(line_time[j])  
                            ls.append([user, game, time])  

            
            with open(path_dic, 'wb') as f:
                pickle.dump(dic_game, f)

            
            percentile_ls = self.generate_percentile(ls)
            for record in percentile_ls:
                if record[2] is None:  
                    record[2] = -1

            
            tensor = torch.tensor(percentile_ls, dtype=torch.float)  
            torch.save(tensor, path_tensor)
            return tensor, dic_game  


    def get_time_score(self, tensor):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path = os.path.join(base_dir, "data_exist/dic_gametime.pkl")
        if not os.path.exists(path):
            dic = {}
            logging.info("reading time score...")
            for game in tqdm(self.app_id_mapping.values()): 
                mask = (tensor[:,1]==game)
                time = tensor[mask,2]
                dic[game] = time
            self.dic_gametime = dic
            with open(path, 'wb') as f:
                pickle.dump(dic,f)


        logging.info("reading time sigmoid score...")
        for game in tqdm(self.app_id_mapping.values()):

            mask = (tensor[:,1]==game)
            tensor[mask,2] = torch.tensor(self.dic_gametime[game])

        return tensor




    def game_genre_inter(self, mapping):
        game_type_inter = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_genre_inter = os.path.join(base_dir, "data_exist/game_genre_inter.pkl")
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])

            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter


    def game_dev_inter(self, mapping):
        game_type_inter = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_genre_inter = os.path.join(base_dir, "data_exist/game_dev_inter.pkl")
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])

            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter



    def game_pub_inter(self, mapping):
        game_type_inter = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_genre_inter = os.path.join(base_dir, "data_exist/game_pub_inter.pkl")
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])

            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter




    def read_app_info(self, path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_dic = os.path.join(base_dir, "data_exist/dic_app_info.pkl")
        if os.path.exists(path_dic):
            with open(path_dic, 'rb') as f:
                dic = pickle.load(f)
            return dic
        else:
            df = pd.read_csv(path, header=None)
            games = np.array(list(df.iloc[:,0])).reshape(-1,1)

            prices = np.array(list(df.iloc[:,3]))

            prices_mean = prices.mean()
            prices = prices.reshape(-1,1)


            dates = df.iloc[:,4]
            dates = np.array(list(pd.to_datetime(dates).astype('int64')))
            dates_mean = dates.mean()
            dates = (dates.astype(float)/dates.max()).reshape(-1,1)
            
            ratings = df.iloc[:,-3].replace(-1,np.nan)
            ratings_mean = ratings.mean()
            ratings = ratings.fillna(ratings_mean).values/100
            ratings = ratings.reshape(-1,1)


            app_info = np.hstack((prices,dates,ratings))
            dic = {}
            for i in range(len(games)):
                dic[self.app_id_mapping[str(games[i][0])]] = app_info[i]

            for game in self.app_id_mapping.keys():
                if game not in games:
                    dic[self.app_id_mapping[game]] = np.array([prices_mean, dates_mean, ratings_mean])


            with open(path_dic,'wb') as f:
                pickle.dump(dic,f)
            return dic

    def Get_dn_views(self, graph):
       
        save_dir = './data_exist'
        dn_graph_path = os.path.join(save_dir, f"dn_graph.bin")
        
        
        if os.path.exists(dn_graph_path):
            try:
                dn_graph, _ = dgl.load_graphs(dn_graph_path)
                logging.info(f"Loaded existing dn graph from {dn_graph_path}")
                return dn_graph[0]
            except Exception as e:
                logging.warning(f"Failed to load existing graph: {e}. Rebuilding graph...")
        
        torch.cuda.empty_cache()
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        dn_graph = graph.clone()
        dn_graph = dn_graph.to(device)
        
        src = dn_graph.edges(etype='play')[0].to(device)
        dst = dn_graph.edges(etype='play')[1].to(device)
        times = dn_graph.edges['play'].data['percentile'].to(device).float()
        
        num_edges = dn_graph.num_edges('play')
        dn_graph.edges['play'].data['noisy'] = torch.zeros(num_edges, dtype=torch.bool, device=device)
        dn_graph.edges['played by'].data['noisy'] = torch.zeros(num_edges, dtype=torch.bool, device=device)
        
        unique_users = torch.unique(src).to(device)
        batch_size = 1024
        
        
        a = 1/3  
        b = 1/3
        c = 1/3 
        
        logging.info(f"Processing {len(unique_users)} users")
        
        for i in tqdm(range(0, len(unique_users), batch_size)):
            batch_users = unique_users[i:i+batch_size]
            
            batch_mask = torch.isin(src, batch_users)
            batch_src = src[batch_mask]
            batch_dst = dst[batch_mask]
            batch_times = times[batch_mask]
            
            user_masks = (batch_src.unsqueeze(1) == batch_users.unsqueeze(0))
            user_game_counts = user_masks.sum(0)
            valid_users = user_game_counts >= 3
            
            if not valid_users.any():
                continue
            
            
            means = torch.zeros(len(batch_users), dtype=torch.float32, device=device)
            stds = torch.zeros(len(batch_users), dtype=torch.float32, device=device)
            skewness = torch.zeros(len(batch_users), dtype=torch.float32, device=device)
            
            valid_user_masks = user_masks[:, valid_users]
            valid_times = batch_times.unsqueeze(1).expand(-1, valid_user_masks.size(1))
            
           
            means[valid_users] = ((valid_times * valid_user_masks).sum(0) / 
                                valid_user_masks.sum(0).float())
            
           
            diff_squared = ((valid_times - means[valid_users]) ** 2) * valid_user_masks
            stds[valid_users] = torch.sqrt(diff_squared.sum(0) / valid_user_masks.sum(0).float())
            
            valid_std = stds > 0
            if not valid_std.any():
                continue
            
            
            valid_diff = (valid_times - means[valid_users]) / stds[valid_users]
            n = valid_user_masks.sum(0).float()
            skewness[valid_users] = (n / ((n-1)*(n-2))) * ((valid_diff ** 3) * valid_user_masks).sum(0)
            
           
            first_term = 1 - valid_times
            
            
            ranks = torch.argsort(torch.argsort(valid_times, dim=0), dim=0)
            normalized_ranks = (ranks).float() / (len(valid_times) - 1)
            normalized_ranks=1-normalized_ranks
            
            min_skew = -4.0092
            max_skew = 2.7554
            normalized_skewness = 1 - (skewness[valid_users] - min_skew) / (max_skew - min_skew)
            normalized_skewness = normalized_skewness.unsqueeze(0).expand(valid_times.size(0), -1)
            
            probs = (a * first_term + 
                    b * normalized_ranks + 
                    c * normalized_skewness) * valid_user_masks
            
          
            noise_mask = (probs >= 0.4) & valid_user_masks
            if noise_mask.any():
                edge_indices = torch.nonzero(batch_mask)[noise_mask.any(1)]
                dn_graph.edges['play'].data['noisy'][edge_indices] = True
                dn_graph.edges['played by'].data['noisy'][edge_indices] = True
        
        
        noise_edges_play = torch.nonzero(dn_graph.edges['play'].data['noisy']).squeeze()
        noise_edges_played_by = torch.nonzero(dn_graph.edges['played by'].data['noisy']).squeeze()
        
        if len(noise_edges_play.shape) > 0:
            dn_graph.remove_edges(noise_edges_play, etype='play')
            logging.info(f"Removed {len(noise_edges_play)} play edges")
        
        if len(noise_edges_played_by.shape) > 0:
            dn_graph.remove_edges(noise_edges_played_by, etype='played by')
            logging.info(f"Removed {len(noise_edges_played_by)} played by edges")
        
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            dn_graph = dn_graph.cpu()
            dgl.save_graphs(dn_graph_path, [dn_graph])
            logging.info(f"Successfully saved dn graph to {dn_graph_path}")
            dn_graph = dn_graph.to(device)
        except Exception as e:
            logging.error(f"Failed to save dn graph: {e}")
            if isinstance(e, PermissionError):
                logging.error(f"Permission denied. Please check if you have write access to {save_dir}")
        finally:
            torch.cuda.empty_cache()
        
        return dn_graph


    def get_social_score(self):
        user_num = 60742
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data_exist/social_score_wi_ci_0.75/")
        result_path = os.path.join(data_path, 'social_score_20.pkl')

        if os.path.exists(result_path):
            logging.info("reading social score matrix...")
            with open(result_path, 'rb') as f:
                mat = pickle.load(f)
            # print(mat[0])
            i_tensor = torch.tensor(mat)
            # print(tensor[:,0])
            return i_tensor

        # if os.path.exists(di_path):
        #     logging.info("reading social score matrix...")
        #     with open(di_path, 'rb') as f:
        #         mat = pickle.load(f)
        #     # print(mat[0])
        #     di_tensor = torch.tensor(mat)
        #     return ci_tensor,di_tensor
        else:
            return None


    def process(self):
        logging.info("reading genre info...")
        self.genre_mapping = self.read_game_genre_mapping(self.genre_path)
        self.genre = self.game_genre_inter(self.genre_mapping)
        logging.info("reading genre,social score info...")
        self.social_score = self.get_social_score() 
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        logging.info("reading user item play time...")
        self.user_game, self.dic_user_game = self.read_play_time_rank(self.train_game_path, self.train_time_path)

        if os.path.exists(os.path.join(base_dir, "data_exist/graph.bin")):
            graph,_ = dgl.load_graphs(os.path.join(base_dir, "data_exist/graph.bin"))
            graph = graph[0]
            self.graph = graph
            
        else:
            graph_data = {
                ('user', 'friend of', 'user'): (self.social_score[:, 0].long(), self.social_score[:, 1].long()),

                ('game', 'genre', 'type'): (torch.tensor(self.genre)[:,0], torch.tensor(self.genre)[:,1]),

                ('type', 'genred', 'game'): (torch.tensor(self.genre)[:,1], torch.tensor(self.genre)[:,0]),

                ('user', 'play', 'game'): (self.user_game[:, 0].long(), self.user_game[:, 1].long()),

                ('game', 'played by', 'user'): (self.user_game[:, 1].long(), self.user_game[:, 0].long())
            }
            graph = dgl.heterograph(graph_data)
            graph.edges['play'].data['time'] = self.user_game[:, 2].to(torch.float32) 
            graph.edges['played by'].data['time'] = self.user_game[:, 2].to(torch.float32)
      
            graph.edges['play'].data['percentile'] = self.user_game[:, 3]
            graph.edges['played by'].data['percentile'] = self.user_game[:, 3]
            graph.edges['friend of'].data['CI'] = self.social_score[:, 2]
            graph.edges['friend of'].data['DI'] = self.social_score[:, 3]
            self.graph = graph
            dgl.save_graphs(os.path.join(base_dir, "data_exist/graph.bin"),[graph])




    def __getitem__(self, i):
        pass

    def __len__(self):
        pass


    def ceshi(self):
        print(self.genre_mapping)

