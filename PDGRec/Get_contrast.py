import torch
import os

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from utils.dataloader_steam import Dataloader_steam_filtered
from utils.parser import parse_args

args = parse_args()


path = "/PDGRec/steam_data"


user_id_path = path + '/users.txt'
app_id_path = path + '/app_id.txt'

genre_path = path + '/Games_Genres.txt'


dataloader = Dataloader_steam_filtered(args,
                                   path,
                                   user_id_path,
                                   app_id_path,
                                   genre_path,                       
                                   device)  


print("Getting denoised graph...")
denoised_graph = dataloader.Get_Contrast_views(dataloader.graph)

a,b,c=dataloader.calculate_user_genre_noise_and_weights(dataloader.graph,denoised_graph,dataloader.genre_mapping,device)
print(b)
output_path = "/PDGRec/user_genre_noise_readable.txt"

with open(output_path, 'w') as f:
    for user_id, genre_stats in a.items():
        
        genre_info = [f"{genre}:{ratio:.3f}" for genre, ratio in genre_stats.items()]
        line = f"User {user_id}: " + ", ".join(genre_info)
        f.write(line + '\n')
