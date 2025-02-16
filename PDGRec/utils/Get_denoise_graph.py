import torch
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from utils.dataloader_steam import Dataloader_steam_filtered
from utils.parser import parse_args

args = parse_args()


path = "./steam_data"


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
