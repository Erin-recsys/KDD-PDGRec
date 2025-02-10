import torch
import os


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from utils.dataloader_steam import Dataloader_steam_filtered
from utils.parser import parse_args

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
else:
    print("CPU")

args = parse_args()

path = "/home/zhangjingmao/data/PDGRec/steam_data"

user_id_path = path + '/users.txt'
app_id_path = path + '/app_id.txt'

genre_path = path + '/Games_Genres.txt'


dataloader = Dataloader_steam_filtered(args,
                                   path,
                                   user_id_path,
                                   app_id_path,
                                   genre_path,                       
                                   device)

'''price_power = -0.1
rg_weight = 0.4
rd_weight = 0.3
rp_weight = 0.3'''

#try:
print("Getting denoised graph...")
denoised_graph = dataloader.Get_Contrast_views(dataloader.graph)

a,b,c=dataloader.calculate_user_genre_noise_and_weights(dataloader.graph,denoised_graph,dataloader.genre_mapping,device)
print(b)
output_path = "/home/zhangjingmao/data/PDGRec/user_genre_noise_readable.txt"

with open(output_path, 'w') as f:
    for user_id, genre_stats in a.items():

        genre_info = [f"{genre}:{ratio:.3f}" for genre, ratio in genre_stats.items()]
        line = f"User {user_id}: " + ", ".join(genre_info)
        f.write(line + '\n')

'''print("Calculating noise ratios...")
genre_noise_ratios = dataloader.calculate_noise_ratios(
    dataloader.graph,
    denoised_graph,

)'''

'''print("Creating weighted graph...")
weighted_graph = dataloader.create_weighted_graph(
    denoised_graph,
    rg_weight
)

print("Process completed successfully!")




















'''except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        torch.cuda.empty_cache()  # 清理GPU缓存
        print("GPU内存不足，请尝试减小batch size或使用CPU")
    else:
        print(f"运行时错误: {str(e)}")
except Exception as e:
    print(f"发生错误: {str(e)}")
finally:
    # 清理内存
    torch.cuda.empty_cache()'''