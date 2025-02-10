import torch
import os

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from utils.dataloader_steam import Dataloader_steam_filtered
from utils.parser import parse_args


args = parse_args()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = base_dir + "/steam_data"

user_id_path = path + '/users.txt'
app_id_path = path + '/app_id.txt'
genre_path = path + '/Games_Genres.txt'

dataloader = Dataloader_steam_filtered(args,
                                   path,
                                   user_id_path,
                                   app_id_path,
                                   genre_path,                       
                                   device)

try:
    print("Getting denoised graph...")
    denoised_graph = dataloader.Get_Contrast_views(dataloader.graph)

    print("Calculating genre noise ratios...")
    genre_noise_ratios = dataloader.calculate_noise_ratios(
        dataloader.graph,
        denoised_graph
    )

    print("Calculating user-genre noise ratios...")
    user_genre_stats = dataloader.calculate_user_genre_ratios(
        dataloader.graph,
        denoised_graph,
        batch_size=500
    )

    sample_users = list(dataloader.user_id_mapping.values())[:5]

    for user_id in sample_users:
        user_preferences = dataloader.get_user_genre_preferences(
            user_id=user_id,
            user_genre_stats=user_genre_stats
        )
        
        print(f"\n {user_id} game genre analysis:")
        print(f"interaction num: {user_preferences['total_genres']}")
        print(f"avg noise ratio: {user_preferences['avg_noise_ratio']:.2f}")
        print("noise ratio (top 5):")
        for genre, ratio in user_preferences['genre_stats'][:5]:
            print(f"genre {genre}: {ratio:.2f}")
            
    print("\nProcess completed successfully!")
    print(f"GPU usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"GPU memory allocate: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        torch.cuda.empty_cache()

    else:
        print(f"error: {str(e)}")
except Exception as e:
    print(f"error: {str(e)}")
finally:
    torch.cuda.empty_cache()