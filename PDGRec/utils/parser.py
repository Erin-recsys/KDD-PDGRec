import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 0.03, type = float,
                        help = 'learning rate')
    parser.add_argument('--ssl_temp', default=0.5, type=float,
                    help='Temperature for SSL loss')
    parser.add_argument('--ssl_game_weight', default=1, type=float,
                    help='Weight for game SSL ')
    parser.add_argument('--ssl_loss_weight', default=10 , type=float,
                    help='Weight for SSL loss ')
    
    parser.add_argument('--balance', default=0, type=float,
                    help='hyper-parameter parameter for balance')#1 for accuracy, 0 for diversity and balance
    parser.add_argument('--K', default = 2, type = float,
                    help = 'hyper-parameter for negative score reweighting')#6.5 for accuracy, 2 for diversity and balance
    parser.add_argument('--use_other', default=1, type=float,
                    help='Whether to add other aggregation.1 for yes, 0 for no')#0 for accuracy, 1 for diversity and balance
    
    parser.add_argument('--weight_CI', default=0.3, type=float,
                    help='Weight for CI ')
    parser.add_argument('--weight_DI', default=0.7, type=float,
                    help='Weight for DI ')
  
    parser.add_argument('--weight_self', default=0.3, type=float,#1 for acc
                    help='Weight for self ')    
    parser.add_argument('--weight_noise', default=0.7, type=float, 
                    help='Weight for noise weight')

    parser.add_argument('--ssl_batch_size', default=2048, type=int,
                    help='Batch size for SSL loss computation')
    parser.add_argument('--price_power', default = -0.1, type = float,
                    help = 'training_percent')
    parser.add_argument('--train_percent', default = 0.8, type = float,
                        help = 'training_percent')
    parser.add_argument('--embed_size', default = 32, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--epoch', default = 10000000, type = int,
                        help = 'epoch number')
    parser.add_argument('--early_stop', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 1024, type = int,
                        help = 'batch size')
    parser.add_argument('--layers', default = 3, type = int,
                        help = 'layer number')
    parser.add_argument('--gpu', default = 2, type = int,
                        help = '-1 for cpu, 0 for gpu:0')
    parser.add_argument('--k', default = [5,10,20], type = list,
                        help = 'negative sampler number for each node')
    parser.add_argument('--gamma', default=80.0, type=float,
                        help='hyper-parameter for aggregation weight')
    parser.add_argument('--layers_and', default=2, type=int,
                        help='hyper-parameter for layer number')
    parser.add_argument('--layers_or', default=4, type=int,
                        help='hyper-parameter for layer number')
    parser.add_argument('--layers_user_game', default=2, type=int,
                        help='hyper-parameter for layer number')
    parser.add_argument('--attention_and', default=True, type=bool,
                        help='hyper-parameter for attention of and layers')
    parser.add_argument('--param_decay', default=0.1, type=float,
                        help='hyper-parameter for decay')
    parser.add_argument('--edge_node_weight', default=True, type=bool,
                        help='hyper-parameter for diversification in bipartite')
    args = parser.parse_args()
    return args
