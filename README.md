# Project Name

This project is the official implementation of the paper: **Playtime-guided Collaborative Denoising and Social Generation for Game Recommendation: Balancing Accuracy and Diversity**

## Project Overview

This project implements the PDGRec proposed in the paper and provides complete code for training, testing, and evaluation. Key features include:
- **Model Implementation**: Implements the PDGRec proposed in the paper.
- **Dataset Support**: Supports the Steam dataset.
- **Training & Testing**: Provides complete training and testing scripts.
- **Evaluation Metrics**: Implements evaluation metrics used in the paper, such as Recall, NDCG, Hit, Precision, Coverage.

## Environment Setup

### Dependencies Installation
Ensure the following dependencies are installed:
- Python 3.9.20
- PyTorch 2.3.1
- Other required libraries

## Run the Preprocessing Script  

Before running, please change to the working directory and extract the following files:  

- `steam_data/user_game.txt.zip`  
- `data_exist/social_score_wi_ci_0.75/social_score_20.zip`  

### Get Denoised Graph and Calculate Weights
After placing the prepared `social_score_20.pkl` file in the `data_exist/social_score_wi_ci_0.75/` directory, proceed with the following steps to generate the denoised graph and calculate the necessary weights.
```bash
python PDGRec/utils/Get_denoised_graph.py  
python PDGRec/utils/Get_noise_graph.py  
python PDGRec/utils/Get_CI_DI_noise_weight.py  
```
## Model Training
```bash
python main.py 
```
