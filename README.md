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

## Run the preprocessing script
Before running, please extract the following files:
- steam_data/user_game.txt.zip
- data_exist/social_score_wi_ci_0.75/social_score_20.pkl.zip
1. Get Social Score
- Option 1: Use preprocessed data (Recommended)\n
  Directly use the extracted social_score_20.pkl
- Option 2: Run preprocessing scripts 
```bash
python PDGRec/utils/get_ut_preference.py
python PDGRec/utils/get_weight.py
python PDGRec/utils/cal_social_score.py
```
2. Get Denoised Graph and Calculate Weights
```bash
python PDGRec/utils/Get_contrast.py
python PDGRec/utils/Get_noise_weight.py
python PDGRec/utils/Get_CIDI_Weight.py
```
### Model Training
```bash
python main.py
```
