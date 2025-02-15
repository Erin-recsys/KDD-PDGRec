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
1. Get social score
```bash
python PDGRec/utils/get_ut_preference.py
python PDGRec/utils/get_weight.py
python PDGRec/utils/cal_social_score.py
```

2. Get denoised graph and calculate noise, CI and DI weights 
```bash
python PDGRec/utils/Get_contrast.py
python PDGRec/utils/Get_noise_weight.py
python PDGRec/utils/Get_Weight.py 
```

### Model Training

```bash
python main.py
```
