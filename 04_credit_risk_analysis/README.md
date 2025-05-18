# Credit Risk Analysis

This project builds a model to assess customer creditworthiness using the "Give Me Some Credit" dataset.

## Folder Structure

- `src/`: Source code modules  
- `data/`: Raw and processed data  
- `notebooks/`: Jupyter notebooks for exploration  
- `reports/`: Evaluation reports and plots  
- `models/`: Saved trained models  

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess data**  
   ```bash
   python src/data_preprocess.py --input data/give_me_some_credit.csv --output data/processed.csv
   ```
2. **Feature engineering**  
   ```bash
   python src/feature_engineering.py --input data/processed.csv --output data/features.csv
   ```
3. **Train models**  
   ```bash
   python src/train.py --input data/features.csv --model_dir models/
   ```
4. **Evaluate**  
   ```bash
   python src/evaluate.py --model_dir models/ --input data/features.csv
   ```
5. **Predict**  
   ```bash
   python src/predict.py --model_dir models/ --features data/features.csv --id 123456
   ```
