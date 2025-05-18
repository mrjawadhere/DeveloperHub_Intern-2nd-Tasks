# Multi-Label Emotion Recognition from Text

This project fine-tunes a transformer model on the GoEmotions dataset to perform multi-label emotion classification.

## Folder Structure

- `src/`: Source code  
- `data/`: Downloaded/processed data  
- `notebooks/`: Exploratory analysis and experiments  
- `reports/`: Evaluation metrics and plots  

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess data & train**  
   ```bash
   python src/train.py --model_name bert-base-uncased --output_dir models/
   ```
2. **Evaluate**  
   ```bash
   python src/evaluate.py --model_dir models/
   ```
3. **Predict**  
   ```bash
   python src/predict.py --model_dir models/ --text "I am so happy and excited!"
   ```
