# Satellite Image Deforestation Monitoring

This project processes Planet satellite imagery to detect and monitor deforestation areas over time.

## Folder Structure

- `src/`: Source code modules  
- `data/`: Raw and processed image data  
- `notebooks/`: Jupyter notebooks for exploration  
- `reports/`: Output maps and visualizations  

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess data**  
   ```bash
   python src/data_preprocess.py --input_dir data/raw --output_dir data/processed
   ```
2. **Train model**  
   ```bash
   python src/train.py --data_dir data/processed --model_dir models/
   ```
3. **Detect changes**  
   ```bash
   python src/change_detection.py --model_dir models/ --t1_dir data/processed/t1 --t2_dir data/processed/t2 --output reports/changes.png
   ```
4. **Visualize**  
   ```bash
   python src/visualization.py --change_map reports/changes.png
   ```
