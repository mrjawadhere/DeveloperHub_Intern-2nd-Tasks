import argparse
import pandas as pd
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--id', required=True, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    row = df[df['Unnamed: 0']==args.id]
    X = row.drop(['SeriousDlqin2yrs','Unnamed: 0'], axis=1, errors='ignore')
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.joblib')]
    model = joblib.load(os.path.join(args.model_dir, model_files[0]))

    prob = model.predict_proba(X)[:,1][0]
    print(f'Customer ID: {args.id}, Default Probability: {prob:.4f}')

if __name__ == '__main__':
    main()
