import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    # Load test features
    df = pd.read_csv(args.input)
    X = df.drop(['SeriousDlqin2yrs','Unnamed: 0'], axis=1, errors='ignore')
    y = df['SeriousDlqin2yrs']
    # Load model
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.joblib')]
    model = joblib.load(os.path.join(args.model_dir, model_files[0]))

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    print(classification_report(y, preds))
    print(f'ROC AUC: {roc_auc_score(y, probs):.4f}')

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('reports/confusion_matrix.png')
    print('Confusion matrix saved to reports/confusion_matrix.png')

if __name__ == '__main__':
    main()
