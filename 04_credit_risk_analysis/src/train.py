import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import joblib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(['SeriousDlqin2yrs','Unnamed: 0'], axis=1, errors='ignore')
    y = df['SeriousDlqin2yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = {
        'rf': RandomForestClassifier(random_state=42),
        'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    params = {
        'rf': {'n_estimators': [100, 200], 'max_depth': [5,10]},
        'xgb': {'n_estimators': [100, 200], 'max_depth': [3,6]}
    }

    os.makedirs(args.model_dir, exist_ok=True)
    best_score = 0
    best_model = None
    best_name = ''
    for name, model in models.items():
        grid = GridSearchCV(model, params[name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_res, y_res)
        preds = grid.predict_proba(X_test)[:,1]
        score = roc_auc_score(y_test, preds)
        print(f'{name} ROC AUC: {score:.4f}')
        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name

    joblib.dump(best_model, os.path.join(args.model_dir, f'{best_name}_model.joblib'))
    print(f'Best model ({best_name}) saved with ROC AUC: {best_score:.4f}')

if __name__ == '__main__':
    main()
