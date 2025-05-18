from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import numpy as np

def detect_isolation_forest(df, features):
    clf = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_if'] = clf.fit_predict(df[features])
    return df

def detect_dbscan(df, features, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    df['anomaly_db'] = clustering.fit_predict(df[features])
    # DBSCAN labels -1 are anomalies
    df['anomaly_db'] = df['anomaly_db'].apply(lambda x: -1 if x == -1 else 1)
    return df
