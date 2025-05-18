import matplotlib.pyplot as plt

def plot_anomalies(df, ticker):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.scatter(df.index[df['anomaly_if'] == -1], df['Close'][df['anomaly_if'] == -1], color='red', label='IF Anomaly')
    ax.scatter(df.index[df['anomaly_db'] == -1], df['Close'][df['anomaly_db'] == -1], color='orange', label='DBSCAN Anomaly')
    ax.plot(df.index, df['yhat'], label='Prophet Forecast')
    ax.legend()
    ax.set_title(f'Anomaly Detection for {ticker}')
    plt.savefig(f'reports/{ticker}_anomalies.png')
    plt.close()
