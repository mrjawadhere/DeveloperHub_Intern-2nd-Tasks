import argparse
from src.data_fetch import fetch_data
from src.indicators import calculate_indicators
from src.anomaly_detection import detect_isolation_forest, detect_dbscan
from src.forecasting import forecast_prophet
from src.visualization import plot_anomalies

def main():
    parser = argparse.ArgumentParser(description='Financial Anomaly Detection Tool')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of ticker symbols')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    args = parser.parse_args()

    data = fetch_data(args.tickers, args.start, args.end)

    for ticker, df in data.items():
        df = calculate_indicators(df)
        df = detect_isolation_forest(df, ['Close', 'SMA_20', 'EMA_20', 'RSI_14'])
        df = detect_dbscan(df, ['Close', 'SMA_20', 'EMA_20', 'RSI_14'])
        df, forecast = forecast_prophet(df)
        plot_anomalies(df, ticker)

if __name__ == '__main__':
    main()
