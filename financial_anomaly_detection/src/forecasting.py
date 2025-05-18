from prophet import Prophet
import pandas as pd

def forecast_prophet(df):
    ts = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet()
    m.fit(ts)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    fc = forecast[['ds', 'yhat']].set_index('ds')
    df = df.join(fc, how='left')
    df['deviation'] = df['Close'] - df['yhat']
    return df, forecast
