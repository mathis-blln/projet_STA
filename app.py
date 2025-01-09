import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import MACD, SMAIndicator, WMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.metrics import MeanAbsolutePercentageError, MeanSquaredError
from sklearn.linear_model import Ridge
import plotly.graph_objects as go




st.title('Stock Price Predictions of CAC 40')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

def main():
    option = st.sidebar.radio('Make a choice', ['Visualize','Recent Data',"Training",'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Training':
        training()
    elif option == 'Predict':
        st.header('To be done')
        # predict()



def calculate_cci(df, period=10):
    TP = (df["High"] + df["Low"] + df["Close"]) / 3
    SMA = TP.rolling(window=period).mean()
    mad = TP.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    CCI = (TP - SMA) / (0.015 * mad)
    return CCI

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date)
    df.columns = df.columns.droplevel('Ticker')
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])

    return df

@st.cache_resource
def download_eco_data(start_date, end_date):

    eco_indicators_for_train = pd.read_csv('data/input/eco_data_daily_clean.csv')
    eco_indicators_for_train =eco_indicators_for_train.drop(columns=["CAC40"])

    eco_indicators_for_predict = pd.read_csv('data/input/eco_data_after_2024-10-30.csv')
    eco_indicators_for_predict =eco_indicators_for_predict.drop(columns=["CAC40"])
    eco_data= pd.concat([eco_indicators_for_train, eco_indicators_for_predict])

    eco_data = eco_data.rename(columns={"DATE": "Date"})
    eco_data['Date'] = pd.to_datetime(eco_data['Date']).dt.date
    eco_data = eco_data[(eco_data['Date'] >= start_date) & (eco_data['Date'] <= end_date)]

    return eco_data.reset_index(drop=True)

@st.cache_resource
def add_indicators(df):
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    # Calculating and shifting technical indicators
    df["MACD"] = MACD(df["Close"].squeeze(), window_slow=26, window_fast=12, window_sign=9).macd().shift(1)
    df["RSI"] = RSIIndicator(df["Close"].squeeze(), window=10).rsi().shift(1)
    df["SMA_5"] = SMAIndicator(df["Close"].squeeze(), window=5).sma_indicator().shift(1)
    df["SMA_10"] = SMAIndicator(df["Close"].squeeze(), window=10).sma_indicator().shift(1)
    df["SMA_20"] = SMAIndicator(df["Close"].squeeze(), window=20).sma_indicator().shift(1)
    df["WMA_5"] = WMAIndicator(df["Close"].squeeze(), window=5).wma().shift(1)
    df["WMA_10"] = WMAIndicator(df["Close"].squeeze(), window=10).wma().shift(1)
    df["WMA_20"] = WMAIndicator(df["Close"].squeeze(), window=20).wma().shift(1)
    df["Close_minus_Open"] = (df["Close"] - df["Open"]).shift(1)
    df["High_minus_Low"] = (df["High"] - df["Low"]).shift(1)
    df["CCI"] = calculate_cci(df).shift(1)
    df["Momentum"] = df["Close"].diff(periods=10).shift(1)
    return df



option = "^FCHI"
scaler = StandardScaler()

def get_date_range():
    min_date = datetime.date(2000, 1, 1)
    max_date = datetime.date(2026, 1, 1)
    today = datetime.date.today()
    fixed_start_date = max(today - datetime.timedelta(days=365), min_date)

    st.write("Select Date Range for Analysis")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            'Start Date',
            value=fixed_start_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            'End Date',
            value=today,
            min_value=min_date,
            max_value=max_date
        )
    return start_date, end_date

# Reusable function for plotting indicators
def plot_indicator(data, indicator_name, column_name, title=None):
    st.write(title or indicator_name)
    st.line_chart(data[column_name])

# Main function for technical indicators
def tech_indicators(option=option):
    start_date, end_date = get_date_range()
    data = download_data(option, start_date, end_date)
    data = add_indicators(data)
    eco_data = download_eco_data(start_date, end_date)

    # data.set_index("Date", inplace=True)
    # eco_data.set_index("Date", inplace=True)

    # Stock Price Evolution
    st.header('Stock Price Evolution')
    col1, col2 = st.columns(2)
    with col1:
        plot_indicator(data, "Close Price", "Close", "Close Price")
    with col2:
        plot_indicator(data, "Log-Returns", "Return", "Log-Returns")

    # Technical Indicators
    st.header('Technical Indicators')
    tech_option = st.selectbox(
        'Choose a Technical Indicator to Visualize',
        ['MACD', 'RSI', 'SMA', 'WMA', 'Close - Open', 'High - Low', 'CCI', 'Momentum']
    )
    plot_map = {
        'MACD': "MACD",
        'RSI': "RSI",
        'SMA': ["SMA_5", "SMA_10", "SMA_20"],
        'WMA': ["WMA_5", "WMA_10", "WMA_20"],
        'Close - Open': "Close_minus_Open",
        'High - Low': "High_minus_Low",
        'CCI': "CCI",
        'Momentum': "Momentum",
    }
    selected_columns = plot_map.get(tech_option)
    if isinstance(selected_columns, list):
        for col in selected_columns:
            plot_indicator(data, tech_option, col)
    else:
        plot_indicator(data, tech_option, selected_columns)

    # Economical Indicators
    st.header('Economical Indicators')
    econ_option = st.selectbox(
        'Choose an Economical Indicator to Visualize',
        ['GDPV', 'IRL', 'IRS', 'GDP', 'CPIH_YTYPCT', 'IRCB', 'UNR', 'YPH', 'UNR_us', 'IRCB_us', 'CPI_us']
    )
    plot_indicator(eco_data, econ_option, econ_option)

# Main function for displaying data
def dataframe(option=option):
    start_date, end_date = get_date_range()
    data = download_data(option, start_date, end_date)
    data = add_indicators(data)
    eco_data = download_eco_data(start_date, end_date)

    st.header('Recent Data')
    st.write("Data from download_data:")
    st.markdown(", ".join(data.columns))
    st.dataframe(data, hide_index=True)

    st.header('Economical data:')
    st.text('Starting from 2024-10-30, we use predictions from OECD')
    st.markdown(", ".join(eco_data.columns))
    st.dataframe(eco_data, hide_index=True)


def run_model(engine, X_train, y_train, X_test, y_test, feature_cols, lag=60):
  with st.spinner("Training the model... Please wait."):
    #Create model
    model = engine()
    #Train model
    model.fit(X_train, y_train)

    #Make predictions on test set
    predicted_prices = model.predict(X_test)

    #Convert predictions
    pred_df = dataset_to_df(X_test, predicted_prices, lag=lag,feature_cols=feature_cols)
    #Get metrics on test set
    test_mse = MeanSquaredError()(predicted_prices, y_test)
    test_mape = MeanAbsolutePercentageError()(predicted_prices, y_test)

    #Make predictions on train set
    predicted_prices = model.predict(X_train)
    #Get metrics on train set
    train_mse = MeanSquaredError()(predicted_prices, y_train)
    train_mape = MeanAbsolutePercentageError()(predicted_prices, y_train)

    # Display metrics in a DataFrame
    metrics_data = {
        "Metric": ["MSE", "MAPE"],
        "Train": [train_mse, train_mape],
        "Test": [test_mse, test_mape]
    }
    metrics_df = pd.DataFrame(metrics_data)
    # Display the rounded DataFrame in Streamlit
    st.dataframe(metrics_df)

    real_df = dataset_to_df(X_test, y_test, lag=lag, feature_cols=feature_cols)

    fig = go.Figure()

    # Add predicted prices
    fig.add_trace(
        go.Scatter(
            x=pred_df['Date'],
            y=pred_df['Close'],
            mode='lines',
            name='Predicted Price',
            line=dict(width=2, color='blue')
        )
    )

    # Add actual prices
    fig.add_trace(
        go.Scatter(
            x=real_df['Date'],
            y=real_df['Close'],
            mode='lines',
            name='Actual Price',
            line=dict(width=2, color='red', dash='dash')  # Dashed line for actual prices
        )
    )

    # Customize layout
    fig.update_layout(
        title={
            'text': 'Predicted vs Actual Prices',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family="Arial")
        },
        xaxis=dict(
            title='Date',
            titlefont=dict(size=16),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title='Price',
            titlefont=dict(size=16),
            tickfont=dict(size=12),
        ),
        legend=dict(
            font=dict(size=14),
            x=0.02, y=0.98,  # Legend position
        ),
        template='plotly_white'  # Clean theme
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

feature_cols = [
            'Close',
            'GDPV', 'IRL', 'IRS', 'GDP', 'CPIH_YTYPCT', 'IRCB', 'UNR',
            'YPH', 'UNR_us', 'IRCB_us', 'CPI_us', 
            'High',
            'Low', 'Open', 'Volume', 'Return', 'High_minus_Low', 'Close_minus_Open',
            'SMA_5', 'SMA_10', 'SMA_20', 'WMA_5', 'WMA_10', 'WMA_20', 'Momentum',
            'RSI', 'CCI', 'MACD',
            "Year", "Month", "Day", "Day of Week"]

def training():
  st.header('Train - Test Split')
  start = datetime.date(2000, 1, 1)
  end = datetime.date(2024, 10, 30)

  df = download_data(option, start_date=start, end_date=end)
  df = add_indicators(df)

  eco_data = download_eco_data(start_date=start, end_date=end)

  df['Date'] = pd.to_datetime(df['Date'])
  eco_data['Date'] = pd.to_datetime(eco_data['Date'])
  final_df = pd.merge(df, eco_data, on="Date", how="left")

  # Prepare data for display
  date_info = {
      "Description": ["Training Data", "Testing Data"],
      "Start Date": [
          "2000-01-01",
          "2023-01-01"
      ],
      "End Date": [
          "2022-12-31",
          "2024-10-20"
      ]
  }

  # Convert to DataFrame
  date_info_df = pd.DataFrame(date_info)
  # Display in Streamlit
  st.dataframe(date_info_df, use_container_width=True, hide_index=True)

  (X_train, y_train), (X_test, y_test) = create_dataset(final_df, feature_cols, dataset_format="lag_features")
  see_train_test_split(X_train, y_train, X_test, y_test, feature_cols)

  st.header('Test performance')
  model = st.selectbox('Choose a model', ['RidgeRegression', 'XGB', 'SVM'])
  if st.button('Predict'):
      if model == 'RidgeRegression':
          engine = Ridge
          run_model(engine, X_train, y_train, X_test, y_test, feature_cols)
      elif model == 'XGB':
          engine = XGBRegressor
          run_model(engine, X_train, y_train, X_test, y_test, feature_cols)
      elif model == 'SVM':
          engine = SVR
          run_model(engine, X_train, y_train, X_test, y_test, feature_cols)
      
######## PREDICT
def forecast_future(
    model, historical_data, eco_data_future, feature_cols, lag=60, forecast_days=30
):
    """
    Forecast future values using the trained model and future eco_data.

    Args:
        model: Trained model for forecasting.
        historical_data: DataFrame containing historical data (2000-2024).
        eco_data_future: DataFrame containing eco_data for forecasting (2024-2026).
        feature_cols: List of feature columns used in the model.
        lag: Number of lag days to consider for features.
        forecast_days: Number of future days to forecast.

    Returns:
        forecast_df: DataFrame with forecasted values.
    """
    # Ensure data is in chronological order
    historical_data = historical_data.sort_values("Date")
    eco_data_future = eco_data_future.sort_values("Date")

    # Combine historical and future data
    combined_data = pd.concat([historical_data, eco_data_future], ignore_index=True)

    # Prepare for iterative forecasting
    forecast_results = []
    for day in range(forecast_days):
        # Extract the latest lag features
        lag_features = combined_data.iloc[-lag:, :][feature_cols].values
        lag_features = scaler.transform(lag_features)
        lag_features = lag_features.flatten().reshape(1, -1)

        # Predict the next day's value
        predicted_close = model.predict(lag_features)[0]

        # Append prediction
        next_date = combined_data.iloc[-1]["Date"] + pd.Timedelta(days=1)
        forecast_results.append({"Date": next_date, "Close": predicted_close})

        # Add the prediction back to the data for the next iteration
        new_row = combined_data.iloc[-1].copy()
        new_row["Date"] = next_date
        new_row["Close"] = predicted_close
        combined_data = combined_data.append(new_row, ignore_index=True)

    # Convert results to DataFrame
    forecast_df = pd.DataFrame(forecast_results)
    return forecast_df

def predict():
    st.header("Forecast Future Prices")
    start = datetime.date(2000, 1, 1)
    end = datetime.date(2024, 10, 30)

    # Prepare data
    historical_data = download_data(option, start_date=start, end_date=end)
    historical_data = add_indicators(historical_data)

    eco_data_future = download_eco_data(start_date=datetime.date(2024, 11, 1), end_date=datetime.date(2026, 1, 1))
    eco_data_future = eco_data_future.merge(historical_data, on="Date", how="left")

    st.write("Training the model on historical data (2000-2024)...")
    (X_train, y_train), (X_test, y_test) = create_dataset(historical_data, feature_cols)

    # Select model
    model_option = st.selectbox("Choose a model", ["RidgeRegression", "XGB", "SVM"])
    if model_option == "RidgeRegression":
        model = Ridge()
    elif model_option == "XGB":
        model = XGBRegressor()
    elif model_option == "SVM":
        model = SVR()

    # Train model
    model.fit(X_train, y_train)
    st.success("Model trained successfully!")

    # Forecast future prices
    forecast_days = st.number_input("Days to forecast:", value=30, min_value=1, max_value=365)
    forecast_df = forecast_future(model, historical_data, eco_data_future, feature_cols, forecast_days=forecast_days)

    # Plot results
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data["Date"], y=historical_data["Close"],
            mode="lines", name="Historical Prices"
        )
    )

    # Forecasted data
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Close"],
            mode="lines", name="Forecasted Prices"
        )
    )

    fig.update_layout(
        title="Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_white"
    )
    st.plotly_chart(fig)


# def predict():
#     model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
#     num = st.number_input('How many days forecast?', value=5)
#     num = int(num)
#     if st.button('Predict'):
#         if model == 'LinearRegression':
#             engine = LinearRegression()
#             model_engine(engine, num)
#         elif model == 'RandomForestRegressor':
#             engine = RandomForestRegressor()
#             model_engine(engine, num)
#         elif model == 'ExtraTreesRegressor':
#             engine = ExtraTreesRegressor()
#             model_engine(engine, num)
#         elif model == 'KNeighborsRegressor':
#             engine = KNeighborsRegressor()
#             model_engine(engine, num)
#         else:
#             engine = XGBRegressor()
#             model_engine(engine, num)




########## TRAINING

def create_timeseries(data, lag):
  x = []
  y = []

  for i in range(lag, len(data)):
      x.append(data[i - lag:i, :])
      y.append(data[i, 0])

  x, y = np.array(x), np.array(y)
  x = np.reshape(x, (x.shape[0], x.shape[1], -1))

  assert len(x) == len(y)
  return x, y

def create_lag_features(data, lag):
  x = []
  y = []

  for i in range(lag, len(data)):
      x.append(data[i - lag:i, :])
      y.append(data[i, 0])

  x, y = np.array(x), np.array(y)
  x = np.reshape(x, (x.shape[0], -1))

  assert len(x) == len(y)
  return x, y

def create_dataset(df, feature_cols, lag=60, dataset_format='timeseries', target="regression"):
  X = df.copy()

  #Replace NaN values
  for name, values in X.items():
    if values.isnull().sum() > 0:
      X[name].interpolate(inplace=True, limit_direction='both')
  assert X.isnull().sum().sum() == 0


  X['Year'] = X['Date'].dt.year
  X['Month'] = X['Date'].dt.month
  X['Day'] = X['Date'].dt.day
  X['Day of Week'] = X['Date'].dt.dayofweek

  #Throw away original date column
  X = X.drop(columns=['Date'])
  year = X['Year']

  #Move target column to first column
  if target == 'regression':
    close = X.pop("Close")
    X.insert(0, "Close", close)
  else:
    raise ValueError()

  #Keep only necessary features
  X = X[feature_cols]
  
  # Normalize data
  if scaler != None:
    X = scaler.fit_transform(X)
  else:
    X = X.to_numpy()

  #Split into train, test
  train_points = year < 2023
  train_data = X[train_points]

  test_points = year >= 2023
  test_points.iloc[test_points.argmax()-lag:] = True
  test_data = X[test_points]

  if dataset_format == 'timeseries':
    X_train, y_train = create_timeseries(train_data, lag)
    X_test, y_test = create_timeseries(test_data, lag)
  elif dataset_format == 'lag_features':
    X_train, y_train = create_lag_features(train_data, lag)
    X_test, y_test = create_lag_features(test_data, lag)

  return (X_train, y_train), (X_test, y_test)


def see_train_test_split(X_train, y_train, X_test, y_test, feature_cols):
    # Create train and test datasets
    train_df = dataset_to_df(X_train, y_train, feature_cols)
    test_df = dataset_to_df(X_test, y_test, feature_cols)
    
    # Create interactive Plotly figure
    fig = go.Figure()

    # Add train data
    fig.add_trace(
        go.Scatter(
            x=train_df['Date'], 
            y=train_df['Close'], 
            mode='lines', 
            name='Train',
            line=dict(width=2, color='blue')
        )
    )

    # Add test data
    fig.add_trace(
        go.Scatter(
            x=test_df['Date'], 
            y=test_df['Close'], 
            mode='lines', 
            name='Test',
            line=dict(width=2, color='red', dash='dash')  # Dashed line for test data
        )
    )

    # Customize layout
    fig.update_layout(
        title={
            'text': 'Closing Price',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family="Arial")
        },
        xaxis=dict(
            title='Date',
            titlefont=dict(size=16),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title='Price',
            titlefont=dict(size=16),
            tickfont=dict(size=12),
        ),
        legend=dict(
            font=dict(size=14),
            x=0.02, y=0.98,  # Legend position
        ),
        template='plotly_white'  # Clean theme
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def next_day(row):
  year, month, day = round(row['Year'].item()), round(row['Month'].item()), round(row['Day'].item())
  new_date = pd.Timestamp(year=year, month=month, day=day)
  new_date += pd.Timedelta(days=3 if int(row['Day of Week'].item()) == 4 else 1)
  return new_date

def dataset_to_df(X, Y, feature_cols,lag=60):
  if X.ndim == 3:
    last_days = X[:, -1, :]
  elif X.ndim == 2:
    n_feat = int(X.shape[1]/lag)
    last_days = X[:, -n_feat:]
  new_df = pd.DataFrame(last_days, columns=feature_cols)
  #Replace closing price with y and scale back to original values
  new_df['Close'] = Y
  if scaler != None:
    new_df = pd.DataFrame(scaler.inverse_transform(new_df), columns=feature_cols)
  #Go to next day (i.e. day of the prediction)
  new_df['Date'] = new_df.apply(next_day, axis=1)

  return pd.DataFrame(new_df, columns=["Date", "Close"])

def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1




if __name__ == '__main__':
    main()
