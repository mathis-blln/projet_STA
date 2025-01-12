import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import date
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from ta.momentum import RSIIndicator, StochasticOscillator
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
from arch import arch_model

scaler = StandardScaler()




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
        garch_model()
    elif option == 'Predict':
        st.header('To be done')
        # predict()



#####################PREREQUIS#######################
@st.cache_resource
def download_data(ticker_symbol, start_date, end_date, prior_days=20):
    """
    Fetch financial data for a ticker between a start and end date,
    including `prior_days` of data before the start date.

    Parameters:
    - ticker_symbol: str, the stock ticker symbol (e.g., '^FCHI').
    - start_date: str, start date for the main data range (e.g., '2000-01-01').
    - end_date: str, end date for the main data range (e.g., '2024-01-01').
    - prior_days: int, number of days before the start date to include (to avoid NA when adding technical indicators).

    Returns:
    - pd.DataFrame, combined dataset including `prior_days` before the start date.
    """
    # # Convert start_date to datetime for manipulation
    # start_datetime = datetime.strptime(start_date, '%Y-%m-%d')

    # Calculate the range for prior data
    prior_start_date = (start_date - timedelta(days=prior_days * 2)).strftime('%Y-%m-%d')
    prior_end_date = (start_date - timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch prior data
    prior_data = yf.download(ticker_symbol, start=prior_start_date, end=prior_end_date)

    # Fetch main data
    main_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Take only the last `prior_days` rows of prior data
    prior_data = prior_data.tail(prior_days)

    # Concatenate the datasets
    data = pd.concat([prior_data, main_data])

    # reset the index to date
    data.reset_index(inplace=True)
    
    # Converting 'Date' to datetime object
    data['Date'] = pd.to_datetime(data['Date'])
    data['Return'] = np.log(data['Close']/data['Close'].shift(1))

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values('Price')


    return data

@st.cache_resource
def download_eco_data(start_date, end_date):

    eco_indicators_for_train = pd.read_csv('data/input/eco_data_daily_clean.csv')
    eco_indicators_for_train =eco_indicators_for_train.drop(columns=["CAC40"])

    eco_indicators_for_predict = pd.read_csv('data/input/eco_data_after_2024-10-30.csv')
    eco_indicators_for_predict =eco_indicators_for_predict.drop(columns=["CAC40"])
    eco_data= pd.concat([eco_indicators_for_train, eco_indicators_for_predict])

    eco_data = eco_data.rename(columns={"DATE": "Date"})
    eco_data['Date'] = pd.to_datetime(eco_data['Date'])
    eco_data = eco_data[(eco_data['Date'].dt.date >= start_date) & (eco_data['Date'].dt.date <= end_date)]

    return eco_data.reset_index(drop=True)

@st.cache_resource
def add_indicators(dataset, shift_values=False):
    """
    Function to generate technical indicators with an option to shift values.

    Parameters:
    - dataset (pd.DataFrame): Input dataset containing 'Close', 'Open', 'High', and 'Low'.
    - shift_values (bool): If True, shifts all calculated indicators by 1 period.

    Returns:
    - pd.DataFrame: Dataset with technical indicators.
    """
    # Create 5, 10 and 20 days Moving Average
    dataset['SMA_5'] = dataset['Close'].rolling(window=5).mean()
    dataset['SMA_10'] = dataset['Close'].rolling(window=10).mean()
    dataset['SMA_20'] = dataset['Close'].rolling(window=20).mean()

    # Create 5, 10 and 20 days Weighted Moving Average
    dataset['WMA_5'] = dataset['Close'].rolling(window=5).apply(lambda x: np.dot(x, np.arange(1, 6))/15, raw=True)
    dataset['WMA_10'] = dataset['Close'].rolling(window=10).apply(lambda x: np.dot(x, np.arange(1, 11))/55, raw=True)
    dataset['WMA_20'] = dataset['Close'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, 21))/210, raw=True)

    # Create MACD
    ema26 = dataset['Close'].ewm(span=26).mean()
    ema12 = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = ema12 - ema26

    # RSI
    dataset['RSI'] = RSIIndicator(dataset['Close'].squeeze(), window=10).rsi()

    # Stochastic Oscillator (%K)
    stochastic_oscillator = StochasticOscillator(
        close=dataset['Close'],
        low=dataset['Low'],
        high=dataset['High'],
        window=14,
        smooth_window=3
    )
    dataset['Stochastic_K'] = stochastic_oscillator.stoch()  # %K
    dataset['Stochastic_D'] = dataset['Stochastic_K'].rolling(window=3).mean()  # %D (smoothed %K)

    # Williams %R
    dataset['Williams_R'] = (dataset['High'].rolling(window=14).max() - dataset['Close']) / \
                            (dataset['High'].rolling(window=14).max() - dataset['Low'].rolling(window=14).min())

    # High-Low Average and Close-Open
    dataset['Close_minus_Open'] = dataset['Close'] - dataset['Open']
    dataset['High_minus_Low'] = dataset['High'] - dataset['Low']

    # Commodity Channel Index (CCI)
    TP = (dataset['High'] + dataset['Low'] + dataset['Close']) / 3
    SMA = TP.rolling(window=10).mean()
    mad = TP.rolling(window=10).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    dataset['CCI'] = (TP - SMA) / (0.015 * mad)

    # Momentum
    dataset['Momentum'] = dataset['Close'].diff(periods=10)

    # Apply shifting if shift_values is True
    if shift_values:
        indicator_columns = [
            'SMA_5', 'SMA_10', 'SMA_20',
            'WMA_5', 'WMA_10', 'WMA_20',
            'MACD', 'RSI', 'Stochastic_K', 'Stochastic_D',
            'Williams_R', 'Close_minus_Open', 'High_minus_Low',
            'CCI', 'Momentum'
        ]
        dataset[indicator_columns] = dataset[indicator_columns].shift(1)

    return dataset

def get_date_range():
    min_date = date(2000, 1, 1)
    max_date = date(2026, 1, 1)
    today = date.today()
    fixed_start_date = max(today - timedelta(days=365), min_date)

    st.sidebar.write("Select date dange for visualisation")
    start_date = st.sidebar.date_input(
        'Start date',
        value=fixed_start_date,
        min_value=min_date,
        max_value=max_date
    )
    end_date = st.sidebar.date_input(
        'End date',
        value=today,
        min_value=min_date,
        max_value=max_date
    )
    return start_date, end_date


def plot_technical_indicators(dataset, selected_variables):
    subplot_mapping = [
        ("Price and Moving Averages", ["SMA_5", "SMA_10", "SMA_20", "WMA_5", "WMA_10", "WMA_20", "RSI"]),
        ("MACD and Momentum", ["MACD", "Momentum"]),
        ("Commodity Channel Index", ["CCI"]),
        ("Stochastic Oscillator and Williams %R", ["Stochastic_K", "Stochastic_D", "Williams_R"]),
        ("High Low Average", ["Close_minus_Open", "High_minus_Low"])
    ]

    for title, available_indicators in subplot_mapping:
        indicators_to_plot = [var for var in selected_variables.get(title, []) if var in available_indicators]

        if indicators_to_plot or title == "Price and Moving Averages":
            fig = go.Figure()

            # Add 'Close' to the 'Price and Moving Averages' plot
            if title == "Price and Moving Averages":
                fig.add_trace(go.Scatter(x=dataset['Date'], y=dataset['Close'], mode='lines', name='Close'))

            for indicator in indicators_to_plot:
                if indicator == "RSI":
                    fig.add_trace(go.Scatter(
                        x=dataset[dataset[indicator] > 70]['Date'], 
                        y=dataset[dataset[indicator] > 70]['Close'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-up'), 
                        name='RSI > 70 (Overbought)'))

                    fig.add_trace(go.Scatter(
                        x=dataset[dataset[indicator] < 30]['Date'], 
                        y=dataset[dataset[indicator] < 30]['Close'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-down'), 
                        name='RSI < 30 (Oversold)'))

                else:
                    fig.add_trace(go.Scatter(x=dataset['Date'], y=dataset[indicator], mode='lines', name=indicator))


            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Value",
                showlegend=True,
                template="plotly_white"
            )


            st.plotly_chart(fig, use_container_width=True)

# Reusable function for plotting indicators
def plot_indicator(data, indicator_name, column_name, title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column_name], mode='lines', name=indicator_name))
    fig.update_layout(
        title=title or indicator_name,
        xaxis_title="Date",
        yaxis_title=indicator_name,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
############################################


## Download data for visualisation and recent data
option = "^FCHI"
start_date, end_date = get_date_range()
new_data = download_data(option, start_date, end_date)
new_data = add_indicators(new_data, shift_values=False)
eco_data = download_eco_data(start_date, end_date)

data = pd.merge(new_data, eco_data, on="Date", how="right")
data.index = data['Date']

############# SECTION VISUALIZE ####################
# Main function for technical indicators
def tech_indicators(option=option):

    # Stock Price Evolution
    col1, col2 = st.columns(2)
    with col1:
        plot_indicator(data, "Close Price", "Close", "Close Price")
    with col2:
        plot_indicator(data, "Log-Returns", "Return", "Log-Returns")

    # Technical Indicators
    st.header('Technical Indicators Visualization')
    st.write('Select variables for each category')

    # User-defined selections for each plot category
    categories = {
        "Price and Moving Averages": [ "RSI", "SMA_5", "SMA_10", "SMA_20", "WMA_5", "WMA_10", "WMA_20"],
        "MACD and Momentum": ["MACD", "Momentum"],
        "Commodity Channel Index": ["CCI"],
        "Stochastic Oscillator and Williams %R": ["Stochastic_K", "Stochastic_D", "Williams_R"],
        "High Low Average": ["Close_minus_Open", "High_minus_Low"]
    }

    # Initialize a dictionary to store user-selected variables for each category
    selected_variables = {}

    cols = st.columns(len(categories))
    for col, (category, variables) in zip(cols, categories.items()):
        with col:
            selected = st.multiselect(
                category, 
                variables, 
                default=variables[:1],
                key=f"select_{category}"
            )
            if not selected:
                st.warning(f"At least one variable must be selected for {category}.")
                selected = [variables[0]]
            selected_variables[category] = selected

    # Plot Technical Indicators using the selected variables
    plot_technical_indicators(data, selected_variables)

    # Plot Economic Indicators
    # # Economical Indicators
    st.header('Economical Indicators')
    econ_option = st.selectbox(
        'Choose an Economical Indicator to Visualize',
        ['GDPV', 'IRL', 'IRS', 'GDP', 'CPIH_YTYPCT', 'IRCB', 'UNR', 'YPH', 'UNR_us', 'IRCB_us', 'CPI_us']
    )
    plot_indicator(data, econ_option, econ_option)


############## SECTION RECENT DATA ####################
# Main function for displaying data
def dataframe():
    st.header('Recent Data')

    # Define variable categories
    cac_40_columns = ["Close", "High", "Low", "Open", "Volume", "Return"]
    technical_columns = ["MACD", "RSI", "SMA_5", "SMA_10", "SMA_20", "WMA_5", "WMA_10", "WMA_20", "Close_minus_Open", "High_minus_Low", "CCI", "Momentum", "Stochastic_K", "Stochastic_D", "Williams_R"]
    eco_columns = ["GDPV", "IRL", "IRS", "GDP", "CPIH_YTYPCT", "IRCB", "UNR", "YPH", "UNR_us", "IRCB_us", "CPI_us"]

    st.write("Select variables to display")

    # Sidebar multiselect for each category
    selected_cac_40 = st.multiselect("CAC 40 Price Variables", cac_40_columns, default=cac_40_columns)
    selected_technical = st.multiselect("Technical Indicators", technical_columns, default=technical_columns)
    selected_eco = st.multiselect("Economic Variables", eco_columns, default=eco_columns)

    # Combine selected variables
    selected_columns = selected_cac_40 + selected_technical + selected_eco 

    # Filter the data
    st.dataframe(data[selected_columns], hide_index=False)



########## SECTION TRAINING ####################

def plot_predictions(pred_df, real_df, n_train, n_test):
    """
    Visualizes predicted and actual prices with separate colors for training and testing data.

    Parameters:
    - pred_df: pd.Series, predicted values
    - real_df: pd.Series, actual values
    - n_train: int, number of training data points
    - n_test: int, number of testing data points

    Returns:
    - None
    """
    fig = go.Figure()


    # Add actual prices for training data
    fig.add_trace(
        go.Scatter(
            x=real_df.index[:n_train],
            y=real_df[:n_train],
            mode='lines',
            name='Actual Price (Train)',
            line=dict(width=2, color='blue')
        )
    )

    # Add actual prices for testing data
    fig.add_trace(
        go.Scatter(
            x=real_df.index[n_train:n_train + n_test],
            y=real_df[n_train:n_train + n_test],
            mode='lines',
            name='Actual Price (Test)',
            line=dict(width=2, color='orange')
        )
    )

    # Add predicted prices for training data
    fig.add_trace(
        go.Scatter(
            x=pred_df.index[:n_train],
            y=pred_df[:n_train],
            mode='lines',
            name='Predicted Price (Train)',
            line=dict(width=1, color='red', dash='longdashdot')
        )
    )


    # Add predicted prices for testing data
    fig.add_trace(
        go.Scatter(
            x=pred_df.index[n_train:n_train + n_test],
            y=pred_df[n_train:n_train + n_test],
            mode='lines',
            name='Predicted Price (Test)',
            line=dict(width=1, color='red', dash='longdashdot')
        )
    )
    # Customize layout
    fig.update_layout(
        title='Predicted vs Actual Prices',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend=dict(
            font=dict(size=14),
            x=0.02, y=0.98,  # Legend position
        ),
        template='plotly_white'  # Clean theme
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def see_train_test_split(y_train, y_test):
    """
    Visualizes the training and testing datasets in one interactive Plotly plot.

    Parameters:
    - X_train: np.array, features of the training set (not used for plotting here)
    - y_train: pd.Series, target of the training set
    - X_test: np.array, features of the test set (not used for plotting here)
    - y_test: pd.Series, target of the test set

    Returns:
    - None
    """

    # Create a combined plot with training and testing data
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(
        x=list(range(len(y_train))), 
        y=y_train, 
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2)
    ))

    # Add testing data
    fig.add_trace(go.Scatter(
        x=list(range(len(y_train), len(y_train) + len(y_test))), 
        y=y_test, 
        mode='lines',
        name='Testing Data',
        line=dict(color='orange', width=2)
    ))

    # Update layout
    fig.update_layout(
        title="Train-Test Split Visualization",
        xaxis_title="Data Points",
        yaxis_title="Target (y)",
        legend=dict(title="Legend"),
        template="plotly_white"
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

##### Train dataset

final_df = pd.read_csv('data/output/final_database.csv')

# date as datetime
final_df['Date'] = pd.to_datetime(final_df['Date'])

# Create train and test datasets
scaler = StandardScaler()

feature_cols = [
    'GDPV', 'IRL', 'IRS', 'GDP', 'CPIH_YTYPCT', 'IRCB', 'UNR',
    'YPH', 'UNR_us',  'CPI_us', 'High_minus_Low', 'Close_minus_Open',
    'SMA_5', 'SMA_10', 'SMA_20', 'WMA_5', 'WMA_10', 'WMA_20', 'Momentum',
    'RSI', 'Williams_R', 'Stochastic_K', 'Stochastic_D', 'CCI', 'MACD'
]

# Select only the columns in `feature_cols` for features
X = final_df[feature_cols].dropna()

# Define the target variable
y = final_df['Close'].copy()

year = final_df['Date'].dt.year

X_train = scaler.fit_transform(X[year < 2023])
X_test = scaler.transform(X[year >= 2023])

y_test = y[year >= 2023].reset_index(drop=True)
y_train = y[year < 2023].reset_index(drop=True)

#Linear Regression
lambda_grid = np.logspace(-2, 4, 50)
ridge_cv = RidgeCV(alphas=lambda_grid,fit_intercept=True).fit(X_train, y_train) #to perform cross validation

# Extract the best alpha (lambda) value
optimal_alpha = ridge_cv.alpha_
print(f"Optimal Alpha (Regularization Strength): {optimal_alpha}")

# Coefficients of the model
coefficients = ridge_cv.coef_
intercept = ridge_cv.intercept_

# Predictions on the train and test sets
y_train_pred = ridge_cv.predict(X_train)
y_test_pred = ridge_cv.predict(X_test)

# Performance metrics for the training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_r_squared = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = train_mse ** 0.5

# Performance metrics for the test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r_squared = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = test_mse ** 0.5

# Prepare the data for the summary table
summary_table = {
    "Metric": ["R²", "MSE", "RMSE", "MAE"],
    "Train": [train_r_squared, train_mse, train_rmse, train_mae],
    "Test": [test_r_squared, test_mse, test_rmse, test_mae]
}

summary_table = pd.DataFrame(summary_table)

residuals_train = y_train_pred - y_train
residuals_test = y_test_pred - y_test

def training():
    st.subheader('Train - Test Split')

    # Prepare data for display
    date_info = {
        "Description": ["Training Data", "Testing Data"],
        "Start Date": ["2000-01-01", "2023-01-01"],
        "End Date": ["2022-12-31", "2024-10-30"]
    }

    date_info_df = pd.DataFrame(date_info)
    st.dataframe(date_info_df, use_container_width=True, hide_index=True)

    # Display the train-test split
    see_train_test_split(y_train, y_test)

    st.subheader('Ridge regression results')

    predictions = pd.Series(np.concatenate([y_train_pred, y_test_pred]), index=final_df["Date"])
    real_values = final_df["Close"]
    real_values.index = final_df["Date"]

    plot_predictions(predictions, real_values, n_train=len(y_train), n_test=len(y_test))

    st.subheader('Ridge regression model performance : ')

    st.write(f"Optimal regularization strength ($\lambda$): {optimal_alpha}")
    st.dataframe(summary_table)

    st.subheader('Features importances (using shap values)')

    # Explain the model predictions using SHAP Linear Explainer
    explainer = shap.Explainer(ridge_cv, X_train)  # SHAP Linear Explainer
    shap_values = explainer(X_train)  # Compute SHAP values for the train set

    #  Visualize SHAP summary as a bar plot
    shap_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean SHAP Value": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Mean SHAP Value", ascending=True)

    fig = px.bar(
        shap_df, 
        x="Mean SHAP Value", 
        y="Feature", 
        orientation='h',
        labels={"Mean SHAP Value": "Mean SHAP Value", "Feature": "Feature"},
        template="plotly_white"
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def garch_model():
    """
    GARCH model analysis and visualization using Plotly.

    Parameters:
    - residuals_train: np.array, residuals of training data
    - residuals_test: np.array, residuals of testing data
    - y_train_pred: np.array, predicted values for training set
    - y_train: pd.Series, actual values for training set
    - y_test_pred: np.array, predicted values for testing set
    - y_test: pd.Series, actual values for testing set
    - final_df: pd.DataFrame, data containing the Date column

    Returns:
    - None
    """
    st.subheader('GARCH model on the residuals of ridge regression')

    # Combine residuals
    residuals_combined = np.concatenate([residuals_train, residuals_test])
    residuals = pd.DataFrame({'residuals': residuals_combined})
    residuals["Date"] = final_df["Date"]

    # Fit GARCH model to residuals
    garch_model_t = arch_model(residuals['residuals'].values, vol='Garch', p=1, q=1, dist='t')
    garch_fit_t = garch_model_t.fit(disp="off", last_obs=len(residuals_train), first_obs=0)

    # Forecast volatility for the test set
    forecast = garch_fit_t.forecast(horizon=len(residuals_test), reindex=False)
    forecast_volatility = np.sqrt(forecast.variance.values[-len(residuals_test):, 0])

    # Compute training and forecasted volatility
    vol_train = pd.Series(garch_fit_t.conditional_volatility).dropna()
    vol_forecast = pd.Series(forecast_volatility).dropna()
    volatility = pd.concat([vol_train, vol_forecast]).dropna()
    volatility.index = residuals["Date"]

    # Plot residuals and volatility using Plotly
    fig = go.Figure()

    # Add residuals
    fig.add_trace(go.Scatter(
        x=residuals["Date"], 
        y=residuals["residuals"], 
        mode='lines',
        name='Residuals',
        line=dict(color='black', width=2)
    ))

    # Add volatility
    fig.add_trace(go.Scatter(
        x=residuals["Date"], 
        y=volatility, 
        mode='lines',
        name='Volatility',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        title='Residuals and Volatility (GARCH Model)',
        xaxis_title='Date',
        yaxis_title='Values',
        legend=dict(title="Legend"),
        template="plotly_white"
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

######## PREDICT
# def forecast_future(
#     model, historical_data, eco_data_future, feature_cols, lag=60, forecast_days=30
# ):
#     """
#     Forecast future values using the trained model and future eco_data.

#     Args:
#         model: Trained model for forecasting.
#         historical_data: DataFrame containing historical data (2000-2024).
#         eco_data_future: DataFrame containing eco_data for forecasting (2024-2026).
#         feature_cols: List of feature columns used in the model.
#         lag: Number of lag days to consider for features.
#         forecast_days: Number of future days to forecast.

#     Returns:
#         forecast_df: DataFrame with forecasted values.
#     """
#     # Ensure data is in chronological order
#     historical_data = historical_data.sort_values("Date")
#     eco_data_future = eco_data_future.sort_values("Date")

#     # Combine historical and future data
#     combined_data = pd.concat([historical_data, eco_data_future], ignore_index=True)

#     # Prepare for iterative forecasting
#     forecast_results = []
#     for day in range(forecast_days):
#         # Extract the latest lag features
#         lag_features = combined_data.iloc[-lag:, :][feature_cols].values
#         lag_features = scaler.transform(lag_features)
#         lag_features = lag_features.flatten().reshape(1, -1)

#         # Predict the next day's value
#         predicted_close = model.predict(lag_features)[0]

#         # Append prediction
#         next_date = combined_data.iloc[-1]["Date"] + pd.Timedelta(days=1)
#         forecast_results.append({"Date": next_date, "Close": predicted_close})

#         # Add the prediction back to the data for the next iteration
#         new_row = combined_data.iloc[-1].copy()
#         new_row["Date"] = next_date
#         new_row["Close"] = predicted_close
#         combined_data = combined_data.append(new_row, ignore_index=True)

#     # Convert results to DataFrame
#     forecast_df = pd.DataFrame(forecast_results)
#     return forecast_df

# def predict():
#     st.header("Forecast Future Prices")
#     start = datetime.date(2000, 1, 1)
#     end = datetime.date(2024, 10, 30)

#     # Prepare data
#     historical_data = download_data(option, start_date=start, end_date=end)
#     historical_data = add_indicators(historical_data)

#     eco_data_future = download_eco_data(start_date=datetime.date(2024, 11, 1), end_date=datetime.date(2026, 1, 1))
#     eco_data_future = eco_data_future.merge(historical_data, on="Date", how="left")

#     st.write("Training the model on historical data (2000-2024)...")
#     (X_train, y_train), (X_test, y_test) = create_dataset(historical_data, feature_cols)

#     # Select model
#     model_option = st.selectbox("Choose a model", ["RidgeRegression", "XGB", "SVM"])
#     if model_option == "RidgeRegression":
#         model = Ridge()
#     elif model_option == "XGB":
#         model = XGBRegressor()
#     elif model_option == "SVM":
#         model = SVR()

#     # Train model
#     model.fit(X_train, y_train)
#     st.success("Model trained successfully!")

#     # Forecast future prices
#     forecast_days = st.number_input("Days to forecast:", value=30, min_value=1, max_value=365)
#     forecast_df = forecast_future(model, historical_data, eco_data_future, feature_cols, forecast_days=forecast_days)

#     # Plot results
#     fig = go.Figure()

#     # Historical data
#     fig.add_trace(
#         go.Scatter(
#             x=historical_data["Date"], y=historical_data["Close"],
#             mode="lines", name="Historical Prices"
#         )
#     )

#     # Forecasted data
#     fig.add_trace(
#         go.Scatter(
#             x=forecast_df["Date"], y=forecast_df["Close"],
#             mode="lines", name="Forecasted Prices"
#         )
#     )

#     fig.update_layout(
#         title="Stock Price Forecast",
#         xaxis_title="Date",
#         yaxis_title="Close Price",
#         template="plotly_white"
#     )
#     st.plotly_chart(fig)


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





if __name__ == '__main__':
    main()
