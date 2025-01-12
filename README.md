# Stock Price Prediction

This project was carried out as part of a time series project with Youssef Esstafa for final-year students at ENSAI.  
We focused on predicting stock prices of the CAC 40. The main objective was to develop a model capable of forecasting future stock prices based on historical stock price data.  
For this purpose, we used data from 2000 to October 30, 2024, to test various models.

This project was carried out by a group of five students:
- Cheryl KOUADIO  
- Mathis BOUILLON  
- Marilène KOUGOUM MOKO MANI  
- Mountaga WANE  
- Mariyam OUYASSIN  

## Models Tested

Before testing the models, a preliminary descriptive analysis of the series was conducted in the `Descriptive_analysis.ipynb` file.  

Three categories of models were tested to forecast stock prices:  
- Time series models such as ARIMA, ARMAX, ARDL, Prophet, and GARCH, located in the folder `Time series models`.  
- Machine learning models such as Ridge Regression and Gradient Boosting with various weak learners, located in the folder `Machine learning models`.  
- Deep learning models such as LSTM and GRU, located in the folder `Deep learning models`.  

## Features

Stock price prediction is a challenging task, and predicting stock prices accurately is notoriously difficult due to the many factors that influence market movements.  
Additionally, historical data alone may not capture all relevant information. Hence, we considered adding new features such as macroeconomic variables and technical indicators.  
These features are available in the folder `data`. The subfolder `input` contains the processed macroeconomic variables, and the subfolder `output` contains the final database used for training.

### Macroeconomic Variables

Here is the list of macroeconomic indicators used:

- Consumer Price Index for the United States (CPI_us)
- Central Bank interest rate for the United States (IRCB_us)
- Unemployment rate for the United States (UNR_us)
- Current household expenditures, including nonprofit institutions serving households (YPH)
- Gross Domestic Product in volume, at market prices (GPDV)
- Long-term interest rate on government bonds (IRL)
- Short-term interest rate (IRS)
- Gross Domestic Product in nominal value, at market prices (GDP)
- Harmonized overall inflation rate (CPIH_YTYPCT)

We selected Economic Outlook No. 116, which is the latest version for the year 2024, from the OECD Data Explorer.  
These data are available in the "Economic Outlook" section within the "Economy" theme and provide forecasts starting from Q3 of 2024.  
Their extraction is detailed in the `creation_eco_var.Rmd` file, which produces two files: one for training the models (`eco_data_daily_clean.csv`) and another for the forward-looking approach (`eco_data_after_2024-10-30.csv`).

### Technical Indicators

Here is the list of technical indicators used:

- Stock High minus Low price (High_minus_Low)  
- Stock Close minus Open price (Close_minus_Open)  
- Simple Moving Average (SMA) for 5, 10, or 20 days (SMA_5, SMA_10, SMA_20)  
- Weighted Moving Average (WMA) for 5, 10, or 20 days (WMA_5, WMA_10, WMA_20)  
- Momentum (Momentum)  
- Stochastic %K (Stochastic_K)  
- Stochastic %D (Stochastic_D)  
- Moving Average Convergence Divergence (MACD)  
- Relative Strength Index (RSI)  
- Larry Williams' %R (Williams_R)  
- Commodity Channel Index (CCI)

For prediction purposes, we use the 1-lag values of these technical indicators.  
Their creation is implemented in the `Process_data.ipynb` file.

## Application

One of the objectives of the project is to deploy an application for the best model in a real-world trading environment as part of a trading system or to use it for investment decision support.  
For this purpose, we created a Streamlit application.

## Setup

1. Clone the repository
```{bash}
git clone git@github.com:mathis-blln/projet_STA.git
```
2. Navigate to the project
```{bash}
cd projet_STA
```
3. Install the required Python packages using pip:
```{bash}
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```{bash}
streamlit run app.py
```
2. The app will open in your default web browser. Use the sidebar to choose options for visualization, recent data display, or making price predictions.
3. Follow the on-screen instructions to input the stock symbol, select a date range, and choose technical indicators or prediction models.


# Time_series_proj-3A
