############# PROCESS DATA

# LOAD LIBRAIRES
library(dplyr)

# LOAD DATA

macro_var <- read.csv("data/input/eco_data_daily_clean.csv", header = TRUE, dec = ".")
macro_var <- macro_var %>% 
  rename(Date = DATE)

CAC_40 <- read.csv("data/input/cac40.csv", header = TRUE, dec = ".")

CAC_40 <- CAC_40 %>% 
  mutate(
    Return = log(Close / lag(Close)),
    High_minus_Low = lag(High - Low),
    Close_minus_Open = lag(Close - Open),
    SMA_5 = lag(SMA(Close, n = 5)),
    SMA_10 = lag(SMA(Close, n = 10)),
    SMA_20 = lag(SMA(Close, n = 20)),
    WMA_5 = lag(WMA(Close, n = 5)),
    WMA_10 = lag(WMA(Close, n = 10)),
    WMA_20 = lag(WMA(Close, n = 20)),
    Momentum = lag(momentum(Close, n = 10)),
    RSI = lag(RSI(Close, n = 10)),
    Williams_R = lag(WPR(CAC_40[, c("High", "Low", "Close")], n = 10)),
    Stochastic_K = lag(stoch(CAC_40[, c("High", "Low", "Close")])[, "fastK"]),
    Stochastic_D = lag(stoch(CAC_40[, c("High", "Low", "Close")])[, "fastD"]),
    CCI = lag(CCI(CAC_40[, c("High", "Low", "Close")], n = 10)),
    MACD = lag(MACD(Close, nFast = 12, nSlow = 26, nSig = 9)[, "macd"])
  )

# Merge data by date and CAC_40
data <- merge(macro_var, CAC_40, by = "Date")

# drop
data <- data %>% dplyr::select(-c("Open", "High", "Low", "Volume", "Adj.Close","CAC40"))
rm(list=c("macro_var", "CAC_40"))

# to csv
write.csv(data, "data/output/final_database.csv", row.names = FALSE)
