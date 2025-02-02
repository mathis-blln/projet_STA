############# GARCH MODEL

# Load libraries
library(rugarch)
library(dplyr)
library(tseries)

# Load data
data <- read.csv("data/output/final_database.csv")
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")

########## GARCH MODEL
# Volatility clusters in time series data
plot.ts(data$Return)

# Autocorrelation of the return
acf(data$Return)

# Autocorrelation of the squared return
acf(data$Return**2)
## The process looks like white noise but with no independance. Hence, we can use a GARCH model.

## Estimation of order of GARCH model
acf(data$Return**2 - mean(data$Return**2))


# [0] On Estimation of GARCH Models with an Application to Nordea Stock Prices
# Motivated by comments in [1] that in practical applications
# GARCH with smaller orders often sufficiently describe the data and in most
# cases GARCH (1, 1) is sufficient, we hence consider four different combinations
# of p = 0, 1 and q = 1, 2 for each period.

#^[1] Statistics of Financial Markets: An Introduction, Professor Dr. Jürgen Franke

data_train <- data[data$Date < "2023-01-01",]
data_test <- data[data$Date >= "2023-01-01",]

# Use of ARMA representation of e^2
find_garch <- function(p_min,p_max,q_min,q_max, data, dist="norm"){
  best_aic <- Inf 
  best_order <- c(0, 0, 0)  
  results <- data.frame(p = integer(),
                        q = integer(),
                        aic = numeric(),
                        relative_gap = numeric(),
                        stringsAsFactors = FALSE)
  
  for (p in p_min:p_max) {
    for (q in q_min:q_max) {
      garch_spec <- ugarchspec( variance.model = list(model ="sGARCH", garchOrder = c(p, q)),
                                mean.model = list(armaOrder = c(0, 0)),
                                distribution.model =dist)
      out <-  ugarchfit(spec =garch_spec, data = data)
      current_aic <- infocriteria(out)[1]*length(data)
      
      if (current_aic < best_aic) {
        best_aic <- current_aic
        best_order <- c(p, 0, q)
      }
      results <- rbind(results, data.frame(p = p, 
                                           q = q, 
                                           aic = current_aic, 
                                           relative_gap = NA))
      #}
    }
  }
  
  results$relative_gap <- (results$aic - best_aic)*100 / best_aic 
  return(results)
}

# Find the best GARCH model
p_min <- 0
p_max <- 1
q_min <- 1
q_max <- 2
results<- find_garch(p_min, p_max, q_min, q_max, data_train$Return**2,dist="norm")

# latex table
print(xtable::xtable(results, type = "latex"))
# => Best model is GARCH(1,1)

########### Fit the GARCH(1,1) model

# Fonction pour ajuster un modèle GARCH avec une distribution spécifiée
fit_garch <- function(data, n_test, distribution = "norm", model ="sGARCH",submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0)) {
  # Spécification du modèle GARCH
  garch_spec <- ugarchspec(
    variance.model = list(model = model, garchOrder = garch_order, submodel=submodel),
    mean.model = list(armaOrder = arma_order),
    distribution.model = distribution
  )
  # Ajustement du modèle GARCH
  garch_fit <- ugarchfit(spec = garch_spec, data = data, out.sample = n_test)
  # Fitted values
  fitted <- list()
  fitted$sigma <-garch_fit@fit$sigma
  fitted$mean <- garch_fit@fit$fitted
  
  return(list(garch_fit = garch_fit, fitted = fitted))
}

# Fonction pour générer des prévisions avec un modèle GARCH ajusté
forecast_garch <- function(garch_fit, data, n_test, n_ahead = 1) {
  garch_forecast <- ugarchforecast(garch_fit, data = data, n.ahead = n_ahead, n.roll = n_test - 1)
  forecast <- list()
  forecast$sigma <- as.numeric(garch_forecast@forecast$sigmaFor)
  forecast$mean <- as.numeric(garch_forecast@forecast$seriesFor)
  return(forecast)
}

# Compute performance
performance <- function(predicted, realized){
  N <- length(realized)
  # Root Mean Squared Error (RMSE)
  rmse <- sum((predicted - realized)**2) / N
  # Mean Absolute Error (MAE)
  mae <- (1 / N) * sum(abs(predicted - realized))
  # Heteroscedasticity-adjusted MAE (HMAE)
  hmae <- (1 / N) * sum(abs( 1 - predicted / realized))
  
  return(list(rmse = rmse, mae = mae, hmae = hmae))
}

# Fonction pour effectuer le workflow complet
run_garch <- function(data,n_test, distribution = "norm",model ="sGARCH",submodel= NULL, garch_order = c(1, 1), arma_order = c(0, 0)) {
  N <- length(data) 
  data_train <- data[1:(N - n_test)]
  data_test <- data[(N- n_test + 1):N]
  
  # Ajustement du modèle
  fit_results <- fit_garch(data,n_test,distribution,model, submodel, garch_order, arma_order)
  garch_fit <- fit_results$garch_fit
  
  # Prévisions
  forecast <- forecast_garch(garch_fit, data,n_test, n_ahead = 1)
  
  # Proxy de volatilité
  # # Lorsqu’on ne dispose pas de données intrajournalières pour calculer des proxies comme la Realized Volatility 
  # # ou la Realized High-Low Volatility, le deuxième meilleur proxy de la variance est généralement le carré du rendement.
  vol_proxy_test <- sqrt(data_test^2)
  vol_proxy_train <- sqrt(data_train^2)
  
  # Performance
  perf_train <- performance(sigma(garch_fit), vol_proxy_train)
  perf_test <- performance(forecast$sigma, vol_proxy_test)
  
  return(list(fit = garch_fit, fitted = fit_results$fitted, forecast = forecast, performance_train = perf_train, performance_test = perf_test, vol_proxy_test = vol_proxy_test, vol_proxy_train=vol_proxy_train))
}


n_test <- nrow(data_test)
res_norm<- run_garch(data$Return, n_test, distribution = "norm",model ="sGARCH", submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0))

res_norm$fit

#### Regarder le paramètre de forme pour indication sur les lois à tester
# Jacques bera test for normality
jarque.bera.test(res_norm$fit@fit$residuals)
# reject the null hypothesis of normality for the distribution of the
# residuals, as a rule of thumb, which implies that the data to be fitted is not
# normally distributed

# Print performance
#res_norm$performance_train
res_norm$performance_test
info <- infocriteria(res_norm$fit)


# Extract AIC and BIC
info[1]
info[2]

########## GARCH(1,1) model with Student distribution
res_std<- run_garch(data$Return, n_test, distribution = "std",model ="sGARCH", submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0))

# Print performance
res_std$performance_test

########## GARCH(1,1) model with skew student distribution
res_sstd<- run_garch(data$Return, n_test, distribution = "sstd", model ="sGARCH", submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0))

# Print performance
res_sstd$performance_test

########## GARCH(1,1) model with generalized error distribution
res_ged<- run_garch(data$Return, n_test, distribution = "ged",model ="sGARCH", submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0))

# Print performance
res_ged$performance_test

########## GARCH(1,1) model with skewed generalized error distribution
res_sged<- run_garch(data$Return, n_test, distribution = "sged",model ="sGARCH", submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0))

# Print performance
res_sged$performance_test


# Example: list of performance metrics
performance_results <- list(
  res_norm$performance_test,
  res_std$performance_test,
  res_sstd$performance_test,
  res_ged$performance_test,
  res_sged$performance_test
)

# Convert list to a dataframe
performance_df <- bind_rows(performance_results, .id = "Model") %>%
  mutate(Model = c("Norm", "Std", "Sstd", "Ged", "Sged"))

# View the structured dataframe
print(performance_df)

# latex
print(xtable::xtable(performance_df, type = "latex", digits = rep(4, ncol(performance_df) + 1)))
# summary of the model
res_sstd$fit

confint(res_sstd$fit)
# => No serial correlation in the residuals but on squared residuals (Use of Ljung-Box test)


## Lljung test till 11 lag
for (i in 1:11) {
  print(Box.test(res_sstd$fit@fit$residuals, lag = i, type = "Ljung-Box")$p.value)
}
test<- res_sstd$fit


#### Plot results of best model
res_sstd$forecast$date <- data_test$Date
res_sstd$fitted$date <- data_train$Date


plot(data_train$Date, data_train$Return, type = "l", col = "black", xlab = "Date", ylab = "Return")
lines(res_sstd$fitted$date, res_sstd$fitted$sigma, col = "red")
lines(data_test$Date, data_test$Return, type = "l", col = "blue", xlab = "Date", ylab = "Return", main = "GARCH(1,1) Forecast")
lines(res_sstd$forecast$date, res_sstd$forecast$sigma, type = "l", col = "red", xlab = "Date", ylab = "Volatility", main = "GARCH(1,1) Forecast")
legend("bottomleft", legend = c("Train","Test", "Volatility predicted"), col = c("black","blue", "red"), lty = 1:1, cex = 0.8)


plot(data_test$Date, res_sstd$vol_proxy_test, type = "l", col = "black", xlab = "Date", ylab = "Return", main = "Realized vs. Forecast")
lines(res_sstd$forecast$date, res_sstd$forecast$sigma, type = "l", col = "red", xlab = "Date", ylab = "Volatility", main = "GARCH(1,1) Forecast")


 ########### Leverage effect

# Leverage effect : Standard GARCH models assume that positive and negative error terms have
# asymmetrie eifect on the volatility. In other words, good and bad news have
# the same eifect on the volatility in this model. In practice this assumption
# is frequently violated, in particular by stock returns, in that the volatility
# increases more after bad news than after good news. This so called Leverage
# Effect appears firstly in Black (1976). => Use of EGARCH or TGARCH

########## eGARCH(1,1) model with skew student distribution
res_sstd_egarch<- run_garch(data$Return, n_test, distribution = "sstd" ,model ="eGARCH", submodel=NULL, garch_order = c(1, 1), arma_order = c(0, 0))

# Print performance
res_sstd_egarch$performance_test


########## TGARCH(1,1) model with skew student distribution
res_sstd_tgarch<- run_garch(data$Return, n_test, distribution = "sstd" ,model ="fGARCH",submodel="TGARCH", garch_order = c(1, 1), arma_order = c(0, 0))

# Print performance
res_sstd_tgarch$performance_test


## add to performance df
performance_results <- list(
  res_norm$performance_test,
  res_std$performance_test,
  res_sstd$performance_test,
  res_ged$performance_test,
  res_sged$performance_test,
  res_sstd_egarch$performance_test,
  res_sstd_tgarch$performance_test
)

# Convert list to a dataframe
performance_df <- bind_rows(performance_results, .id = "Model") %>%
  mutate(Model = c("Norm", "Std", "Sstd", "Ged", "Sged", "Sstd_egarch", "Sstd_tgarch"))

performance_df

# latex
print(xtable::xtable(performance_df, type = "latex", digits = rep(4, ncol(performance_df) + 1)))
