##### FARIMA-MODEL
library(arfima)

# Chargement des données
data <- read.csv("data/output/final_database.csv")
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")


# Keep only Return
Return <- data$Return
data$Date <- as.Date(data$Date)


## Étape 2 : Analyse graphique
plot(data$Date,Return, main = "Rendements Logarithmiques du CAC 40", ylab = "Rendements", col = "blue", type = "l")
acf(Return, main = "ACF des rendements", lag.max = 150, ylim = c(0, 0.5))

plot(data$Date,Return**2, main = "Rendements Logarithmiques du CAC 40", ylab = "Rendements", col = "blue", type = "l")
acf(Return**2, main = "ACF des rendements au carré", lag.max = 150, ylim = c(0, 0.5))

pacf(Return, main = "PACF des rendements")

acf(Return**2, lag.max = 80, main = "ACF des carrés des rendements")

# Test de stationnarité avec ADF
adf_test <- adf.test(Return)
print(adf_test)

arfima(Return)

# L'autocorrélogramme des rendements indique cette chronique est stationnaire, ce qui est confirmé par le test ADF (p.value <0.05).
# L'autocorrélogramme du carré des rendements semble indiquer la persistence des corrélations. Cela motive l'estimation d'un modèle FARIMA.
# L'orde maximal ar est 5 et l'ordre maximal ma est 2.

# Étape 3 : Grid search pour le modèle FARIMA avec arfima
grid_search_farima <- function(data, max_p = 5, max_q = 2) {
  results <- list()  # Liste pour stocker les modèles
  
  for (p in 0:max_p) {
    for (q in 0:max_q) {
      # Ajustement du modèle FARIMA
      model <- arfima::arfima(data, order = c(p, 0, q))
      
      # Extraction du AIC
      aic <- AIC(model)
      
      # Vérification de la blancheur des résidus
      residuals <- residuals(model)$Mode1
      lb_pval <- Box.test(residuals, lag = 20, type = "Ljung-Box")$p.value
      
      # Stockage des résultats
      results <- append(results, list(data.frame(
        p = p,
        q = q,
        d = model$modes[[1]]$dfrac,
        aic = aic,
        LBox_pval = lb_pval
      )))
    }
  }
  
  # Convertir la liste de résultats en un data frame
  results_df <- do.call(rbind, results)
  
  # Tri des modèles par AIC croissant
  results_df <- results_df[order(results_df$aic), ]
  
  Return(results_df)
}

# Recherche des modèles
farima_results <- grid_search_farima(as.numeric(Return))

# Affichage des meilleurs résultats
farima_results

# Le modèle optimal obtenu est un FARIMA(3,d,2) avec d=-0.0277 < 0, ce qui met en doute la persistence d'une mémoire longue et contrarie un ajustement FARIMA faible. Toutefois, d n'est pas significativement différent de 0, donnant ainsi la possibilité de s'orienter vers un modèle ARIMA(3,0,2).


# Étape 4 : Sélection du meilleur modèle et vérification des diagnostics
best_model <- arfima::arfima(as.numeric(Return), order = c(3, 0, 2))

# Résumé du meilleur modèle
print(summary(best_model))


# Vérification des résidus
best_residuals <- residuals(best_model)$Mode1

par(mfrow = c(2, 1))
acf(best_residuals, main = "ACF des Résidus du FARIMA")
pacf(best_residuals, main = "PACF des Résidus du FARIMA")

adf.test(best_residuals)
Box.test(best_residuals, lag = 20, type = "Ljung-Box")

# Les résidus du modèle optimal passe le test de blancheur des résidus (p.value>0.7) et ces résidus sont stationnaire comme l'indique le corrélogramme (conirmé par le test ADF, p.value<0.05).
# Ces résisdus remplissent donc les pré-requis pour passer au modèle GARCH.

## Étape 5 : Modèle GARCH sur les résidus FARIMA
Dans cette section nous estimons un GARCH(p,q) avec p et q choisis avec parcimonie tout en privilégiant un AIC minimal.

# Estimation des modèles GARCH(p, q) avec rugarch
grid_search_garch <- function(residuals, max_p = 2, max_q = 2) {
  results <- data.frame(p = integer(), q = integer(), aic = numeric(), lb_pval = numeric())
  
  for (p in 0:max_p) {
    for (q in 0:max_q) {
      tryCatch({
        spec <- ugarchspec(
          variance.model = list(model = "sGARCH", garchOrder = c(p, q)),
          mean.model = list(armaOrder = c(0, 0)),
          distribution.model = "norm"
        )
        fit <- ugarchfit(spec = spec, data = residuals)
        
        # Stockage des résultats
        aic <- infocriteria(fit)[1]
        lb_pval <- Box.test(residuals(fit), lag = 20, type = "Ljung-Box")$p.value
        results <- rbind(results, data.frame(p = p, q = q, aic = aic, lb_pval = lb_pval))
      }, error = function(e) {})
    }
  }
  
  results <- results[order(results$aic), ]
  Return(results)
}

# Appel à la fonction
garch_results <- grid_search_garch(best_residuals)
print(garch_results)


# Le meilleur compromis conduit au modèle GARCH(1,1) comme modèle optimal.


# Modèle GARCH optimal
spec_garch <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                         distribution.model = "std")  # Distribution t-student

garch_fit <- ugarchfit(spec = spec_garch, data = best_residuals)




# Résultats du modèle GARCH
show(garch_fit)


## Étape 6 : Vérification des résidus standardisés GARCH

# Étape 6 : Vérification des résidus standardisés GARCH
garch_residuals <- residuals(garch_fit, standardize = TRUE)
garch_residuals <- ts(garch_residuals)

par(mfrow = c(2, 1))
acf(garch_residuals, lag.max = 80, main = "ACF des Résidus Standardisés du GARCH")
pacf(garch_residuals, lag.max = 80, main = "PACF des Résidus Standardisés du GARCH")
Box.test(garch_residuals, lag = 20, type = "Ljung-Box")
#ArchTest(garch_residuals, lags = 20)




# Étape 7 : Extraire la volatilité conditionnelle journalière
volatility_daily <- sigma(garch_fit)


chart_Series(Return$Volatility)
chart_Series(Return$ADBE.Adjusted)


# Exporter en CSV avec les dates et la volatilité
write.csv(data.frame(Date = index(Return), coredata(Return)), "volatilite_estimee.csv", row.names = FALSE)



## Etape 8: Forcasting avec FARIMA-GARCH


# Prévisions du modèle FARIMA
farima_forecast <- predict(best_model, n.ahead = 100)  # 10 prévisions

# Prévisions du modèle GARCH
garch_forecast <- sigma(ugarchforecast(garch_fit, n.ahead = 100))


# Combinaison des prévisions de FARIMA et de GARCH
predicted_Returns <- farima_forecast$pred
predicted_volatility <- ts(garch_forecast)

# Calcul des intervalles de confiance
lower_bound <- predicted_Returns - 1.96 * predicted_volatility
upper_bound <- predicted_Returns + 1.96 * predicted_volatility

# Affichage des résultats
# 5. Création d'un DataFrame pour afficher les résultats
forecast_results <- data.frame(
  PredictedReturns = predicted_Returns,
  LowerBound = lower_bound,
  UpperBound = upper_bound,
  PredictedVolatility = predicted_volatility
)
# Renommage des colonnes du dataframe forecast_results
colnames(forecast_results) <- c("PredictedReturns", "LowerBound", "UpperBound", "PredictedVolatility")

# Affichage des résultats avec les nouvelles colonnes
print(forecast_results)



# 6. Tracer les prévisions et les intervalles de confiance
library(ggplot2)

forecast_results$Date <- seq.Date(from = Sys.Date() + 1, by = "day", length.out = 100)  # Période de prévision (vous pouvez ajuster la date)

# Tracer les rendements prévisionnels et les intervalles de confiance
ggplot(forecast_results, aes(x = Date)) +
  geom_line(aes(y = PredictedReturns), color = "blue", size = 1) +
  geom_ribbon(aes(ymin = LowerBound, ymax = UpperBound), fill = "grey", alpha = 0.3) +
  labs(title = "Prévisions des Rendements avec Intervalles de Confiance", 
       x = "Date", 
       y = "Rendement Prévisionnel") +
  theme_minimal()

# Tracer la volatilité prévisionnelle
ggplot(forecast_results, aes(x = Date, y = PredictedVolatility)) +
  geom_line(color = "red", size = 1) +
  labs(title = "Prévisions de la Volatilité", 
       x = "Date", 
       y = "Volatilité Prévisionnelle") +
  theme_minimal()



