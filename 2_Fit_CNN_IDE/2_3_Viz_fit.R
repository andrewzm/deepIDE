####################################################
## Title: Plot the squared prediction errors on the
##        training the data and the validation data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

## Plot the Objective values
load("./intermediates/Objective_series.rda")

## Extract all Training and Validation costs to date
TrainCosts <- unlist(Objective)
ValCosts <- unlist(Objective_val)
ValCosts <- ValCosts[ValCosts > 0] # Remove the zero entries

## Plot the Costs
png("img/Objectives.png")
par(mfrow = c(2,2))
plot(TrainCosts)
XX <- data.frame(idx = 1:length(TrainCosts), O = TrainCosts)
Osmooth <- predict(loess(O ~ idx, data = XX, span = 0.25))
plot(Osmooth)
plot(ValCosts)
XX <- data.frame(idx = 1:length(ValCosts), O = ValCosts)
Osmooth <- predict(loess(O~idx, data = XX, span = 0.25))
plot(Osmooth)
dev.off()
