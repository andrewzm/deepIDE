## Plot the Objective values
load("./intermediates/Objective_series.rda")

## Plot the Costs
png("img/Objectives.png")
par(mfrow = c(2,2))

## Extract all Training and Validation costs to date
TrainCosts <- unlist(Objective)
ValCosts <- unlist(Objective_val)
ValCosts <- ValCosts[ValCosts > 0] # Remove the zero entries


plot(TrainCosts)
XX <- data.frame(idx = 1:length(TrainCosts), O = TrainCosts)
Osmooth <- predict(loess(O ~ idx, data = XX, span = 0.25))
plot(Osmooth)
plot(ValCosts)
XX <- data.frame(idx = 1:length(ValCosts), O = ValCosts)
Osmooth <- predict(loess(O~idx, data = XX, span = 0.25))
plot(Osmooth)
dev.off()
