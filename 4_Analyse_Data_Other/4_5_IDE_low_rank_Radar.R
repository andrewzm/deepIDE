####################################################
## Title: Low-rank IDE for the radar data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

## Load packages and data
library("IDE")
load("../1_Preproc_data/intermediates/Radar_data.rda")

## Set up observation variables
valblock_idx <- which(radar_STIDF$timeHM %in% c("10:15"))
obs_idx <- setdiff(1:mtot, valblock_idx)
radar_obs <- radar_STIDF[obs_idx, ]
radar_valblock <- radar_STIDF[valblock_idx, ]

## Construct IDE model
IDEmodel <- IDE(f = z ~ 1,
                data = radar_obs,
                dt = as.difftime(10, units = "mins"),
                grid_size = 41,
                forecast = 1)

## Fit the model
fit_results_radar2 <- fit.IDE(IDEmodel,
                              parallelType = 1)

## Predict using the fitted model
pred_IDE_block <- predict(fit_results_radar2$IDEmodel,
                          newdata = radar_valblock)


## Save results
save(fit_results_radar2, pred_IDE_block, file = "intermediates/IDE_Radar_results.rda")
