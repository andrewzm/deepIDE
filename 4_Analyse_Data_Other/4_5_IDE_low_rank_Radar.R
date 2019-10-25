library("IDE")

load("../1_Preproc_data/intermediates/Radar_data.rda")
valblock_idx <- which(radar_STIDF$timeHM %in% c("10:15"))
obs_idx <- setdiff(1:mtot, valblock_idx)
radar_obs <- radar_STIDF[obs_idx, ]
radar_valblock <- radar_STIDF[valblock_idx, ]

IDEmodel <- IDE(f = z ~ 1,
                data = radar_obs,
                dt = as.difftime(10, units = "mins"),
                grid_size = 41,
                forecast = 1)

fit_results_radar2 <- fit.IDE(IDEmodel,
                              parallelType = 1)

pred_IDE_block <- predict(fit_results_radar2$IDEmodel,
                          newdata = radar_valblock)

save(fit_results_radar2, pred_IDE_block, file = "intermediates/IDE_Radar_results.rda")