library(gstat)
library(sp)

load("Results_CNNIDE_Radar.rda")

## Spatial-only kriging
Z <- all_data$Z[[11]]
C <- all_data$C[[11]]
sobs <- as.data.frame(as.matrix(C) %*% as.matrix(all_data$sgrid))
sobs$Z <- Z
coordinates(sobs) <- ~s1 + s2
vv <- variogram(object = Z ~ s1 + s2,
                data = sobs,
                cutoff = 0.7)
vgm_model <- vgm(psill = 0.4, model = "Exp", range = 0.1, Err = sigma2e)
vgm_fit <- fit.variogram(vv, vgm_model, fit.sills = c(FALSE, TRUE))
plot(vv, vgm_fit)

sgrid <- all_data$sgrid
coordinates(sgrid) <- ~s1 + s2
gridded(sgrid) <- TRUE

pred_krige <- krige0(Z ~ s1 + s2,
                     locations = ~s1 + s2,
                     data = sobs,
                     newdata = sgrid,
                     model = vgm_fit,
                     computeVar = TRUE)

results_kriging <- list()
results_kriging[[11]] <- data.frame(filter_mu_kriging = pred_krige$pred,
                                   filter_sd_kriging = as.numeric(pred_krige$var))
save(results_kriging, file = paste0("../Radar_data/Results_kriging_Radar.rda"))