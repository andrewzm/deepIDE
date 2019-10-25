####################################################
## Title: Spatial-only kriging for SST data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

## Load libraries
library("gstat")
library("sp")

## Create intermediates directory
if(!(dir.exists("intermediates"))) dir.create("intermediates")

for(zone in 1:19L) {
  load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_", zone, ".rda"))
  taxis <- 3:nrow(taxis_df)    
  results_kriging <- list()
  ## Spatial-only kriging
  for(i in taxis) {
    Z <- all_data$Z[[i]]
    C <- all_data$C[[i]]
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
    
    results_kriging[[i]] <- data.frame(filter_mu_kriging = pred_krige$pred,
                                     filter_sd_kriging = as.numeric(pred_krige$var))
    cat(paste0("Kriging Zone ", zone, " Time point ", i, "\n"))
  }
  save(results_kriging, file = paste0("intermediates/Results_kriging_Zone_", zone, ".rda"))
}
