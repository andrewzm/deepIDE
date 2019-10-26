#############################################
## Title: Analyse results from radar data set
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
#############################################

## Load libraries and source code
library("animation")
library("dplyr")
library("ggplot2")
library("gridExtra")
library("tidyr")
library("verification")
library("R.utils")
sourceDirectory("../common/") 

## Extract the means and sds of the radar data
load("../1_Preproc_data/intermediates/Radar_data.rda")
mean_radar <- apply(radar_array, 1, mean)
sd_radar <- apply(radar_array, 1, sd)

## Get the grid and mask from the CNNIDE results
load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Radar.rda"))
sgrid <- all_data$sgrid
taxis <- taxis_df$idx
maskidx <- which(sgrid$s1 > 0.1 & sgrid$s1 < 0.9 &  # we train on CNN only ..
                 sgrid$s2 > 0.1 & sgrid$s2 < 0.9) # inside this square box ..
mask <- rep(1, 64^2)                                # to avoid boundary effects
mask[-maskidx] <- 0
maskidx_orig <- maskidx

## Load the low-rank IDE results
load(paste0("../4_Analyse_Data_Other/intermediates/IDE_Radar_results.rda"))
lowrankIDE_Results <- as.data.frame(pred_IDE_block)

## Load the full-rank IDE results
load("../4_Analyse_Data_Other/intermediates/Results_fullrank_IDE_Radar.rda")
fullrankIDE_Results <- results_IDE[[12]]

## Put the results of the last 3 frames in a long data frame
xygrid <- expand.grid(s1 = 1:64, s2 = 1:64)
plot_df <- data.frame(z = c(radar_array[10,,],
                            radar_array[11,,],
                            radar_array[12,,],
                            results[[12]]$fcast_mu*sd_radar[12] + mean_radar[12],
                            fullrankIDE_Results$fcast_mu_IDE*sd_radar[12] + mean_radar[12],
                            lowrankIDE_Results$Ypred),
                      name = c(rep(" Observed (09:55)", 4096),
                               rep(" Observed (10:05)", 4096),
                               rep(" Observed (10:15)", 4096),
                               rep("CNN-IDE Nowcast (10:15)", 4096),
                               rep("Full-rank IDE Nowcast (10:15)", 4096),
                               rep("Low-rank IDE Nowcast (10:15)", 4096))) %>%
           cbind(xygrid)

## Plot these results
g <- ggplot(plot_df) + geom_tile(aes(s1, s2, fill = pmax(z, -10))) + 
  facet_wrap(~name) + theme_bw() +
  theme(text = element_text(size = 14)) +
  scale_fill_distiller(palette = "Spectral", name = "dBZ")
ggsave(g, file = "./img/radar.png", dpi = 300, width = 9, height = 6)

ggplot(dplyr::filter(plot_df, name %in% c(" Observed (10:15)", 
                                          "CNNIDE Nowcast (10:15)", 
                                          "Full-rank IDE Nowcast (10:15)"))) + 
         geom_contour(aes(s1, s2, z = pmax(z, 0), colour= name), bins = 5, size = 0.3) + theme_bw() +
         theme(text = element_text(size = 14)) +
      scale_fill_distiller(palette = "Spectral", name = "dBZ")

## Extract the measurement-error variance
sigma2eps <- var(filter(plot_df, s2 < 20 & s1 > 40 & name == " Observed (09:55)")$z)
print(sigma2eps)

## Extract results from the Radar Study
Radar_results <- rbind(
  (summarystats(results[[12]]$truth[maskidx]*sd_radar[12] + mean_radar[12],
              results[[12]]$fcast_mu[maskidx]*sd_radar[12] + mean_radar[12],
              sqrt((results[[12]]$fcast_sd[maskidx]*sd_radar[12])^2 + sigma2eps),
              name = "CNN-IDE",
              time = 12,
              zone = 1)),
(summarystats(results[[12]]$truth[maskidx]*sd_radar[12] + mean_radar[12],
              fullrankIDE_Results$fcast_mu_IDE[maskidx]*sd_radar[12] + mean_radar[12],
              sqrt((fullrankIDE_Results$fcast_sd_IDE[maskidx]*sd_radar[12])^2 + sigma2eps),
              name = "Full-rank IDE (window)",
              time = 12,
              zone = 1)),
(summarystats(results[[12]]$truth[maskidx]*sd_radar[12] + mean_radar[12],
              lowrankIDE_Results$Ypred[maskidx],
              sqrt(lowrankIDE_Results$Ypredse[maskidx]^2 + sigma2eps),
              name = "Low-rank IDE",
              time = 12,
               zone = 1)))

## Print results as a table
Radar_results %>% rename(Model = method) %>%
   dplyr::select(-time, -zone) %>%
   xtable::xtable() %>%
   print(include.rownames = FALSE)


