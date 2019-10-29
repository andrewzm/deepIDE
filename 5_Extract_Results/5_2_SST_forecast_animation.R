####################################################
## Title: Do animation of the prediction / forecasts
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

## Load librarties
library("animation")
library("dplyr")
library("ggplot2")
library("grid")
library("gridExtra")
library("R.utils")
sourceDirectory("../common/")

## Plot in zone 1
plot_zone <- 1
load(file = "../1_Preproc_data/intermediates/TrainingData3D.rda")
load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_", 
            plot_zone, ".rda"))
rm(d); gc()

## 64 x 64 grid on Zone 1
zone1grid <- expand.grid(lon = seq(-70, -64.5, length.out = W),
                         lat = seq(35, 40.25, length.out = H))

## Function to print one image
SST_t <- function(i) {

  ## Unnormalise  
  idx <- filter(all_data$taxis_df, zone == plot_zone)$idx[i]
  this_mean <- means_df$meansst[idx]
  this_sd <- means_df$sdsst[idx]
  spatgrid <- zone1grid
  spatgrid$mu <- results[[i]]$filter_mu * this_sd + this_mean
  spatgrid$se <- results[[i]]$filter_sd * this_sd
  spatgrid$mu2 <- results[[i]]$fcast_mu * this_sd + this_mean
  spatgrid$se2 <- results[[i]]$fcast_sd * this_sd
  t2 <- results[[i]]$date[1]
  SSTmin <- 10
  SSTmax <- 27

  ## Plot filtered SST
  g1 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(pmax(mu, SSTmin), SSTmax))) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(SSTmin, SSTmax),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle("Filtered")

  ## Plot filtered standard errors
  g2 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(se, 1.0))) +
    scale_fill_distiller(palette = "BrBG",
                         limits = c(0, 1.0),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() +  ggtitle("Filtered s.e.")

  ## Plot forecast
  g3 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(pmax(mu2, SSTmin), SSTmax))) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(SSTmin, SSTmax),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle("Forecast (from previous day)")

  ## Plot forecast standard errors
  g4 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(se2, 1.0))) +
    scale_fill_distiller(palette = "BrBG",
                         limits = c(0, 1.0),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() +  ggtitle("Forecast s.e.")
  
  grid.arrange(g1, g2, g3, g4, nrow = 2,
               top = textGrob(paste0("Date: ", t2) ,
                              gp = gpar(fontsize=20)))
}

## Run through each time point and generate plots
gen_anim <- function() {
  for(i in seq_along(taxis_df$t)){  # for each time point
   if(i > 3)
      SST_t(i)            # plot data at this time point
  }
}

## Generate animation
setwd("./anim")
ani.options(interval = 0.05)    # 0.2s interval between frames
saveHTML(gen_anim(),            # run the main function
         autoplay = FALSE,      # do not play on load
         loop = FALSE,          # do not loop
         verbose = FALSE,       # no verbose
         outdir = ".",          # save to current dir
         single.opts = "'controls': ['first', 'previous',
                                     'play', 'next', 'last',
                                      'loop', 'speed'],
                                      'delayMin': 0",
         htmlfile = "SST_filter.html")  # save filename
setwd("..")

ggsave(SST_t(107), file = "./img/VideoFrame107.png", width = 7, height = 7)
