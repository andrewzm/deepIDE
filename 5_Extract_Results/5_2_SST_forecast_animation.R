library("animation")
library("dplyr")
library("ggplot2")
library("grid")
library("gridExtra")

plot_zone <- 1
source("../common/utils.R")
load(file = "../1_Preproc_data/intermediates/TrainingData3D.rda")
load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_", 
            plot_zone, ".rda"))
rm(d); gc()

zone1grid <- expand.grid(lon = seq(-70, -64.5, length.out = 64),
                         lat = seq(35, 40.25, length.out = 64))

SST_t <- function(i) {
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
  
  g1 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(pmax(mu, SSTmin), SSTmax))) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(SSTmin, SSTmax),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle("Filtered")
  
  g2 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(se, 1.0))) +
    scale_fill_distiller(palette = "BrBG",
                         limits = c(0, 1.0),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() +  ggtitle("Filtered s.e.")
  
  g3 <- ggplot(spatgrid) +
    geom_tile(aes(lon ,lat, fill = pmin(pmax(mu2, SSTmin), SSTmax))) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(SSTmin, SSTmax),
                         name = "degC") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle("Forecast (from previous day)")
  
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

gen_anim <- function() {
  #for(i in seq_along(taxis_df$t)){  # for each time point
  for(i in 1:5){
    if(i > 3)
      SST_t(i)            # plot data at this time point
  }
}

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
