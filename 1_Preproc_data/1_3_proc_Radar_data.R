####################################################################
## Title: Preprocess Radar Data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
## Description: Takes the Radar dataset as described in Wikle et al.
##              (2019) and regrids it on a 64 x 64 grid. Then puts
##              the data into the same format as the SST data
###################################################################
library("dplyr")
library("ggplot2")
library("gstat")
library("spacetime")
library("sp")
library("STRbook")

## Load the radar dataset from STRbook
data("radar_STIDF", package = "STRbook")

## Make time into hourly-minut format
radar_STIDF$timeHM <- format(time(radar_STIDF), "%H:%M")

## Make a spatial grid on which to regrid the data
## (Regridding onto 64 x 64)
s1range <- range(radar_df$s1)
s2range <- range(radar_df$s2)
newgrid <- expand.grid(s1 = seq(s1range[1], s1range[2], length.out = 64),
                       s2 = seq(s2range[1], s2range[2], length.out = 64))

## We now go through each data point, do the regridding, and add the
## results to a data-frame list and an array

## Initialise:
time_axis <- unique(radar_df$t)
radar_df2_list <- list()
radar_array <- array(0, dim = c(length(time_axis), 64, 64))
oldgrid <- filter(radar_df, t == time_axis[1])

## For each time point
for(i in seq_along(time_axis)) {
  
  ## Extract the data for this time point on original grid
  oldgrid$z <- filter(radar_df, t == time_axis[i])$z
  
  ## Do IDW to regrid onto 64 x 64 and rename columns
  radar_df2_list[[i]] <- idw(formula = z ~ 1,       # dep. variable
                             locations = ~ s1 + s2, # inputs
                             data = oldgrid,        # data set
                             newdata = newgrid,     # prediction grid
                             idp = 5) %>%           # inv. dist. pow.
    mutate(z = var1.pred,
           date = time_axis[i],
           t = i) %>%
    dplyr::select(-var1.pred, -var1.var)
  
  ## Also add to array, to match SST data
  radar_array[i,,] <- matrix(radar_df2_list[[i]]$z, 64, 64)
}

## Make a big data frame in long format by rbinding the dfs at each time pt
radar_df2 <- data.table::rbindlist(radar_df2_list)

## Visualise, to check the regridded output is reasonable
ggplot(radar_df2) + geom_tile(aes(s1, s2, fill = z)) + 
  facet_wrap(~t) + scale_fill_distiller(palette = "Spectral")

## Create a space-time grid where space is 64 x 64
newgrid_t <- expand.grid(s1 = seq(s1range[1], s1range[2], length.out = 64),
                         s2 = seq(s2range[1], s2range[2], length.out = 64),
                         t = seq_along(time_axis))

## Create and STIDF from this gridded data (reqd. for IDE)
radar_STIDF <- STIDF(sp = SpatialPoints(newgrid_t[,2:1]), 
                     time = rep(time_axis, each = 64^2), 
                     data = dplyr::select(radar_df2, z))
radar_STIDF$timeHM <- format(time(radar_STIDF), "%H:%M")

## Save both the array data and the STIDF data
save(radar_array, radar_STIDF, 
     file = paste0("intermediates/Radar_data.rda"))