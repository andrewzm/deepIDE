####################################################################
## Title: SST Animation
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
## Description: Generates an animation of SST in the North Atlantic
##              from the SST data
###################################################################

## Load packages and utils.R
library("animation")
library("ggplot2")
library("lubridate")
library("ncdf4")
source("../common/utils.R")

## Load data and extract variables
sst <- ncdf4::nc_open("data/global-analysis-forecast-phy-001-024_1551608429013.nc")
sstvec <- ncdf4::ncvar_get(sst, "thetao")
lon <- ncvar_get(sst, "longitude")
lat <- ncvar_get(sst, "latitude")
time <- ncvar_get(sst, "time") # Hours since 1950-01-01
startdate <- as.Date("1950-01-01")
dates <- startdate + days(floor(time/24))
dates <- startdate + hours(time)

## Define lon-lat grid
spatgrid <- expand.grid(lon = lon,
                        lat = lat)

## Create a function that takes an index t and returns a spatial image
## of the SST in the North Atlantic
SST_t <- function(t) {
    spatgrid$thetao <- c(sstvec[,,t])
    t2 <- dates[t] # extract date
    SSTmin <- -4   # min colour
    SSTmax <- 27   # max colour

    ## Generate plot
    g <- ggplot(spatgrid) +
        geom_tile(aes(lon,lat,fill = pmin(pmax(thetao, SSTmin), SSTmax))) +
        scale_fill_gradientn(colours = nasa_palette,
                             limits = c(SSTmin, SSTmax),
                             name = "degC") +
        theme_bw() +
        xlab("Longitude (deg)") +
        ylab("Latitude (deg)") +
        coord_fixed() +
        geom_text(data = data.frame(x = -30, y = 58, label = t2),
                  aes(x,y,label = label))
}
 
## Function that plots the SST data in a sequence of images
gen_anim <- function() {
    for(t in seq_along(time)){  # for each time point
       plot(SST_t(t))           # plot data at this time point
    }
}

## Generate animation
setwd("./anim")
ani.options(interval = 0.05)    # 0.2s interval between frames
saveHTML(gen_anim(),            # run the main function
         autoplay = FALSE,      # do not play on load
         loop = FALSE,          # do not loop
         verbose = FALSE,       # no verbose
         outdir = "anim",          # save to current dir
         single.opts = "'controls': ['first', 'previous',
                                     'play', 'next', 'last',
                                      'loop', 'speed'],
                                      'delayMin': 0",
         htmlfile = "SST_anim.html")  # save filename
setwd("..")