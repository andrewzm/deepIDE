####################################################################
## Title: SST Zone Extraction
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
## Description: Extracts SST data into 19 zones of size 64 x 64
####################################################################

## Load packages and utils.R
library("dplyr")
library("ggplot2")
library("lubridate")
library("ncdf4")
library("reticulate")
source("../common/utils.R")

## Load data and extract variables
sst <- ncdf4::nc_open("data/global-analysis-forecast-phy-001-024_1551608429013.nc")
sstvec <- ncdf4::ncvar_get(sst, "thetao")
lon <- ncvar_get(sst, "longitude")
lat <- ncvar_get(sst, "latitude")
dlon <- mean(diff(lon))
dlat <- mean(diff(lat))
time <- ncvar_get(sst, "time") # Hours since 1950-01-01
startdate <- as.Date("1950-01-01")
dates <- startdate + days(floor(time/24))
dates <- startdate + hours(time)

## Define lon-lat grid
spatgrid <- expand.grid(lon = lon,
                        lat = lat)

## Get a latitude strip on which we define our zones
spatgrid$id <- 1:nrow(spatgrid)
spatgridsub <- filter(spatgrid, lat >= 35 & lat <= 53)

## Create a coarse grid of size 64 x 64 overlaying the strip we just extracted
cellwidth <- dlon*64L
eps <- dlon/2
breaks_lon <- seq(min(spatgridsub$lon) - eps, 
                  max(spatgridsub$lon), 
                  by = cellwidth)
breaks_lat <- seq(min(spatgridsub$lat) - eps, 
                  max(spatgridsub$lat), 
                  by = cellwidth)
spatgridsub$lonbox <- cut(spatgridsub$lon, breaks_lon, labels = FALSE)
spatgridsub$latbox <- cut(spatgridsub$lat, breaks_lat, labels = FALSE)
idx <- spatgridsub$id # id of pixel
spatgridsub$thetao <- as.numeric(sstvec[,,3])[idx]


## Show a plot of SST in one zone
spatgridsub$thetao <- as.numeric(scale(spatgridsub$thetao))
myscale <- scale_fill_gradientn(colours = nasa_palette, name = "degC", na.value="white")
ggplot(filter(spatgridsub, lonbox == 6 & latbox == 3)) + 
  geom_tile(aes(lon, lat, fill = thetao)) + 
  coord_fixed() + myscale

## Find which zones contain land, or which boxes are not 64x64 (at edges) and remove them
## New grid will be in spatgridsub2
NAboxes <- which(is.na(sstvec[,,1][idx]))
rmtable <- group_by(spatgridsub, lonbox, latbox) %>%
           summarise(rm = (length(thetao) < 64^2 | any(is.na(thetao))))
rmtable$zone <- 1:nrow(rmtable)
spatgridsub2 <- left_join(spatgridsub, rmtable) %>%
                filter(!rm)
zones <- unique(spatgridsub2$zone)
nZones <- length(zones)
idx <- spatgridsub2$id   # update idx

## Find the zone boundaries
zonebounds <- spatgridsub2 %>%    # Find zone boundaries
    group_by(zone) %>%
    summarise(minlon = min(lon),
              minlat = min(lat),
              maxlon = max(lon),
              maxlat = max(lat))

## Make a big polygon data frame contain zone polygons from the bounds
zonebounds2 <- NULL 
for(i in 1:nZones) {
    zonebounds2 <- rbind(zonebounds2,
                         data.frame(zone = i,
                                    lon = c(zonebounds[i,]$minlon,
                                            zonebounds[i,]$minlon,
                                            zonebounds[i,]$maxlon,
                                            zonebounds[i,]$maxlon,
                                            zonebounds[i,]$minlon),
                                    lat = c(zonebounds[i,]$minlat,
                                            zonebounds[i,]$maxlat,
                                            zonebounds[i,]$maxlat,
                                            zonebounds[i,]$minlat,
                                            zonebounds[i,]$minlat)))
    }

## Plot the SST data within the zones
world <- map_data(c("world"))  # world map
spatgrid$sst <- c(sstvec[,,1]) # SST everywhere (including outside zones)
gzones <- ggplot(filter(spatgrid, id %in% spatgridsub2$id)) +
    geom_tile(aes(lon, lat, fill = sst)) +
    scale_fill_gradientn(colours = nasa_palette, name = "degC",
                         na.value = "white") +
    geom_path(data = zonebounds2, aes(lon, lat, group = zone)) +
    geom_path(data = world, aes(long, lat, group = group), colour = "dark gray") +
    coord_fixed(xlim = c(-75, -17), ylim = c(30, 56)) + theme_bw() +
    xlab("Longitude (deg)") + ylab("Latitude (deg)") +
    geom_text(data = zonebounds, aes(maxlon - 1, maxlat-1, label = seq_along(zone)), 
              colour = "black", size = 3, fontface = "bold")
ggsave(gzones, file = "img/Zones.png",
       height = 3, width = 7.5)

## Now, find the means and standard deviations in each zone for standarising
## First, we take the SST vector and put it into an npixels x T matrix
sst_in_zones <- reticulate::array_reshape(sstvec, c(prod(dim(sstvec)[1:2]),
                                                         dim(sstvec)[3]),
                                                    order = "F")[idx,]

## Add on the lon-lat info to this npixels x T matrix and then put into long format
sst_in_zones_df <- cbind(select(spatgridsub2, lon, lat, zone), sst_in_zones) %>%
                   tidyr::gather(time, sst, -lon, -lat, -zone)

## We now group by zone and time and find the means and sds for each zone/time combination
means_df <- mutate(sst_in_zones_df, time = as.integer(time)) %>%
  group_by(zone, time) %>%
  arrange(zone, time) %>%
  summarise(meansst = mean(sst),
            sdsst = sd(sst))

## Define a helper function, which just takes the SST in a zone and standardises it using
## the mean and standard deviation of the data in that zone
scale_sst <- function(thetao, z, t) {
  mu <- filter(means_df, zone == z & time == t)$meansst
  sd <- filter(means_df, zone == z & time == t)$sdsst
  (thetao - mu)/sd
}

## We are now ready to produce the data sets for training
## We will have three data sets:
## d: array containing sequences (Y_{t}, Y_{t+1}, Y_{t+2}) where each Y_i has size 64 x 64
## dfinal: array containing (Y_{t+2}) of size 64 x 64
## dpred: array containing (Y_{t+3}) of size 64 x 64
## These attays are constructed so that they aligned time-wise
## (E.g., the 5th entry of d contains (Y_5, Y_6, Y_7). that of dfinal contains (Y_7)
## and that of dpred contains (Y_8) 

## Initialise
nT <- (length(time) - 3)
d <- array(0, dim = c(nT * length(zones), 64L, 64L, 3L))
dfinal <- array(0, dim = c(nT * length(zones), 64L, 64L))
dpred <- array(0, dim = c(nT * length(zones), 64L, 64L))
means <- sds <- NULL

## For each zone and time point
for(i in seq_along(zones)) {
  cat(paste0("Processing Zone ", i, "\n"))
  for(j in seq_along(time[1:nT])) {

    ## Calculate the index for this zone/time point  
    thisidx <- (i - 1)*nT + j

    ## Add the SST to our grid corresponding to this (jth) time point and scale
    spatgridsub2$thetao <- c(sstvec[,,j])[idx]
    im1 <- filter(spatgridsub2, zone == zones[i])$thetao %>%
           scale_sst(zones[i], j)

    ## Do the same for the (j+1)th time point
    spatgridsub2$thetao <- c(sstvec[,,j + 1])[idx]
    im2 <- filter(spatgridsub2, zone == zones[i])$thetao %>%
      scale_sst(zones[i], j + 1)

    ## Do the same for the (j+2)th time point
    spatgridsub2$thetao <- c(sstvec[,,j + 2])[idx]
    im3 <- filter(spatgridsub2, zone == zones[i])$thetao %>%
      scale_sst(zones[i], j + 2)

    ## Do the same for the (j+3)th time point
    spatgridsub2$thetao <- c(sstvec[,,j + 3])[idx]
    im4 <- filter(spatgridsub2, zone == zones[i])$thetao %>%
      scale_sst(zones[i], j + 3)

    ## Add to d, dfinal, and dpred, as discussed above
    d[thisidx,,,1] <- im1
    d[thisidx,,,2] <- im2
    d[thisidx,,,3] <- im3
    dfinal[thisidx,,] <- im3
    dpred[thisidx,,] <- im4
  }
}

## Save data to disk
spatgrid <- select(spatgridsub2, lon, lat, zone)
save(spatgrid, file = "intermediates/TrainingDataLocs.rda")
save(means_df, d, file = "intermediates/TrainingData3D.rda")
save(dfinal, file = "intermediates/TrainingDataFinals.rda")
save(dpred, file = "intermediates/TrainingDataPreds.rda")



