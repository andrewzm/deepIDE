####################################################
## Title: FRK for the radar data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

## Load packages and data
library("FRK")
library("dplyr")
library("sp")
load("../1_Preproc_data/intermediates/Radar_data.rda")

## Set up obs. variables
mtot <- length(radar_STIDF)
valblock_idx <- which(radar_STIDF$timeHM %in% c("10:15"))
obs_idx <- setdiff(1:mtot, valblock_idx)
radar_obs <- radar_STIDF[obs_idx, ]
radar_valblock <- radar_STIDF[valblock_idx, ]

## Spatial basis functions
G_spatial <- auto_basis(manifold = plane(),     # fns on plane
                        data = radar_obs,       # project
                        nres = 2,               # 2 res.
                        type = "bisquare",      # bisquare.
                        regular = 1)            # irregular

## Temporal basis functions
t_grid <- matrix(seq(0, 12, length = 5))
G_temporal <- local_basis(manifold = real_line(), # fns on R1
                          type = "bisquare",      # bisquare
                          loc = t_grid,           # centroids
                          scale = rep(3.5, 5))    # aperture par.

## Spatio-temporal basis functions
G <- TensorP(G_spatial, G_temporal)  # take the tensor product

## Set up BAUs
BAUs <- auto_BAUs(manifold = STplane(),   # ST field on plane
                  type = "grid",          # gridded (not "hex")
                  data = radar_STIDF,
                  cellsize = c(1.65, 2.38, 10), # BAU cell size
                  nonconvex_hull = FALSE, # convex boundary
                  convex = 0,             # no hull extension
                  tunit = "mins")         # time unit is "mins"
BAUs$fs = 1       # fs variation prop. to 1

## Fix measurement error variance (see 5_3)
sigma2_eps <- 16.23585
radar_obs$std <- sqrt(sigma2_eps)

## Do FRK
S <- FRK(f = z ~ 1,
         BAUs = BAUs,
         data = list(radar_obs), # (list of) data
         basis = G,           # basis functions
         n_EM = 20,            # max. no. of EM iterations
         tol = 0.01)          # tol. on log-likelihood

## Predict
FRK_pred <- predict(S)
df_block_over <- over(radar_valblock, FRK_pred)

## Save results
save(df_block_over, file = "intermediates/FRK_Radar_results.rda")
                      
