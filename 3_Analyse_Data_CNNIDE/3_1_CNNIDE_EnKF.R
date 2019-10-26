##########################################################
## Title: Use EnKF with CNNIDE to do predictions/forecasts
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
#########################################################

## Load packages and source files
library("tensorflow")
library("dplyr")
library("fields")
library("ggplot2")
library("gridExtra")
library("gstat")
library("Matrix")
library("lubridate")
library("R.utils")
sourceDirectory("../common") 

## Create intermediates directory
if(!(dir.exists("intermediates"))) dir.create("intermediates")

            
## Analysing radar data or SST data?
radar_data <- FALSE

## Construct the graph
CNNgraph <- createCNNgraph(W = W,  H = H,
                           N_Channels = N_Channels,
                           N_Filters = N_Filters,
                           sqrtN_Basis = sqrtN_Basis, 
                           patch_size_flow = patch_size_flow,
                           patch_size_diff = patch_size_diff,
                           border_mask = border_mask)
list2env(CNNgraph, envir = .GlobalEnv)


## Load weights from Stage 1 and assign to graph
load(file = "../2_Fit_CNN_IDE/intermediates/SSTIDE_weights_TF.rda")
run <- tf$Session()$run
All_Vars_tf <- tf$trainable_variables(scope = NULL)
for(i in 1:length(All_Vars_tf)) {
  run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
}

## Load covariance function parameters from Stage 2
load("../2_Fit_CNN_IDE/intermediates/SST_cov_pars.rda")

## Load radar or SST data
if(radar_data) {
  load("../1_Preproc_data/intermediates/Radar_data.rda")
  nT <- dim(radar_array)[1]
  mean_radar <- apply(radar_array, 1, mean)
  sd_radar <- apply(radar_array, 1, sd)
  for(i in 1:nT) {
    radar_array[i,,] <- (radar_array[i,,] - mean_radar[i]) / sd_radar[i]    
  }
  nZones <- 1
  dfinal <- radar_array
} else {
  load("../1_Preproc_data/intermediates/TrainingDataFinals.rda")
}

### Initialise
if(radar_data) {
  nObs <- 4096L 
  nParticles <- 32L               # 32 particles
  sigma2e <- (sqrt(16.23585) / sd_radar[1])^2 # estimated in 5_3
} else {
  nObs <- 1024L        
  nParticles <- 64L               # 64 particles
  sigma2e <- 0.01                 # measurement-error variance
}

two_step_fcasts <- FALSE
taper_l <- 0.2

## Create spatial grid
s1 <- seq(0, 1, length.out = W)  # grid points for s1
s2 <- seq(0, 1, length.out = H)  # grid points for s2
sgrid <- expand.grid(s1 = s1, s2 = s2)  # spatial grid in long format

### Set up  EnKF graph for doing EnKF on GPU

## Compute cholesky factor of Sigma
L_full <- tf$cholesky(SIGMA)  # Lower Cholesky
L_tiled <- tf$expand_dims(L_full, 0L)
L_tiled <- tf$tile(L_tiled, c(thisN_Batch, 1L, 1L))

## Propagate the ensemble forward one step
normvars_process <- tf$placeholder(dtype = "float32", shape = c(nParticles, W*H, 1L))
Ypred_noisy <- Ypred + tf$linalg$matmul(L_tiled, normvars_process)

## Compute the forecast ensemble mean and covariance
## Note: Tapering is REALLY important otherwise we get rubbish results
Ypred_mean <- tf$reduce_mean(Ypred_noisy, axis = 0L, keepdims = FALSE)
Ypred_COV_untapered <- tf_cov(tf$squeeze(Ypred_noisy, 2L), thisN_Batch = thisN_Batch)
Taper <- tfconst(Wendland1_R(taper_l, run(D)))
Ypred_COV_uninflated <- tf$multiply(Ypred_COV_untapered, Taper)
Ypred_COV <- Ypred_COV_uninflated +
  0.1*tf$diag(tf$diag_part(Ypred_COV_uninflated))   

## Update the ensemble
C_tf <- tf$placeholder(dtype = "float32", shape = c(nObs, W*H))  # tile the observations
C_tfT <- tf$transpose(C_tf)    # transpose of observation matrix
Temp1 <- tf$matmul(Ypred_COV, tf$transpose(C_tf))
Temp2 <- tf$matmul(tf$matmul(C_tf, Ypred_COV), C_tfT) + sigma2e*tf$eye(nObs)
K_tf <- tf$matmul(Temp1, tf$matrix_inverse(Temp2))  # Kalman gain
K_tf_tiled <- tf$expand_dims(K_tf, 0L)              # Replicated for each member
K_tf_tiled <- tf$tile(K_tf_tiled, c(thisN_Batch, 1L, 1L))
C_tf_tiled <- tf$expand_dims(C_tf, 0L)              # Replicate observation matrix
C_tf_tiled <- tf$tile(C_tf_tiled, c(thisN_Batch, 1L, 1L))

normvars_obs1 <- normvars_obs2 <- tf$placeholder(dtype = "float32", shape = c(nParticles, nObs, 1L))
Z_tf <- tf$placeholder(dtype = "float32", shape = list(nObs, 1L))  # tile the observations
Z_tf_tiled  <- tf$expand_dims(Z_tf, 0L)
Z_tf_tiled  <- tf$tile(Z_tf_tiled, c(thisN_Batch, 1L, 1L))
Zsim_tf <- tf$linalg$matmul(C_tf_tiled, Ypred_noisy) +  # simulate the observations
  sqrt(sigma2e)*normvars_obs2
Ypred_updated <- Ypred_noisy + tf$linalg$matmul(K_tf_tiled, Z_tf_tiled - Zsim_tf)  # update

## Smooth1
Ysmooth_COV_untapered1 <- tf_cov(tf$squeeze(data_current_long, 2L),
                                 tf$squeeze(Ypred_noisy, 2L),
                                 thisN_Batch = thisN_Batch)
Ysmooth_COV1 <- tf$multiply(Ysmooth_COV_untapered1, Taper)
Temp1smooth1 <- tf$matmul(Ysmooth_COV1, tf$transpose(C_tf))
Temp2smooth1 <- tf$matmul(tf$matmul(C_tf, Ypred_COV), C_tfT) + sigma2e*tf$eye(nObs)
Ksmooth_tf1 <- tf$matmul(Temp1smooth1, tf$matrix_inverse(Temp2smooth1))  # Kalman gain
Ksmooth_tf_tiled1 <- tf$expand_dims(Ksmooth_tf1, 0L)              # Replicated for each member
Ksmooth_tf_tiled1 <- tf$tile(Ksmooth_tf_tiled1, c(thisN_Batch, 1L, 1L))
Ysmooth1 <- data_current_long + tf$linalg$matmul(Ksmooth_tf_tiled1, Z_tf_tiled - Zsim_tf)  # update

## Smooth2
data_previous <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))
data_previous_long <- reshape_3d_to_2d(data_previous, W*H)
Ysmooth_COV_untapered2 <- tf_cov(tf$squeeze(data_previous_long, 2L),
                                 tf$squeeze(Ypred_noisy, 2L),
                                 thisN_Batch = thisN_Batch)
Ysmooth_COV2 <- tf$multiply(Ysmooth_COV_untapered2, Taper)
Temp1smooth2 <- tf$matmul(Ysmooth_COV2, tf$transpose(C_tf))
Temp2smooth2 <- tf$matmul(tf$matmul(C_tf, Ypred_COV), C_tfT) + sigma2e*tf$eye(nObs)
Ksmooth_tf2 <- tf$matmul(Temp1smooth2, tf$matrix_inverse(Temp2smooth2))  # Kalman gain
Ksmooth_tf_tiled2 <- tf$expand_dims(Ksmooth_tf2, 0L)              # Replicated for each member
Ksmooth_tf_tiled2 <- tf$tile(Ksmooth_tf_tiled2, c(thisN_Batch, 1L, 1L))
Ysmooth2 <- data_previous_long + tf$linalg$matmul(Ksmooth_tf_tiled2, Z_tf_tiled - Zsim_tf)  # update

## Standard Kalman filter assuming known propagator matrix
## Just for checking...
Imat <- tf$eye(W*H)
SIGMAobs <- tf$diag(rep(sigma2e, nObs))
Yinit <- tf$placeholder(dtype = "float32", shape = list(W*H, 1L), name = "Yinit")
Covinit <- tf$placeholder(dtype = "float32", shape = list(W*H, W*H), name = "Covinit")

K_IDE0 <- K_IDE[1,,]
Covforecast <- tf$matmul(tf$matmul(K_IDE0, Covinit), tf$transpose(K_IDE0)) +  SIGMA
Ytilde <- Z_tf - tf$matmul(C_tf, tf$matmul(K_IDE0, Yinit))
S <- tf$matmul(tf$matmul(C_tf, Covforecast), tf$transpose(C_tf)) + SIGMAobs
Sinv <- tf$matrix_inverse(S)
Kalman <- tf$matmul(tf$matmul(Covforecast, tf$transpose(C_tf)), Sinv)
Yfilter <- tf$matmul(K_IDE0, Yinit) + tf$matmul(Kalman, Ytilde)
Covfilter <- tf$matmul((Imat - tf$matmul(Kalman, C_tf)), Covforecast)

## For each zone
for(zone in 1L:nZones) {
  
  if(!radar_data) {
        
    ## If SST data, then load the zone-specific covariances 
    run(tf$assign(log_sigma2, log_sigma2_zone_all[[zone]]))
    run(tf$assign(log_rho, log_rho_zone_all[[zone]]))

    set.seed(zone)                  # set seed
    
    ## Set time axis
    test_zones_dates <- filter(Date_Zone_Map,
                               year(startdate) == "2018" &
                                 month(startdate) %in% 9:12)
    this_zone <- zone
    taxis_df <- filter(test_zones_dates, zone == this_zone)
    taxis <- taxis_df$idx
  } else {
    taxis <- 1:nT
    taxis_df <- tibble(idx = 1:nT,
                       t = idx,
                       zone = 1,
                       currentdate = unique(time(radar_STIDF)))
  }

  ## Initialise data vectors and incidence matrices
  C <- z_df_list <- Z <- list()   # list of data and obs. matrices
  results <- list()               # list of results
  
  ## Simulate data with measurement error (if SST, otherwise just return data if radar data)
  ## Store data in long format (Z) and data frame (z_df_list)
  for(i in seq_along(taxis)) {
    obsidx <- sample(1:(W*H), nObs) 
    C[[i]] <- sparseMatrix(i = 1:nObs,   # measurement-mapping matrix
                           j = obsidx, 
                           x = 1, 
                           dims = c(nObs, W*H))
    Z[[i]] <- as.numeric(C[[i]] %*% c(dfinal[taxis[i],,]) + 
                           (!radar_data)*rnorm(nObs, 0, sqrt(sigma2e)))  ## add noise if not radar data
    z_df_list[[i]] <- cbind(sgrid[obsidx,], z = Z[[i]], t = i)
  }
  
  ## Collapse z_df_list into one long data frame
  z_df <- Reduce("rbind", z_df_list)
  ## ggplot(z_df) + geom_point(aes(s1, s2, colour = z)) + facet_wrap(~t) +
  ##    scale_colour_gradientn(colours = nasa_palette)
  
  
  ## Initialise the first particles and run the EnkF for each zone in test set
  
  ## Pframe contains the current `set' of three data points needed for state
  Pframe <- array(0, dim = c(nParticles, W, H, tau))
  SCOV <- run(SIGMA)          # extract cov. matrix
  cholCOV <- t(chol(SCOV))    # find lower Cholesky factor
  for(i in 1:tau) {           # for each image 
    mean_i <- idw(formula = z ~ 1,       # do IDW using the data
                  locations = ~ s1 + s2,  
                  data = z_df_list[[i]],
                  newdata = sgrid,
                  idp = 10)$var1.pred %>% 
      matrix(nrow = W)
    
    ## For each ensemble member take the IDW and add `prior' disturbance
    for(j in 1:nParticles) {
      Pframe[j,,,i] <- mean_i + matrix(cholCOV %*% matrix(rnorm(W * H),
                                                          ncol = 1L), W, H)
    }
  }
  Pframe2 <- Pframe
  Yfilter_prev_step <- matrix(apply(Pframe[,,,tau,drop=FALSE], 2:4, mean), W*H, 1)
  Covfilter_prev_step <- SCOV
  
  ## Now run the EnKF on the simulated data
  ## For each time point
  for(i in seq_along(taxis)) {
    
    ## The following code indicates to us that assuming the variances are
    ## are temporally invariant is detrimental to prediction. If we
    ## use those estimated from the full-rank IDE, predictions improve
    ## considerably
      
    ## warning("Using saved var values from IDE run") ## Improves Zone 6
    ## load(paste0("Results/IDEcovparsZone", zone, ".rda"))
    ## run(tf$assign(log_sigma2, log_sigma2_IDE[[zone]][i]))
    ## run(tf$assign(log_rho, log_rho_IDE[[zone]][i]))
    
    cat(paste0("Filtering Zone ", zone, " Time point ", i, "\n"))
    
    ## Ignore the first three time points (init. condition)
    if(i > tau) {
      
      ## Since the system is nonlinera, run the EnKF a couple of times,
      ## on each sequence to help improve the inferred dynamics
      for(kk in 1:2) {

        ## Update using the current Pframe
        ## Note that "tau" corresponds to "t - 1", "tau - 1" to "t - 2", etc.
        fd <- dict(data_in = Pframe,
                   data_current = Pframe[,,,tau],
                   data_previous = Pframe[,,,tau - 1],
                   C_tf = as.matrix(C[[i]]),
                   Z_tf = as.matrix(Z[[i]]),
                   normvars_process = array(rnorm(W*H*nParticles), dim = c(nParticles, W*H, 1L)),
                   normvars_obs1 = array(rnorm(nObs*nParticles), dim = c(nParticles, nObs, 1L)),
                   normvars_obs2 = array(rnorm(nObs*nParticles), dim = c(nParticles, nObs, 1L)))
        
        ## Extract forecast and updated ensemble
        ParticleUpdates <- run(list(Ypred_noisy, Ypred_updated, Ysmooth1, Ysmooth2),
                               feed_dict = fd)
        
        ## Do not use smoothed data when forecasting just do it for kk == 1
        if(kk == 1) ParticleForecasts <- ParticleUpdates[[1]]
        ParticlePreds <- ParticleUpdates[[2]]
        ParticleSmooth1 <- ParticleUpdates[[3]]
        ParticleSmooth2 <- ParticleUpdates[[4]]

        Pframe[,,,tau] <-  array(c(ParticleSmooth1), c(nParticles, W, H, 1))
        Pframe[,,,tau - 1] <-  array(c(ParticleSmooth2), c(nParticles, W, H, 1))
      }
      
      ## Redo with Pframe2 for two-day ahead forecast (same time point, just with un-updated ensemble at t-1)
      if(two_step_fcasts) {
        fd2 <- dict(data_in = Pframe2,
                    data_current = Pframe2[,,,tau],
                    normvars_process = array(rnorm(W*H*nParticles), dim = c(nParticles, W*H, 1L)))
        ParticleForecasts2 <- run(Ypred_noisy, feed_dict = fd2)
      }
      
      ## Shift the particles backwards to produce a new `Pframe'
      ## Don't really need to do this as updating smoothed components below
      for(j in 1:(tau - 1)) {
        Pframe[,,,j] <- Pframe[,,,j + 1]
        if(two_step_fcasts) Pframe2[,,,j] <- Pframe2[,,,j + 1]    
      }
      
      ## New particles (filtered ones) at tau
      Pframe[,,,tau] <-  array(c(ParticlePreds),
                               c(nParticles, W, H, 1))
      if(two_step_fcasts) Pframe2[,,,tau] <-  array(c(ParticleForecasts),
                                                    c(nParticles, W, H, 1))
      
      ## Smoothed particles at tau - 1
      if(two_step_fcasts) Pframe2[,,,tau - 1] <- Pframe[,,,tau - 1] # filtered
      Pframe[,,,tau - 1] <-  array(c(ParticleSmooth1),              # smoothed
                                   c(nParticles, W, H, 1))
      
      ## Smoothed particles at tau - 2
      if(two_step_fcasts) Pframe2[,,,tau - 2] <- Pframe[,,,tau - 2] # smoothed1
      Pframe[,,,tau - 2] <-  array(c(ParticleSmooth2),              # smoothed2
                                   c(nParticles, W, H, 1))
      
      ## Collect results into results data frame
      results[[i]] <-
        data.frame(filter_mu = apply(ParticlePreds, 2, mean),
                   filter_sd = apply(ParticlePreds, 2, sd),
                   fcast_mu = apply(ParticleForecasts, 2, mean),
                   fcast_sd = apply(ParticleForecasts, 2, sd),
                   truth = as.numeric(dfinal[taxis[i],,]),
                   zone = zone,
                   date = taxis_df$currentdate[i])
      
      if (two_step_fcasts) {
        results[[i]]$fcast_mu2 = apply(ParticleForecasts2, 2, mean)
        results[[i]]$fcast_sd2 = apply(ParticleForecasts2, 2, mean)
      }
      
      ## Save wind directions for a specific zone
      if(zone == 11 & i %in% 10:12)
        save(Pframe, ParticlePreds, ParticleForecasts,
             nParticles,
             file = paste0("intermediates/data_for_dir_plots_Zone11_i",
                           i, ".rda"))

      ## Save the kernel parameters at the first time point for use as initial conditions
      ## with the full rank IDE  
      if(i == (tau + 1)) {
        u_pars <- run(flow_coef, feed_dict = fd)[, 1:sqrtN_Basis^2, ] %>% apply(2, mean)
        v_pars <- run(flow_coef, feed_dict = fd)[, -(1:sqrtN_Basis^2), ] %>% apply(2, mean)
        Dpars <- run(D_coef, feed_dict = fd) %>% apply(2, mean)
        if(!radar_data) {
          kernel_fname <- paste0("intermediates/kinit_zone", zone, ".rda")
        } else {
          kernel_fname <- "intermediates/kinit_Radar.rda"
        }
        save(u_pars, v_pars, Dpars, file = kernel_fname)
      }
    }
  }

  ## Save the data and the results (data is also used by the other algorithms)
  all_data <- list(Z = Z, C = C, sgrid = sgrid, zone = zone, taxis_df = taxis_df)
  if(!radar_data) {
    save_fname = paste0("intermediates/Results_CNNIDE_Zone_", zone, ".rda")
  } else {
    save_fname = paste0("intermediates/Results_CNNIDE_Radar.rda")
  }
  save(all_data, results, sigma2e, taxis_df, file = save_fname)
}

## Some plots illustrating that the EnKF is acting sensibly

## Plot forecast filtered and truth for checking
par(mfrow = c(1,3))
XX <- run(Ypred_mean, feed_dict = fd); image.plot(matrix(XX, 64, 64))
YY <- apply(run(Ypred_updated, feed_dict = fd), 2, mean); image.plot(matrix(YY, 64, 64))
ZZ <- dfinal[taxis[i],,]; image.plot(matrix(ZZ, 64, 64))

## Plot histogram of prediction errors
hist(run(Z_tf_tiled[1,,] - Zsim_tf[1,,], feed_dict = fd))

## Plot the first three scenes, one conditional realisation of the
## forecast, the forecast, and the truth at the fourth scene
myscale <- scale_fill_gradientn(colours = nasa_palette, name = "Z")
pred_mu <- apply(ParticlePreds, 2, mean)
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = as.numeric(Pframe[1,,,1]))) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = as.numeric(Pframe[1,,,2]))) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = as.numeric(Pframe[1,,,3]))) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = ParticlePreds[10,,])) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = pred_mu)) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = c(dfinal[taxis[12],,]))) + myscale

