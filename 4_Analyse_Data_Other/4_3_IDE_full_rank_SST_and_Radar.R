####################################################
## Title: Full rank IDE for SST and radar data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################
library("dplyr")
library("gstat")
library("Matrix")
library("tensorflow")
library("R.utils")
sourceDirectory("../common")

## Analyse SST or radar data?
radar_data <- FALSE

## Use nlags = 2 to match CNNIDE
nlags <- 2L
if(radar_data) {
  nZones <- 1L
  nObs <- 4096L
  sigma2e <- (sqrt(16.23585) / 8.671842)^2 # estimated in 5_3
} else {
  nZones <- 19L
  nObs <- 1024L           # 1024 observations 
  sigma2e <- 0.01
}

## Set up IDE graph
N <- 200                 # number of gradient descents
ds1 <- mean(diff(s1))            # s1 spacings
ds2 <- mean(diff(s2))            # s2 spacings
ds <- sqrt(ds1 * ds2)            # area of pixel
sgrid_tf <- tfconst(as.matrix(sgrid))   # spatial grid

kernelside <- seq(0, 1, length.out = sqrtN_Basis)      # locs of kernel basis ..
kernellocs <- as.matrix(expand.grid(s1 = kernelside,   # functions on grid
                                    s2 = kernelside))
kernelsds <- 1/(1.5*sqrtN_Basis)       # width of keel basis functions
kernellocs_tf <- tfconst(kernellocs)   # put as TF variable
kernelsds_tf <- tfconst(kernelsds)     # put as TF variable

PHI_tf <- sqexp(sgrid_tf, 
                kernellocs_tf, 
                kernelsds)             # basis function matrix (4096 x 64)

flowinit_unnorm <- tf$placeholder(dtype = "float32", shape = c(sqrtN_Basis^2, 2L), name = "flowinit")
flow_coeffs_unnorm <- tfVar(flowinit_unnorm, "flowcoeffs")
flow_coeffs <- ((tf$sigmoid(flow_coeffs_unnorm) - 0.5) * 2)*5/64 # max/min = 5 pixels  

Dinit_unnorm <- tf$placeholder(dtype = "float32", shape = c(sqrtN_Basis^2, 1L), name = "Dinit")
diff_coeffs_unnorm <- tfVar(Dinit_unnorm, "diffcoeffs")
diff_coeffs <- tf$sigmoid(diff_coeffs_unnorm)*tfconst(0.0001)

flow_long <- tf$matmul(PHI_tf, flow_coeffs)
diff_long <- tf$matmul(PHI_tf, diff_coeffs)
sgridlong_tf <- tf$expand_dims(sgrid_tf, 0L) # Spatial grid as (1 x 4096 x 2)
K_IDE <- k_tf1(s = sgrid_tf,             # Construct kernel which is ..
               r = sgrid_tf,             # of size N_Batches x 4096 x 4096
               w = flow_long,
               D = diff_long)

## Obs cov. matrix
SIGMAobs <- tf$diag(rep(sigma2e, nObs))

## Init
Z <- tf$placeholder(dtype = "float32", shape = list(nlags + 1, nObs, 1), name = "Z")
Cobs <- tf$placeholder(dtype = "float32", shape = list(nlags + 1, nObs, W*H), name = "Cobs")
Yinit <- tf$placeholder(dtype = "float32", shape = list(W*H, 1L), name = "Yinit")
Covinit <- tf$placeholder(dtype = "float32", shape = list(W*H, W*H), name = "Covinit")
log_sigma2init <- tf$placeholder(dtype = "float32", shape = list())
log_rhoinit <- tf$placeholder(dtype = "float32", shape = list())

## Create the variables for the Matern
log_sigma2 <- tfVar(log_sigma2init, "log_sigma2") # 0.4
log_rho <- tfVar(log_rhoinit, "log_rho") # 0.1
sigma2 <- tf$exp(log_sigma2)
rho <- tf$exp(log_rho)

## Create the distance matrix and covariance matrix
d1 <- tf$square(sgrid_tf[, 1] - sgrid_tf[, 1, drop = FALSE])
d2 <- tf$square(sgrid_tf[, 2] - sgrid_tf[, 2, drop = FALSE])
D <- tf$sqrt(d1 + d2)
SIGMA <- Matern32(sigma2, rho, D)

## Create forecasted and filtered vectors/matrices
Yforecast <- Yfilter <- list()
Covforecast <- Covfilter <- list()
Ytilde <- Kalman <- S <- Sinv <- list()
Yfilter[[1]] <- Yinit
Covfilter[[1]] <- Covinit

Imat <- tf$eye(W*H)
NLL <- NegLikPart1 <- NegLikPart2 <- list()
Yforecast[[1]] <- Covforecast[[1]] <- tfconst(0)
for(j in 2 : (nlags + 2)) {
  Yforecast[[j]] <- tf$matmul(K_IDE, Yfilter[[j-1]])
  Covforecast[[j]] <- tf$matmul(tf$matmul(K_IDE, Covfilter[[j-1]]), tf$transpose(K_IDE)) +
                      SIGMA
  if(j <= (nlags + 1)) {
    Ytilde[[j]] <- Z[j,,] - tf$matmul(Cobs[j,,], Yforecast[[j]])
    S[[j]] <- tf$matmul(tf$matmul(Cobs[j,,], Covforecast[[j]]), tf$transpose(Cobs[j,,])) + SIGMAobs
    Sinv[[j]] <- tf$matrix_inverse(S[[j]])
    Kalman[[j]] <- tf$matmul(tf$matmul(Covforecast[[j]], tf$transpose(Cobs[j,,])), Sinv[[j]])
    Yfilter[[j]] <- Yforecast[[j]] + tf$matmul(Kalman[[j]], Ytilde[[j]])
    Covfilter[[j]] <- tf$matmul((Imat - tf$matmul(Kalman[[j]], Cobs[j,,])), Covforecast[[j]])
    
    NegLikPart1[[j]] <- logdet_tf(tf$cholesky(S[[j]]))
    NegLikPart2[[j]] <- tf$matmul(tf$matmul(tf$transpose(Ytilde[[j]]), Sinv[[j]]), Ytilde[[j]])
    NLL[[j]] <- NegLikPart1[[j]] + NegLikPart2[[j]]
  }
}
Cost <- tf$reduce_sum(NLL[-1])

## These were before run(init) but I think it was slowing things down
trainer <- tf$train$AdamOptimizer(0.1)$minimize(Cost)
run <- tf$Session()$run
init <- tf$global_variables_initializer()

logit <- function(x) -log(1/x - 1)
sigmoid <- function(x) 1 / (1 + exp(-x))

normflow_pars <- function(x) (sigmoid(x) - 0.5) * 2*5/64
unnormflow_pars <- function(x) logit(32*x/5 + 0.5)

normdiff_pars <- function(x) sigmoid(x)*0.0001
unnormdiff_pars <- function(x) logit(x/0.0001)

zmat <- array(0, dim = c(nlags + 1, nObs, 1))
Cobsmat <- array(0, dim = c(nlags + 1, nObs, W*H))
log_sigma2_IDE <- log_rho_IDE <- list()

## For each zone
for(zone in 1L:nZones) {

  ## Load data
  if(!radar_data) {
    load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_", zone, ".rda"))
    load(paste0("../3_Analyse_Data_CNNIDE/intermediates/kinit_zone", zone, ".rda"))
  } else {
    load("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Radar.rda")
    load("../3_Analyse_Data_CNNIDE/intermediates/kinit_Radar.rda")
  }
  taxis <- taxis_df$idx
  results_IDE <- list()
  log_sigma2_IDE[[zone]] <- log_rho_IDE[[zone]] <- NA

  ## For each time point
  for(i in seq_along(taxis)[-(1:nlags)]) {

      zmat <- array(
          Reduce("c", all_data$Z[(i - nlags) : i]),
          dim = c(nObs, nlags + 1, 1)) %>%
          aperm(c(2,1,3))

      Cobsmat <- array(
          Reduce("c", lapply(all_data$C[(i - nlags) : i], as.numeric)),
          dim = c(nObs, W*H, nlags + 1)) %>%
          aperm(c(3,1,2))

     Zinit <- cbind(Cobsmat[1,,] %*% as.matrix(sgrid),
                    z = zmat[1,,]) %>%
         as.data.frame()
      
     if(i == nlags + 1) {   ## IDW on first time step
       Yinit_vec <- idw(formula = z ~ 1,
                      locations = ~ s1 + s2,  
                      data = Zinit,
                      newdata = sgrid,
                      idp = 10)$var1.pred %>% 
                              matrix()
       Covinit_mat <- 0.1*diag(W*H)
       log_sigma2_scalar <- log(0.4)
       log_rho_scalar <- log(0.1)
     } else {  ## else use filtered version from previous run on second time point
         Yinit_vec <- run(Yfilter[[2]], feed_dict = fd)
         Covinit_mat <- run(Covfilter[[2]], feed_dict = fd)
         log_sigma2_scalar <- run(log_sigma2)
         log_rho_scalar <- run(log_rho)
     }

      sgrid$Yinit <- c(Yinit_vec)

      ## Set initialisers (variables will change in optim)
      fd_init <- dict(flowinit_unnorm = unnormflow_pars(cbind(u_pars, v_pars)),
                      Dinit_unnorm = unnormdiff_pars(matrix(Dpars)),
                      log_sigma2init = log_sigma2_scalar,
                      log_rhoinit = log_rho_scalar)
      run(init, feed_dict = fd_init)

      fd <- dict(Z = zmat,
                 Cobs = Cobsmat,
                 Yinit = Yinit_vec,     # Yinit does not change
                 Covinit = Covinit_mat) # Covinit does not change
      cost <- rep(0, N)

      ## Sometimes ML may not converge, if so ignore results.
      tryCatch(
       for(j in 1:N) {
        cat(paste0(j, " "))
        TFrun <- run(list(trainer, Cost), feed_dict = fd)
        cost[j] <- TFrun[[2]]
        if(j > 1)
        if(abs(cost[j] - cost[j-1]) < 0.5) break
       }, error = function(e) cat("Cholesky error. Stopping optimisation")) 
      
      ## Use what was estimated as initial conditions for next batch
      TFres <- run(list(flow_coeffs = flow_coeffs,
                        diff_coeffs = diff_coeffs,
                        Yfilter = Yfilter,
                        Covfilter = Covfilter,
                        Yforecast= Yforecast,
                        Covforecast = Covforecast),
                        feed_dict = fd)

      ## Get the kernel parameters to be initial conditions for next time point
      u_pars <- TFres$flow_coeffs[, 1]
      v_pars <- TFres$flow_coeffs[, 2]
      Dpars <- as.numeric(TFres$diff_coeffs)

      log_sigma2_IDE[[zone]][i] <- run(log_sigma2)
      log_rho_IDE[[zone]][i] <- run(log_rho)

      ## Compile results
      if(i < length(taxis))
          results_IDE[[i + 1]] <- 
              data.frame(fcast_mu_IDE = as.numeric(TFres$Yforecast[[nlags+2]]),
                         fcast_sd_IDE = sqrt(diag(TFres$Covforecast[[nlags+2]])))
      
      if(i == (nlags + 1)) {
          results_IDE[[i]] <- 
              data.frame(fcast_mu_IDE = as.numeric(TFres$Yfilter[[nlags+1]]),
                         fcast_sd_IDE = sqrt(diag(TFres$Covfilter[[nlags+1]])))
      } else {
          results_IDE[[i]]$filter_mu_IDE = as.numeric(TFres$Yfilter[[nlags+1]])
          results_IDE[[i]]$filter_sd_IDE = sqrt(diag(TFres$Covfilter[[nlags+1]]))
      }
      
      cat(paste0("IDE Zone ", zone, " Time point ", i, "\n"))
      
  }

  ## Save results
  if(radar_data) {
    save_fname1 <- paste0("intermediates/Results_fullrank_IDE_Radar.rda")
    save_fname2 <- paste0("intermediates/IDEcovparsRadar.rda")
  } else {
    save_fname1 <- paste0("intermediates/Results_IDE_Zone_", zone, ".rda")
    save_fname2 <-  paste0("intermediates/IDEcovparsZone", zone, ".rda")
  }
  save(results_IDE, file = save_fname1)
  save(log_sigma2_IDE, log_rho_IDE, file = save_fname2)
}
