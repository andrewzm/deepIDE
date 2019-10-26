####################################################
## Title: Stage 2 fitting for the CNN-IDE (cov pars)
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

library("tensorflow")
library("dplyr")
library("ggplot2")
library("R.utils")
sourceDirectory("../common") 

## Construct the graph
CNNgraph <- createCNNgraph(W = W,  H = H,
                           N_Channels = N_Channels,
                           N_Filters = N_Filters,
                           sqrtN_Basis = sqrtN_Basis, 
                           patch_size_flow = patch_size_flow,
                           patch_size_diff = patch_size_diff,
                           border_mask = border_mask)
list2env(CNNgraph, envir = .GlobalEnv)

## Update weights using trained weights from Stage 1
load(file = "./intermediates/SSTIDE_weights_TF.rda")
run <- tf$Session()$run
All_Vars_tf <- tf$trainable_variables(scope = NULL)
for(i in 1:length(All_Vars_tf)) {
    run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
}


## Load the fitting variables and the data
load("../1_Preproc_data/intermediates/TrainingData3D.rda")
load("../1_Preproc_data/intermediates/TrainingDataFinals.rda")
load("../1_Preproc_data/intermediates/TrainingDataPreds.rda")

## TensorFlow graph for the covariance matrix
log_sigma2_zone_all <- log_rho_zone_all <- rep(NA, nZones)
log_s2_zone <- tf$placeholder(dtype = "float32")
log_rho_zone <- tf$placeholder(dtype = "float32")
s2_zone <- tf$exp(log_s2_zone)
rho_zone <- tf$exp(log_rho_zone)
SIGMA_zone <-  Matern32(s2_zone, rho_zone, D)
SIGMA_zone_inv <- tf$matrix_inverse(SIGMA_zone)
R_zone <- tf$cholesky(SIGMA_zone)

## For each zone
for(zn in 1:nZones) {

    ## Initialise
    cat(paste0("Estimating parameters in Zone ", zn, " ...\n"))
    count <- 0
    Ytilde_cumsum <- 0

    ## See which images are in this zone
    zoneidx <- intersect(filter(Date_Zone_Map, zone == zn)$idx,
                         idxTrain)
    N_zone <- length(zoneidx)

    ## Divide these images into batches of size at most 32
    N_Cov_Batches <- floor(length(zoneidx)/32L)           # about 237
    Cov_Batch_idx <- rep(1:N_Cov_Batches, each = 32L)     # batch ID
    if((partbatch <- length(zoneidx) %% 32L) > 0) {       # if not an exact multiple of 32
        N_Cov_Batches <- N_Cov_Batches + 1           # then there is an extra batch
        Cov_Batch_idx <- c(Cov_Batch_idx, rep(N_Cov_Batches, partbatch))
    }
    
    ## Create an idx--batch map
    cov_batches <- data.frame(idx = zoneidx,
                              batch = Cov_Batch_idx)
    
    ## Graph to compute the negative log-likelihood
    ## Note: This can be put outside the for loop to increase comp. efficiency
    for(i in unique(cov_batches$batch)) {
        this_idx <- filter(cov_batches, batch == i)$idx
        fd <- dict(data_in = d[this_idx,,,,drop = F],          # Create dictionary
                   data_current = dfinal[this_idx,,,drop=F],
                   data_future = dpred[this_idx,,,drop=F])
        res <- run(list(Ypred_masked, data_future_masked), feed_dict = fd)
        Ytilde <- res[[2]][,,1] - res[[1]][,,1]
        if(is.null(dim(Ytilde)))
            Ytilde <- matrix(Ytilde,nrow = 1)
        Ytilde_cumsum <- Ytilde_cumsum + crossprod(Ytilde)
    }
    Ytilde_cumsum_tf <- tfconst(Ytilde_cumsum)
    nllik_tf <- tfconst(N_zone) * logdet_tf(R_zone) +
        tf$linalg$trace(tf$matmul(SIGMA_zone_inv,
                                  Ytilde_cumsum_tf))
    nllik <- function(log_theta) {
        fd <- dict(log_s2_zone = log_theta[1],
                   log_rho_zone = log_theta[2])
        as.numeric(run(nllik_tf, feed_dict = fd))
    }

    ## Optimise the NLL
    optimres <- optim(c(-4, -2), nllik, control = list(trace = 1))

    ## Return the ML parameter estimates
    log_sigma2_zone_all[zn] <- optimres$par[1]
    log_rho_zone_all[zn] <- optimres$par[2]
}

## Save the parameter estimates
save(log_sigma2_zone_all, log_rho_zone_all, file = "./intermediates/SST_cov_pars.rda")

## Plot the parameter estimates
png("./img/covpars_by_zone.png")
par(mfrow = c(2,1))
plot(exp(log_sigma2_zone_all))
plot(exp(log_rho_zone_all))
dev.off()

