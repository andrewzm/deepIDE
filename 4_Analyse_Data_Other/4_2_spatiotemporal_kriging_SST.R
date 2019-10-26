####################################################
## Title: Spatial-temporal kriging for SST data
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

library("dplyr")
library("ggplot2")
library("tensorflow")
library("tidyr")
library("R.utils")
sourceDirectory("../common")

## Define covariance function
covfun_tf <- function(sigma2, tau, anis, s1, s2) {
    d1 <- tf$square(s2[, 1] - s1[, 1, drop = FALSE])
    d2 <- tf$square(s2[, 2] - s1[, 2, drop = FALSE])
    d3 <- tf$square(anis*(s2[, 3] - s1[, 3, drop = FALSE]))
    D <- tf$sqrt(d1 + d2 + d3)
    tf$multiply(sigma2, tf$exp(-D / tau))
}

## Set up graph
nlags <- 3L
nObs <- 1024L           # 1024 observations 
n <- as.integer(nObs * (nlags + 1))

load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_1.rda"))
taxis <- taxis_df$idx
sobs_tf <- tf$placeholder(dtype = "float32", shape = list(n, 3L))
Z_tf <- tf$placeholder(dtype = "float32", shape = list(n, 1L))
logtheta_tf <- tf$placeholder(dtype = "float32", shape = list(3L))
sigma2e_tf <- tf$constant(sigma2e)
anis <- tf$exp(logtheta_tf[1])
sigma2_tf <- tf$minimum(tf$exp(logtheta_tf[2]), exp(4))
tau_tf <- tf$minimum(tf$exp(logtheta_tf[3]), exp(4))
SIGMA_tf <- covfun_tf(sigma2_tf, tau_tf, anis, sobs_tf, sobs_tf) + sigma2e_tf * tf$eye(n)
L_tf <- tf$cholesky(SIGMA_tf)
LinvZ_tf <- tf$matrix_solve(L_tf, Z_tf)
Zt_SIGMAinv_Z_tf <- tf$matmul(tf$transpose(LinvZ_tf), LinvZ_tf)
NLL <- tf$reduce_sum(tf$log(tf$diag_part(L_tf))) + 0.5*Zt_SIGMAinv_Z_tf

## Function to optimise
run_negloglik_tf_optim <- function(logtheta, sobs, Z) {
    fd <- dict(logtheta_tf = logtheta,
               sobs_tf = sobs,
               Z_tf = Z)
    run(NLL, feed_dict = fd)
}

## Prediction grid 
sgrid <- all_data$sgrid
predgrid_tf <- tf$placeholder(dtype = "float32", shape = list(as.integer(nrow(sgrid) * (nlags + 2)),
                                                              3L))
## Prediction equations 
SIGMAstarobs <- covfun_tf(sigma2_tf, tau_tf, anis, predgrid_tf, sobs_tf)
SIGMAinv_Z_tf <- tf$matrix_solve(tf$transpose(L_tf), LinvZ_tf)
Ypred_tf <- tf$matmul(SIGMAstarobs, SIGMAinv_Z_tf)
SIGMAinv_tf <- tf$matrix_inverse(SIGMA_tf)
Temp1 <- tf$matmul(SIGMAstarobs, SIGMAinv_tf)
Temp2 <- tf$multiply(Temp1, SIGMAstarobs)
Temp3 <- tf$reduce_sum(Temp2, 1L)
Ypredse_tf <- tf$sqrt(sigma2_tf - Temp3)

## For each zone
for(zone in 1:nZones) {

    ## Load results/data and initialise
    load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_", zone, ".rda"))
    results_STK <- list()

    ## For each time point
    for(i in seq_along(taxis)) {
     if(i > nlags) {

        ## Get out data, residuals and obs locations
        z <- Reduce("c", all_data$Z[(i - nlags) : i])
        C <- Reduce("rbind", all_data$C[(i - nlags) : i])
        sobs <- as.data.frame(as.matrix(C) %*% as.matrix(all_data$sgrid))
        sobs$z <- z
        sobs$t <- rep(taxis[i - nlags] : taxis[i], each = 1024)
        LM <- lm(z ~ s1 + s2, data = sobs)
        residuals <- LM$residuals

        ## Find the ML estimates
        initpars <- rep(log(0.5), 3)
        t1 <- Sys.time()
        theta <- optim(par = initpars,
                       fn = run_negloglik_tf_optim,
                       control = list(maxit = 400),
                       sobs = as.matrix(sobs[,c("s1", "s2", "t")]),
                       Z = as.matrix(residuals))
        t2 <- Sys.time()

        ## Print results
        data.frame(Est. = exp(theta$par),
                   row.names = c("anis", "sigma2", "tau"))
        print(t2 - t1)

        ## Prediction
        predgrid <- crossing(t = taxis[(i - nlags) : (i + 1)], sgrid) %>%
                     select(s1, s2, t)
        fd <- dict(logtheta_tf = theta$par,
                    sobs_tf = as.matrix(sobs[,c("s1", "s2", "t")]),
                    Z_tf = as.matrix(residuals),
                    predgrid_tf = predgrid)
        predgrid$Ypred <- as.numeric(run(Ypred_tf, feed_dict = fd) +
                                   predict(LM, predgrid))
        predgrid$Ypredse <- as.numeric(run(Ypredse_tf,feed_dict = fd))

        ## Print image
        if(zone == 1L & i == 10L) {
          g <- ggplot(predgrid) + geom_tile(aes(s1, s2, fill = Ypred)) + 
               facet_wrap(~t) +
               scale_fill_gradientn(colours = nasa_palette) + theme_bw()
          ggsave(g, filename = paste0("img/STK_Zone", zone,"_Time", i, ".png"))
        }
        presentgrid <- filter(predgrid, t == taxis[i])
        fcastgrid <- filter(predgrid, t == taxis[i + 1])
        
        ## Compile results
        if(i < length(taxis))
            results_STK[[i + 1]] <- data.frame(fcast_mu_STK = fcastgrid$Ypred,
                                               fcast_sd_STK = fcastgrid$Ypredse)
        
        if(i == (nlags + 1)) {
            results_STK[[i]] <- data.frame(filter_mu_STK = presentgrid$Ypred,
                                           filter_sd_STK = presentgrid$Ypredse)
        } else {
            results_STK[[i]]$filter_mu_STK = presentgrid$Ypred
            results_STK[[i]]$filter_sd_STK = presentgrid$Ypredse
        }
        
        cat(paste0("STK Zone ", zone, " Time point ", i, "\n"))
     }
    }

    ## Save results
    save(results_STK, file = paste0("intermediates/Results_STK_Zone_", zone, ".rda"))
}
