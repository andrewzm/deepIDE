library(dplyr)
library(devtools)
library(ggplot2)
library(gstat)
library(spacetime)
library(sp)
library(tensorflow)
library(tidyr)

source("../utils.R")
for(zone in 1:19L) {
    load(paste0("Results_CNNIDE_Zone_", zone, ".rda"))
    taxis <- taxis_df$idx
    results_kriging <- list()
    
    for(i in seq_along(taxis)) {
        z <- all_data$Z[[i]]
        C <- all_data$C[[i]]
        sobs <- as.data.frame(as.matrix(C) %*% as.matrix(all_data$sgrid))
        sobs$z <- z
        sobs$t <- taxis[i]

        logtheta_tf <- tf$placeholder(dtype = "float32", shape = list(2L))
        LM <- lm(z ~ s1 + s2, data = sobs)
        residuals <- LM$residuals
        Z_tf <- tf$constant(as.matrix(residuals), dtype = "float32")
        sobs_tf <- tf$constant(as.matrix(sobs[,c("s1", "s2")]), dtype = "float32")

        covfun_tf <- function(sigma2, tau, s1, s2) {
            d1 <- tf$square(s2[, 1] - s1[, 1, drop = FALSE])
            d2 <- tf$square(s2[, 2] - s1[, 2, drop = FALSE])
            D <- tf$sqrt(d1 + d2)
            tf$multiply(sigma2, tf$exp(-D / tau))
        }

        ## covfun_tf <- function(sigma2, tau, anis, s1, s2) {
        ##     d1 <- tf$square(s2[, 1] - s1[, 1, drop = FALSE])
        ##     d2 <- tf$square(s2[, 2] - s1[, 2, drop = FALSE])
        ##     DS2 <- (d1 + d2)
        ##     DT2 <- (s2[, 3] - s1[, 3, drop = FALSE])

        ##     Temp1 <-  anis*DT2 + 1
        ##     Temp2 <- -(DS2 * tau) / Temp1
        ##     tf$divide(tf$multiply(sigma2, tf$exp(Temp2)), Temp1)
        ## }

        n <- nrow(sobs_tf)
        sigma2e_tf <- tf$constant(sigma2e)
        sigma2_tf <- tf$minimum(tf$exp(logtheta_tf[1]), exp(4))
        tau_tf <- tf$minimum(tf$exp(logtheta_tf[2]), exp(4))
        SIGMA_tf <- covfun_tf(sigma2_tf, tau_tf, sobs_tf, sobs_tf) + sigma2e_tf * tf$eye(n)
        L_tf <- tf$cholesky(SIGMA_tf)
        LinvZ_tf <- tf$matrix_solve(L_tf, Z_tf)
        Zt_SIGMAinv_Z_tf <- tf$matmul(tf$transpose(LinvZ_tf), LinvZ_tf)
        NLL <- tf$reduce_sum(tf$log(tf$diag_part(L_tf))) + 0.5*Zt_SIGMAinv_Z_tf

        fd = dict(logtheta_tf = c(0.2, 0.3))
        run <- tf$Session()$run
        run(NLL, feed_dict = fd)

        run_negloglik_tf_optim <- function(logtheta) {
            fd <- dict(logtheta_tf = logtheta)
            run(NLL, feed_dict = fd)
        }

        t3 <- Sys.time()
        theta <- optim(par = rep(log(0.5), 2),
                       fn = run_negloglik_tf_optim,
                       control = list(maxit = 400))
        t4 <- Sys.time()

        ## Print results
        data.frame(Est. = exp(theta$par),
                   row.names = c("sigma2", "tau"))
        print(t4 - t3)

        sgrid <- all_data$sgrid
        newgrid <- sgrid
        predgrid_tf <- tf$constant(as.matrix(newgrid), dtype = "float32")

        SIGMAstarobs <- covfun_tf(sigma2_tf, tau_tf, predgrid_tf, sobs_tf)
        SIGMAinv_Z_tf <- tf$matrix_solve(tf$transpose(L_tf), LinvZ_tf)
        Ypred_tf <- tf$matmul(SIGMAstarobs, SIGMAinv_Z_tf)

        SIGMAinv_tf <- tf$matrix_inverse(SIGMA_tf)
        Temp1 <- tf$matmul(SIGMAstarobs, SIGMAinv_tf)
        Temp2 <- tf$multiply(Temp1, SIGMAstarobs)
        Temp3 <- tf$reduce_sum(Temp2, 1L)
        Ypredse_tf <- tf$sqrt(sigma2_tf - Temp3)

        fd <- dict(logtheta_tf = theta$par)
        newgrid$Ypred <- as.numeric(run(Ypred_tf,feed_dict = fd) +
                                   predict(LM, newgrid))
        newgrid$Ypredse <- as.numeric(run(Ypredse_tf,feed_dict = fd))

         g <- ggplot(newgrid) + geom_tile(aes(s1, s2, fill = Ypred)) + 
             facet_wrap(~t) +
             scale_fill_gradientn(colours = nasa_palette) + theme_bw()
        ## ggsave(g, filename = paste0("../STkrigingpics/STK_Zone", zone,"_Time", i, ".png"))
        ## ggsave(g, filename = paste0("~/Dropbox/TEMP.png", zone,"_Time", i, ".png"))

        results_kriging[[i]] <- data.frame(filter_mu_kriging = newgrid$Ypred,
                                           filter_sd_kriging = newgrid$Ypredse)
        cat(paste0("Kriging Zone ", zone, " Time point ", i, "\n"))
      }
      save(results_kriging, file = paste0("Results_kriging_Zone_", zone, ".rda"))
}
