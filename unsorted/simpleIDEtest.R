library(IDE)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(Matrix)

rm(list=ls())
set.seed(1)
source("~/Wollongong/deepIDE/netarchitectures.R")

###################
## Helper Functions
###################
run <- tf$Session()$run
tfconst <- function(x, name = NULL) tf$constant(x, dtype = "float32", name = name)
tfVar <- function(x, name = NULL) tf$Variable(x, dtype = "float32", name = name)

################
## Simulate data
################
buffer <- 10L
W <- 41L
H <- 41L
pcW_buffer <- buffer/(W - 1)
pcH_buffer <- buffer/(H - 1)

Wx <- W + 2L*buffer
Hx <- H + 2L*buffer
T <- 25L
taxis <- 1:T
N_Sims <- 400L
N_Channels <- 3L
N_Data <-  T - N_Channels
N_Filters <- 16L
N_Batch <- 16L
s1 <- seq(0, 1, length.out = W)
s2 <- seq(0, 1, length.out = H)
ds1 <- mean(diff(s1))
ds2 <- mean(diff(s2))
ds <- sqrt(ds1 * ds2)
sgrid <- expand.grid(s1 = s1, s2 = s2)

s1x <- seq(-pcW_buffer, 1 + pcW_buffer, length.out = Wx)
s2x <- seq(-pcH_buffer, 1 + pcH_buffer, length.out = Hx)
sgridx <- expand.grid(s1 = s1x, s2 = s2x)
#ggplot(sgrid) + geom_tile(aes(s1, s2, fill =  Y1train))

k <- function(theta, s, r) {
  r1_minus_s1 <- outer(r[,1], s[,1], '-')
  r2_minus_s2 <- outer(r[,2], s[,2], '-')
  theta[1] * exp( - (1/theta[2]) * ((r1_minus_s1 - theta[3])^2 + 
                                      (r2_minus_s2 - theta[4])^2)) * ds^2 
}

Y <- Yval <- array(dim = c(W, H, T*N_Sims))
Yx <- Yvalx <- array(dim = c(Wx, Hx, T*N_Sims))
thetasave <- thetasave_val <- NULL
for(j in 1:N_Sims) {
  sctrain <- runif(2, 0.1, 0.9) #c(0.5, 0.5)
  scval <- runif(2, 0.1, 0.9)   #c(0.4, 0.8)
  sgrid$Y1train <- exp(-((sgrid$s1 - sctrain[1])^2 + (sgrid$s2 - sctrain[2])^2) / 0.05^2)
  sgrid$Y1val <- exp(-((sgrid$s1 - scval[1])^2 + (sgrid$s2 - scval[2])^2) / 0.05^2)
  sgridx$Y1train <- exp(-((sgridx$s1 - sctrain[1])^2 + (sgridx$s2 - sctrain[2])^2) / 0.05^2)
  sgridx$Y1val <- exp(-((sgridx$s1 - scval[1])^2 + (sgridx$s2 - scval[2])^2) / 0.05^2)
  ## theta <- c(300, 0.001, -0.01, 0.01)
  ## K0 <- matrix(k(theta, matrix(c(0.5, 0.5),1 ,2), sgrid), W, H)
  theta <- c(300, 0.001, runif(2, -0.02, 0.02))
  thetasave <- rbind(thetasave, theta)
  K <- t(k(theta, sgridx, sgridx))
  K[which(K < 1e-10, arr.ind = TRUE)] <- 0
  K <- as(K, "dgCMatrix")
  
  theta_val <- c(300, 0.001, runif(2, -0.02, 0.02))
  thetasave_val <- rbind(thetasave_val, theta_val)
  K_val <- t(k(theta_val, sgridx, sgridx))
  K_val[which(K_val < 1e-10, arr.ind = TRUE)] <- 0
  K_val <- as(K_val, "dgCMatrix")
  
  Y[,,(j-1)*T + 1] <- matrix(sgrid$Y1train, W, H)
  Yval[,,(j-1)*T + 1] <- matrix(sgrid$Y1val, W, H)
  
  Yrunx <- sgridx$Y1train
  Yrunvalx <- sgridx$Y1val
  
  for(i in 2:T) {
    ## if(i == round(T/2)) {
    ##   theta[3] <- -theta[3]
    ##   theta[4] <- -theta[4]
    ##   K <- t(k(theta, sgrid, sgrid))
    ## }
    Yrunx <- K %*% Yrunx
    Yrunx_mat <- matrix(Yrunx, Wx, Hx)
    Y[,,(j-1)*T + i] <- Yrunx_mat[(buffer+1):(buffer + W),
                                  (buffer+1):(buffer + H)]
    
    Yrunvalx <- K_val %*% Yrunvalx
    Yrunvalx_mat <- matrix(Yrunvalx, Wx, Hx)
    Yval[,,(j-1)*T + i] <- Yrunvalx_mat[(buffer+1):(buffer + W),
                                        (buffer+1):(buffer + H)]
  }
}

singleImages <- singleImages_val <- list()
for(j in 1:N_Sims) {
  singleImages[[j]] <- singleImages_val[[j]] <- list()
  for(i in seq_along(taxis)) {
    singleImages[[j]][[i]] <- Y[,,(j-1)*T + i]
    singleImages_val[[j]][[i]] <- Yval[,,(j-1)*T + i]
  }
}

dval <- d <- array(dim = c(N_Data*N_Sims, H, W, N_Channels))
dfinal_val <- dfinal <- array(dim = c(1, H, W, N_Data*N_Sims)) # Treat final images as channels for eff. conv.
dpred_val <- dpred <- array(dim = c(1, H, W, N_Data*N_Sims))
for(j in 1:N_Sims)
for(i in 1:N_Data) {
  for(k in 1:N_Channels) {
    d[(j-1)*N_Data + i,,,k] <- singleImages[[j]][[i + k - 1]]
    dval[(j-1)*N_Data + i,,,k] <- singleImages_val[[j]][[i + k - 1]]
  }
  dfinal[1,,,(j-1)*N_Data + i] <- singleImages[[j]][[i + N_Channels - 1]]
  dfinal_val[1,,,(j-1)*N_Data + i] <- singleImages_val[[j]][[i + N_Channels - 1]]
  
  dpred[1,,,(j-1)*N_Data + i] <- singleImages[[j]][[i + N_Channels]]
  dpred_val[1,,,(j-1)*N_Data + i] <- singleImages_val[[j]][[i + N_Channels]]
  
}

############
## TRAIN CNN
############
k_tf2 <- function(theta_tf, r) {
  d1 <- tf$square(r[,, 1] - theta_tf[, 3, drop = FALSE] - tfconst(0.5))
  d2 <- tf$square(r[,, 2] - theta_tf[, 4, drop = FALSE] - tfconst(0.5))
  tfconst(100)*theta_tf[, 1, drop = FALSE] * tf$exp(-(d1 + d2) / (theta_tf[, 2, drop = FALSE]/tfconst(100))) * tfconst(ds^2)
}
sgrid_tf <- tfconst(as.matrix(sgrid))
dfinal_tf <- tfconst(dfinal)
dpred_tf <- tfconst(dpred)
d_tf <- tfconst(d, "data")
dval_tf <- tfconst(dval, "data_val")

## ## Generate data using GPU (doesn't make a difference)
## theta2A <- theta
## theta2A[1] <- theta2A[1]/100; theta2A[2] <- theta2A[2]*100
## theta2B <- theta2A
## theta2B[3] <- -theta2B[3]
## theta2B[4] <- -theta2B[4]
## #theta_tf <- tfVar(matrix(0.1, N_Batches, 4))
## theta_tf_true <- tfconst(rbind(matrix(rep(theta2A, floor(T/2) - 1), 
##                                       nrow = floor(T/2) - 1, byrow = TRUE),
##                                matrix(rep(theta2B, ceiling(T/2) + 1), 
##                                       nrow = ceiling(T/2) + 1, byrow = TRUE)))
## Ypred_true_list <- K_Square_true <- list()
## Ypred_true_list[[1]] <- tfconst(matrix(sgrid$Y1, W, H)) %>% 
##   tf$expand_dims(0L) %>% 
##   tf$expand_dims(3L)
## for(i in 2L:T) {
##   K_IDE_Long_true <- k_tf2(theta_tf = theta_tf_true[i ,,drop = FALSE],
##                            r = tf$reshape(sgrid_tf, c(1L, dim(sgrid))))
##   K_Square_true[[i - 1]] <- tf$reshape(K_IDE_Long_true, shape = c(H, W, 1L, 1L))
##   Ypred_true_list[[i]] <- tf$nn$conv2d(Ypred_true_list[[i-1]], 
##                                        K_Square_true[[i - 1]],
##                                        strides = c(1L, 1L, 1L, 1L), 
##                                        padding = "SAME")
##   #if(round(T/2) == i)
##   #  Ypred_true_list[[i]] <- Ypred_true_list[[1]]  
## }
## dfinal_tf <- tf$stack(Ypred_true_list[-c(1:(N_Channels - 1), T)], axis = 3L) %>%
##   tf$reshape(c(1L, W, H, N_Batches))
## dpred_tf <- tf$stack(Ypred_true_list[-(1:N_Channels)], axis = 3L) %>%
                                        #  tf$reshape(c(1L, W, H, N_Batches))

#####################
## BACKGROUND NETWORK
#####################

data_in <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H, N_Channels))
data_current <- tf$placeholder(dtype = "float32", shape = list(1L, W, H, NULL))
data_future <- tf$placeholder(dtype = "float32", shape = list(1L, W, H, NULL))

nnet <- Conv2net(data_in)
theta_tf <- nnet$theta_tf

###############
## FINAL LAYER
##############

#theta_tf <- theta_tf_true[(N_Channels+1):T,]
K_IDE_Long <- k_tf2(theta_tf,
              tf$reshape(sgrid_tf, c(1L, dim(sgrid))))
K_Square <- tf$transpose(
              tf$matrix_transpose(tf$reshape(K_IDE_Long, shape = c(1L, -1L, W, H))))
Ypred <- tf$nn$depthwise_conv2d(data_current, K_Square,
                               strides = c(1L, 1L, 1L, 1L), padding = "SAME")
Cost <- tf$reduce_mean(tf$square(data_future - Ypred))

###########
## TRAINING
###########

cat("Learning weight parameters... \n")
nsteps <- 100000
global_step = tf$Variable(0, trainable = TRUE)
## init_learning_rate <- 0.1
init_learning_rate <- 0.001
lr <- tf$train$exponential_decay(learning_rate = init_learning_rate,
                                  global_step = global_step,
                                  decay_steps = nsteps,
                                  decay_rate = 0.96,
                                  staircase = TRUE)
##trainnet <- tf$train$GradientDescentOptimizer(lr)$minimize(Cost, global_step = global_step)
trainnet = tf$train$AdamOptimizer(init_learning_rate)$minimize(Cost)
init <- tf$global_variables_initializer()
run(init)

Objective <- Objective_val <- rep(0, nsteps)
options(scipen = 3, digits = 3)
for(i in 1:nsteps) {
    idx <- sample(1:(N_Data*N_Sims), N_Batch, replace = FALSE)
    fd <- dict(data_in = d[idx,,,],
               data_current = dfinal[,,,idx,drop=F],
               data_future = dpred[,,,idx,drop=F])
    run(trainnet, feed_dict = fd)
    Objective[i] <- run(Cost, feed_dict = fd)
  ## Objective[i] <- run(Cost, feed_dict = dict(data_in = d))
  
  if((i %% 100) == 0) {
      ## thetavals <- round(colMeans(run(theta_tf, feed_dict = dict(data_in = d))), 4)
    fd <- dict(data_in = dval[idx,,,],
               data_current = dfinal_val[,,,idx,drop=F],
               data_future = dpred_val[,,,idx,drop=F])
      Objective_val[i/100] <- run(Cost, feed_dict = fd)
      thetavals <- round(colMeans(run(theta_tf, feed_dict = fd)), 4)
      cat(paste0("Step ", i, " ... Cost: ", Objective_val[i/100], " theta: ",
                 paste(thetavals, collapse = " ") , "\n"))
   }
}

stop()

view_idx <- sample(1:(N_Data*T), 1)
fd <- dict(data_in = dval[view_idx,,,,drop=FALSE],
           data_current = dfinal_val[,,,view_idx,drop=FALSE],
           data_future = dpred_val[,,,view_idx,drop=FALSE])
im <- fields::image.plot
png("Temp1.png"); im(run(data_current[1,,,1], feed_dict = fd)); dev.off()
png("Temp2.png"); im(run(data_future[1,,,1], feed_dict = fd)); dev.off()
png("Temp3.png"); im(run(Ypred[1,,,1], feed_dict = fd)); dev.off()


run(theta_tf, feed_dict = fd)


#png("Temp1.png"); plot(thetasave[,3]); dev.off()
#png("Temp2.png"); plot(run(theta_tf, feed_dict = fd)[,3]); dev.off()




conv1wts_df <- expand.grid(s1 = 1:5, s2 = 1:5, inchannel = 1:N_Channels, filternum = 1:N_Filters)
conv1wts_df$z <- as.numeric(run(nnet$conv1wts, feed_dict = fd))
g <- ggplot(conv1wts_df) +
    geom_tile(aes(s1, s2, fill = z)) +
    facet_grid(filternum ~ inchannel) +
    scale_fill_distiller(palette = "Spectral") + theme_bw()
ggsave(g, file = "FiltersLayer1.png", height = N_Filters, width = N_Channels + 2)

conv2wts_df <- expand.grid(s1 = 1:5, s2 = 1:5, inchannel = 1:N_Filters, filternum = 1:N_Filters)
conv2wts_df$z <- as.numeric(run(nnet$conv2wts, feed_dict = fd))
g <- ggplot(conv2wts_df) +
    geom_tile(aes(s1, s2, fill = z)) +
    facet_grid(filternum ~ inchannel) +
    scale_fill_distiller(palette = "Spectral") + theme_bw()
ggsave(g, file = "FiltersLayer2.png", height = N_Filters, width = N_Filters + 2)

relu1out_df <- expand.grid(batch = 1:N_Data, s1 = 1:W, s2 = 1:H, channel = 1:N_Filters)
relu1out_df$z <- as.numeric(run(nnet$relu1out, feed_dict = fd))
g <- ggplot(filter(relu1out_df, batch %% 5 == 1)) +
  geom_tile(aes(s1, s2, fill = z)) +
  facet_grid(batch ~ channel) +
  scale_fill_distiller(palette = "Spectral") + theme_bw()

pool1out_df <- expand.grid(batch = 1:N_Batches, s1 = 1:(ceiling(W/2)), s2 = 1:(ceiling(H/2)), 
                           channel = 1:N_Filters)
pool1out_df$z <- as.numeric(run(nnet$norm1out))
g <- ggplot(filter(pool1out_df, batch %% 5 == 1)) +
  geom_tile(aes(s1, s2, fill = z)) +
  facet_grid(batch ~ channel) +
  scale_fill_distiller(palette = "Spectral") + theme_bw()
