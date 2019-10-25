library(IDE)
library(FRK)
library(sp)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(Matrix)
library(tidyr)

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
W <- 64L
H <- 64L
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
# 
# s1x <- seq(-pcW_buffer, 1 + pcW_buffer, length.out = Wx)
# s2x <- seq(-pcH_buffer, 1 + pcH_buffer, length.out = Hx)
# sgridx <- expand.grid(s1 = s1x, s2 = s2x)
#ggplot(sgrid) + geom_tile(aes(s1, s2, fill =  Y1train))

set.seed(1)
zlocs <- data.frame(s1 = runif(300, min = -pcW_buffer, max = 1 + pcH_buffer),
                    s2 = runif(300, min = -pcH_buffer, max = 1 + pcW_buffer))

## Spatial decomposition
Y_basis <- auto_basis(manifold = plane(),
                      data = SpatialPoints(zlocs),
                      regular = 1,
                      nres = 2)
r <- nbasis(Y_basis)

## Kernel decomposition
G_const <- constant_basis()

## Other parameters
sigma2_eta <- 0.01^2
sigma2_eps <- 0.01^2

## Spatial domain
bbox <- matrix(c(-pcW_buffer, -pcH_buffer, 1 + pcW_buffer,1 + pcH_buffer),2,2)
sx <- IDE:::construct_grid(bbox, Wx)
alpha <- matrix(0,r,T)

## Kernel
G <- auto_basis(plane(), data = SpatialPoints(sx$s_grid_df),nres = 1)
nbk <- nbasis(G)
K_basis <- list(G_const, G_const, G, G)
time_map <- data.frame(timeind = paste0("Y",0:(T-1)),
                       time = as.Date(0:(T-1), origin = "2017-12-01"),
                       stringsAsFactors = FALSE)
## Construct matrices
Sigma_eta <- sigma2_eta * Diagonal(r)
Q_eta <- Sigma_eta %>% solve()
Mfun <- IDE:::construct_M(Y_basis, sx)

Y <- array(dim = c(W, H, T*N_Sims))
thetasave <- thetasave_val <- NULL

for(j in 1:N_Sims) {
  k <- list(150, 0.002, 0.1*rnorm(nbk), 0.1*rnorm(nbk))
  thetasave <- rbind(thetasave, unlist(k))
  alpha[sample(1:r,10),1] <- 1
  M <- Mfun(K_basis, k)
  PHI <- eval_basis(Y_basis, sx$s_grid_mat)
  sx$s_grid_df$Y0 <- (PHI %*% alpha[,1]) %>% as.numeric()
  Yrunx_mat <- matrix(sx$s_grid_df$Y0, Wx, Hx)
  Y[,,(j-1)*T + 1] <- Yrunx_mat[(buffer+1):(buffer + W),
                                (buffer+1):(buffer + H)]
  for(i in 2:T) {
    alpha[,i] <- (M %*% alpha[, i-1]) %>% as.numeric() +
      sqrt(sigma2_eta)*rnorm(r)
    Yrunx_mat<- matrix(PHI %*% alpha[,i], Wx, Hx)
    Y[,,(j-1)*T + i] <- Yrunx_mat[(buffer+1):(buffer + W),
                                  (buffer+1):(buffer + H)]
    #sx$s_grid_df[paste0("Y",i)] <- 
  }
}

singleImages <- list()
for(j in 1:N_Sims) {
  singleImages[[j]] <- list()
  for(i in seq_along(taxis)) {
    singleImages[[j]][[i]] <- Y[,,(j-1)*T + i]
  }
}

d <- array(dim = c(N_Data*N_Sims, H, W, N_Channels))
dfinal <- array(dim = c(N_Data*N_Sims, H, W )) # Treat final images as channels for eff. conv.
dpred <- array(dim = c(N_Data*N_Sims, H, W))
for(j in 1:N_Sims)
for(i in 1:N_Data) {
  for(k in 1:N_Channels) {
    d[(j-1)*N_Data + i,,,k] <- singleImages[[j]][[i + k - 1]]
  }
  dfinal[(j-1)*N_Data + i,,] <- singleImages[[j]][[i + N_Channels - 1]]
  dpred[(j-1)*N_Data + i,,] <- singleImages[[j]][[i + N_Channels]]
}
#dfinal <- d[c(-(1:(N_Channels - 1)), -N_Data*N_Sims),,,1]
#dpred <- d[-(1:N_Channels),,,1]
############
## TRAIN CNN
############
k_tf2 <- function(flow_long, r) {
  d1 <- tf$square(r[,, 1] - flow_long[,, 3, drop = FALSE] - r[,, 1, drop = FALSE])
  d2 <- tf$square(r[,, 2] - flow_long[,, 4, drop = FALSE] - r[,, 2, drop = FALSE])
  tfconst(100)*flow_long[,, 1, drop = FALSE] * tf$exp(-(d1 + d2) / (flow_long[,, 2, drop = FALSE]/tfconst(100))) * tfconst(ds^2)
}
sgrid_tf <- tfconst(as.matrix(sgrid))
dfinal_tf <- tfconst(dfinal)
dpred_tf <- tfconst(dpred)
d_tf <- tfconst(d, "data")

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
data_current <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))
data_future <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))

nnet <- ConvDecovnet(data_in)
flow <- nnet$flow
flow_long <- tf$reshape(flow, c(-1L, W*H, 4L))
###############
## FINAL LAYER
##############

#theta_tf <- theta_tf_true[(N_Channels+1):T,]
K_IDE <- k_tf2(flow_long,
               tf$reshape(sgrid_tf, c(1L, W*H, 2L)))

## K_IDE_Long is already in the right format -- need
## to just multiple it by data_current, which however
## is in the wrong format
data_current_long <- tf$reshape(data_current, c(-1L, W*H, 1L))
data_future_long <- tf$reshape(data_future, c(-1L, W*H, 1L))
Ypred <- tf$matmul(K_IDE, data_current_long)
Cost <- tf$reduce_mean(tf$square(data_future_long - Ypred))


# flow_long <- tfconst(array(c(rep(3, W*H), rep(0.1, W*H), rep(-0.02, W*H), rep(-0.02, W*H)), dim = c(1L, W*H, 4L)))
# K_IDE <- k_tf2(flow_long,
#                tf$reshape(sgrid_tf, c(1L, W*H, 2L)))
# data_current_long <- tf$reshape(dfinal_tf[1,,], c(-1L, W*H, 1L))
# data_future_long <- tf$reshape(dpred_tf[1,,], c(-1L, W*H, 1L))
# Ypred <- tf$matmul(K_IDE, data_current_long)

###########
## TRAINING
###########

cat("Learning weight parameters... \n")
nsteps <- 100000
global_step = tf$Variable(0, trainable = TRUE)
## init_learning_rate <- 0.1
init_learning_rate <- 0.0001
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
               data_current = dfinal[idx,,,drop=F],
               data_future = dpred[idx,,,drop=F])
    run(trainnet, feed_dict = fd)
    Objective[i] <- run(Cost, feed_dict = fd)
  ## Objective[i] <- run(Cost, feed_dict = dict(data_in = d))
  
  if((i %% 10) == 0) {
      ## thetavals <- round(colMeans(run(theta_tf, feed_dict = dict(data_in = d))), 4)
      ## fd <- dict(data_in = dval[idx,,,],
      ##          data_current = dfinal_val[,,,idx,drop=F],
      ##          data_future = dpred_val[,,,idx,drop=F])
      ## Objective_val[i/100] <- run(Cost, feed_dict = fd)
      ## thetavals <- round(colMeans(run(theta_tf, feed_dict = fd)), 4)
      cat(paste0("Step ", i, " ... Cost: ", Objective[i], "\n"))
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
