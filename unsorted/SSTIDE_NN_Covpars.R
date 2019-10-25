###################################################
## Title: IDE-CNN for prediction of SST
## Date: 17 May 2019
## Author: Andrew Zammit-Mangion
###################################################

############################################
## Part 1: Load Packages and other functions
############################################
library(IDE)
library(FRK)
library(sp)
library(tensorflow)
library(dplyr)
library(fields)
library(ggquiver)
library(ggplot2)
library(gridExtra)
library(gstat)
library(lubridate)
library(Matrix)
library(tidyr)

rm(list=ls())
set.seed(1)
source("./netarchitectures.R")
source("./utils_tf.R")
source("./utils.R")
train_net <- TRUE

## Helper Functions
run <- tf$Session()$run
init <- tf$global_variables_initializer()

tfconst <- function(x, name = NULL) tf$constant(x, dtype = "float32", name = name)
tfVar <- function(x, name = NULL) tf$Variable(x, dtype = "float32", name = name)

######################################################
## Part 2: Load data and massage into required formats
######################################################
#load("./SSTdata/TrainingData3D.rda")
#load("./SSTdata/TrainingDataFinals.rda")
#load("./SSTdata/TrainingDataPreds.rda")
load("~/cache/TrainingData3D.rda")
load("~/cache/TrainingDataFinals.rda")
load("~/cache/TrainingDataPreds.rda")

tau <- 3L                   # lags to consider
patch_size_flow <- 5L       # kernel size for flow convs
patch_size_diff <- 5L       # kernel size for diffusion convs
nT <- 4456L                 # total number of time points
nT_Train_Val <- 4000L       # time points for training + validatino
nT_Test <- 4456L - 4000L    # time points for testing
nZones <- dim(d)[1] / nT    # number of spatial zones
W <- dim(d)[2]              # 64
H <- dim(d)[3]              # 64
N_Channels <- dim(d)[4]     # 3 channels
N_Data <- dim(d)[1]         # total number of data points (nT * nZones)
idxTrain_Val <- rep(1:nT_Train_Val, nZones) +      # indices of train/val data
    rep((0:(nZones-1)*nT), each = nT_Train_Val)
idxTest <- setdiff(1:N_Data, idxTrain_Val)         # indices of test data
N_Data_Train <- round(length(idxTrain_Val) * 9 / 10) # number of training data
N_Data_Val <- round(length(idxTrain_Val) * 1 / 10)   # number of val data
N_Filters <- 64L         # number of convolution filters in first layer
N_Batch <- 16L           # minibatch size in SGD
sqrtN_Basis <- 8L        # sqrt of number of basis functions for IDE vector fields
s1 <- seq(0, 1, length.out = W)  # grid points for s1
s2 <- seq(0, 1, length.out = H)  # grid points for s2
ds1 <- mean(diff(s1))            # s1 spacings
ds2 <- mean(diff(s2))            # s2 spacings
ds <- sqrt(ds1 * ds2)            # area of pixel
sgrid <- expand.grid(s1 = s1, s2 = s2)  # spatial grid in long format
maskidx <- which(sgrid$s1 > 0.1 & sgrid$s1 < 0.9 &  # we train on CNN only ..
                 sgrid$s2 > 0.1 & sgrid$s2 < 0.9)   # inside this square box ..
mask <- rep(1, W*H)                                 # to avoid boundary effects
mask[-maskidx] <- 0

## Index notes: Note that first day in each zone is the day 2006-12-27
##   We index by first day of the 4-day (tau = 3) batch which is 2006-12-27
##   We have 4000 test/val data points (incl. 12/27), i.e., up to 2017-12-08
##   We have 456 test data points, i.e., up to 2019-03-09
##   Note that this includes data up to 2019-03-12 (since 2019-03-09 contains 4 images,
##   three used for analysis and one for forecast/traning)

## For the "current day", we store the third image, i.e., the first record
## in this data set is of 2006-12-29, and goes up to 2019-03-11
## For the "future day", we store the fourth image, i..e, the first record
## in this data set is of 2006-12-30, and goes up to 2019-03-12.

Date_Zone_Map <- tibble(idx = 1:N_Data,
                        t = rep(1:nT, nZones),
                        zone = rep(1:nZones, each = nT)) %>%
    left_join(tibble(t = 1:nT,
                     startdate = as.Date("2006-12-26") + 1:nT,
                     currentdate = startdate + 2,
                     futuredate = startdate + 3))

                            

##################################
## Part 3: Set up TensorFlow graph
##################################

### 3.1: Data placeholders

## An input "data point" consisting of N_Batches of three frames 
data_in <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H, N_Channels))

## The final data point of data_in (for N_Batches)
data_current <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))

## The future data point for data_in (for N_Batches)
data_future <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))

## Number of batches
thisN_Batch <- tf$shape(data_in)[1]

### 3.2: Kernel setup
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
PHI2_tf <- tf$expand_dims(PHI_tf, 0L)  # make (1 x 4096 x 64)      
PHI3_tf <- tf$tile(PHI2_tf, c(thisN_Batch, 1L, 1L)) # tile for N_Batches

### 3.3: CNN architecture for the IDE parameters

### 3.3.1: CNN for the horizontal and vertical parameters

## Construct convnet that takes in the data (with N_Batches) and outputs
## a matrix of dimension N_Batches x 128 (the first N_Batches x 64 are for u
## while the second N_Batches x 64 are for v)
nnet_flow <- ConvIDEnetsemiV2(data_in = data_in, 
                              N_Channels = N_Channels, 
                              N_Filters = N_Filters, 
                              sqrtN_Basis = sqrtN_Basis,
                              kernel_size = patch_size_flow)

## Make size N_Batches x 128 x 1
flow_coef <- tf$expand_dims(nnet_flow$flow_coef, 2L)

## Multiply PHI with weights to get images of u and v
u_long <- tf$matmul(PHI3_tf, flow_coef[, 1:(sqrtN_Basis^2),])
v_long <- tf$matmul(PHI3_tf, flow_coef[, sqrtN_Basis^2 + 1:(sqrtN_Basis^2),])

### 3.3.2: CNN for the diffusion coefficients

## Construct convnet that takes in the data (with N_Batches) and outputs
## a matrix of dimension N_Batches x 64 
nnet_D <- ConvIDEnet_diffusion(data_in = data_in, 
                               N_Channels = N_Channels, 
                               N_Filters = N_Filters, 
                               sqrtN_Basis = sqrtN_Basis,
                               kernel_size = patch_size_diff)

## Make size N_Batches x 64 x 1
D_coef <- tf$expand_dims(nnet_D$D_coef, 2L)

## Multiply PHI with weights to get image of D
D_long <- tf$matmul(PHI3_tf, D_coef[, 1:(sqrtN_Basis^2),])


### 3.3.3: The top layer

w <- tf$concat(list(u_long, v_long), axis = 2L)  # Put into size (N_Batches x 4096 x 2)
sgridlong_tf <- tf$expand_dims(sgrid_tf, 0L) # Spatial grid as (1 x 4096 x 2)
K_IDE <- k_tf2(s = sgridlong_tf,             # Construct kernel which is ..
               r = sgridlong_tf,             # of size N_Batches x 4096 x 4096
               w = w,
               D = D_long)

## K_IDE_Long is in the right format -- we now just need
## to just multiply it by data_current for prediction
## But we need to make the 64 x 64 data into 4096 x 1 format
## We cannot just use tf$reshape, as the reordering is different than
## what R does when reshaping! We therefore have another function 
## reshape_3d_to_2d() to do the reordering.
## E.g., tf$reshape(data_current, c(-1L, W*H, 1L)) would be wrong
data_current_long <- reshape_3d_to_2d(data_current, W*H)
data_future_long <- reshape_3d_to_2d(data_future, W*H)
Ypred <- tf$matmul(K_IDE, data_current_long)

## Now we mask when comparing, to reduce boundary effects
mask_tf <- tfconst(array(mask, dim = c(1L, W*H, 1L)))
mask_tf <- tf$tile(mask_tf, c(thisN_Batch, 1L, 1L))
Ypred_masked <- tf$multiply(Ypred, mask_tf)
data_future_masked <- tf$multiply(data_future_long, mask_tf)

## The basic cost function is then just the mean squared error 
Cost1 <- tf$reduce_mean(tf$square(data_future_masked - Ypred_masked))

### 3.3.4: The top layer with Matern covariance function

## Create the variables for the Matern
## log_sigma2 <- tfVar(log(0.4), "log_sigma2")
## log_rho <- tfVar(log(0.1), "log_rho")
log_sigma2_nnet <- ConvIDEnet_sigmapars(data_in = data_in, 
                               N_Channels = N_Channels, 
                               N_Filters = 16L,
                               kernel_size = 5L)
log_sigma2 <- log_sigma2_nnet$LogCovPar
sigma2 <- tf$exp(log_sigma2)

log_rho_nnet <- ConvIDEnet_sigmapars(data_in = data_in, 
                               N_Channels = N_Channels, 
                               N_Filters = 16L,
                               kernel_size = 5L)
log_rho <- log_rho_nnet$LogCovPar
rho <- tf$exp(log_rho)

## Create the distance matrix and covariance matrix
d1 <- tf$square(sgrid_tf[, 1] - sgrid_tf[, 1, drop = FALSE])
d2 <- tf$square(sgrid_tf[, 2] - sgrid_tf[, 2, drop = FALSE])
D <- tf$sqrt(d1 + d2)
SIGMA <- Matern32(sigma2, rho, D)

## Now create a mask matrix of size 2500 x 4096, so that only the 
## 2500 elements not masked out are returned when premultiplying
## So this is a bit differen from mask_tf which just zeroed out
maskmat <- matrix(0, sum(mask), W*H)
for(i in seq_along(maskidx)) {
  maskmat[i,maskidx[i]] <- 1
}
maskmat_tf <- tfconst(maskmat)

## Expand it for the number of batches we have
maskmat_tiled <- tf$expand_dims(maskmat_tf, 0L)
maskmat_tiled <- tf$tile(maskmat_tiled, c(thisN_Batch, 1L, 1L))

## Now extract the subset of covariance matrix we are interested in
SIGMA_sub <- tf$linalg$matmul(tf$linalg$matmul(maskmat_tiled, SIGMA), tf$linalg$transpose(maskmat_tiled))

## And compute the likelihood based on this
## Compute the upper Cholesky and tile it for all the batches
R_tiled <- tf$linalg$transpose(tf$linalg$cholesky(SIGMA_sub))
Rinv_tiled <- tf$linalg$inv(R_tiled)

## Extract the part of data and the predictions that we need (all batches)
data_future_sub <- tf$linalg$matmul(maskmat_tiled, data_future_long)
Ypred_sub <- tf$linalg$matmul(maskmat_tiled, Ypred) 

## Find the difference Ytilde^T
Ydiff <- tf$linalg$transpose(data_future_sub - Ypred_sub)

## Compute log determinants
logdet_part <- tf$reduce_sum( -0.5 *logdet_tf(R_tiled))

## Compute quadratic part
YdiffRinv <- tf$matmul(Ydiff, Rinv_tiled)
squared_part_Batch <- -0.5 * tf$linalg$matmul(YdiffRinv, tf$linalg$transpose(YdiffRinv))
squared_part <- tf$reduce_sum(squared_part_Batch)

## Negative log-likelihood
Cost2 <- -(logdet_part + squared_part)

#############################
## Part 4: Training the graph
#############################
set.seed(1)                                         # set seed
nepochs <- 30                                       # 30 epochs
nsteps_per_epoch <- floor(N_Data_Train / N_Batch)   # number of steps per epoch (4275)
init_learning_rate <- 0.00005                       # learning rate for CNN weights
init_learning_rate_cov <- 0.00001#0.0000005                 # learning rate for cov pars

## Optimiser for IID residuals
trainnet1 <- tf$train$AdamOptimizer(init_learning_rate)$minimize(Cost1)

## Optimiser for correlated residuals (currently unused)
trainnet2 <- tf$train$AdamOptimizer(init_learning_rate)$minimize(Cost2)

## Optimiser for covariance function parameters
trainnetcov <- (tf$train$GradientDescentOptimizer(init_learning_rate_cov))$
    minimize(Cost2, var_list = c(log_sigma2_nnet$wts,
                                 log_rho_nnet$wts))

## Store cost function values in these lists
Objective <- Objective_val <- list()  

## Store summary stats of each epoch in this data frame
Epoch_train <- Epoch_val <- data.frame(mean = rep(0, nepochs),
                                       median = rep(0, nepochs),
                                       sd = rep(0, nepochs),
                                       mad = rep(0, nepochs))

## Split the non-test data into training and validation
idxTrain <- sample(idxTrain_Val, N_Data_Train)
idxVal <- setdiff(idxTrain_Val, idxTrain)

## Initialise global variables
init <- tf$global_variables_initializer()
run(init)
options(scipen = 3, digits = 3)
load(file = "~/cache/SSTIDE_weights_TF.rda")
All_Vars_tf <- tf$trainable_variables(scope = NULL)
for(i in 1:1:10) {
    run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
}

stop()
## Train the net?
if(train_net)
  
  ## For each epoch
  for(epoch in 1:nepochs) {
    
    ## Do all validation cases
    cat("Running validation... \n")
    N_Val_Batches <- floor(length(idxVal)/32L)            # about 237
    Val_Batch_idx <- rep(1:N_Val_Batches, each = 32L)     # batch ID
    if((partbatch <- length(idxVal) %% 32L) > 0) {        # if not an exact multiple of 32
        N_Val_Batches <- N_Val_Batches + 1           # then there is an extra batch
        Val_Batch_idx <- c(Val_Batch_idx, rep(N_Val_Batches, partbatch))
    }
    
    ## Create an idx--batch map
    val_batches <- data.frame(idx = idxVal,
                              batch = Val_Batch_idx)
    
    ## For each batch compute the NLL
    ValCosts <- lapply(unique(val_batches$batch), function(i) {
            idx_val <- filter(val_batches, batch == i)$idx
            fd <- dict(data_in = d[idx_val,,,],
                       data_current = dfinal[idx_val,,,drop=F],
                       data_future = dpred[idx_val,,,drop=F])
            run(Cost1, feed_dict = fd)
    })
    
    ## Compute summary statistics at this epoch
    Epoch_val[epoch, ] <- c(mean(unlist(ValCosts)),
                            median(unlist(ValCosts)),
                            sd(unlist(ValCosts)),
                            mad(unlist(ValCosts)))
    
    ## Initialise
    epoch_order <- idxTrain
    count <- 0
    Objective[[epoch]] <- Objective_val[[epoch]] <- rep(0, nsteps_per_epoch)
    
    while(length(epoch_order) >= N_Batch) {
    
        train_cov_pars <- (epoch >= 1L)#(28L))       # Estimate cov. pars if epoch >= 28
        count <- count + 1                       # increment batch number in epoch
        idx <- epoch_order[1:N_Batch]            # Take first N_Batch
        epoch_order <- epoch_order[-(1:N_Batch)] # Remove first batch from list
        fd <- dict(data_in = d[idx,,,],          # Creat dictionary
                   data_current = dfinal[idx,,,drop=F],
                   data_future = dpred[idx,,,drop=F])
        
        if(train_cov_pars) {
            run(trainnetcov, feed_dict = fd)
          #} else {
            ## Train everything together between 21 and 30  
          #  run(trainnet2, feed_dict = fd)
          #}
          # if((count %% 10) == 0) cat(paste0("rho = ", run(rho), "\n"))  
        } else {
          ## Train network on its own  
          run(trainnet1, feed_dict = fd)
        }
        
        ## Get Cost for this batch
        Objective[[epoch]][count] <- run(Cost1, feed_dict = fd)
        
        ## Every 10 samples do a random validation check (not sure if needed)
        if((count %% 10) == 0) {
            idx_val <- sample(idxVal, N_Batch, replace = FALSE)
            fd <- dict(data_in = d[idx_val,,,],
                       data_current = dfinal[idx_val,,,drop=F],
                       data_future = dpred[idx_val,,,drop=F])
            Objective_val[[epoch]][count/10] <- run(Cost1, feed_dict = fd)
            cat(paste0("Epoch ", epoch, " ...  Step ", count, 
                       " ... Cost: ", Objective_val[[epoch]][count/10], "\n"))
        }

        ## Every 1000 iterations do some plots
        if((count %% 1000) == 0) {
            png("Temp.png")
            par(mfrow = c(2,2))
            
            ## Extract all Training and Validation costs to date
            if(epoch > 1) {
              TrainCosts <- c(unlist(Objective[1:(epoch-1)]),
                              Objective[[epoch]][1:count])
              ValCosts <- setdiff(c(unlist(Objective_val[1:(epoch-1)]),
                        Objective_val[[epoch]][1:count]), 0)
            } else {
              TrainCosts <-  Objective[[1]][1:count]
              ValCosts <- Objective_val[[1]][1:count]
            }

            ## Plot the costs
            plot(TrainCosts)
            XX <- data.frame(idx = 1:length(TrainCosts), O = TrainCosts)
            Osmooth <- predict(loess(O ~ idx, data = XX, span = 0.25))
            plot(Osmooth)
         
            plot(ValCosts)
            XX <- data.frame(idx = 1:length(ValCosts), O = ValCosts)
            Osmooth <- predict(loess(O~idx, data = XX, span = 0.25))
            plot(Osmooth)
            dev.off()

        
            ## Plot some figures showing predictions
            myscale <- scale_fill_gradientn(colours = nasa_palette, name = "Z")
            ii <- 1
            sgrid$Ytest1 <- as.numeric(run(data_in[ii,,,1], feed_dict = fd))
            sgrid$Ytest2 <- as.numeric(run(data_in[ii,,,2], feed_dict = fd))
            sgrid$Ytest3 <- as.numeric(run(data_in[ii,,,3], feed_dict = fd))
            sgrid$Ytest4 <- as.numeric(run(data_future_long[ii,,], feed_dict = fd))
            sgrid$Ytest5 <- as.numeric(run(data_current_long[ii,,], feed_dict = fd))
            sgrid$Ytest6 <- as.numeric(run(Ypred[ii,,], feed_dict = fd))
            ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Ytest1)) + myscale + 
                     coord_fixed(), file = "Test1.png", width = 8, height = 8)
            ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Ytest2)) + myscale + 
                     coord_fixed(), file = "Test2.png", width = 8, height = 8)
            ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Ytest3)) + myscale + 
                     coord_fixed(), file = "Test3.png", width = 8, height = 8)
            ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Ytest4)) + myscale + 
                     coord_fixed(), file = "Test4.png", width = 8, height = 8)
            ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Ytest5)) + myscale + 
                     coord_fixed(), file = "Test5.png", width = 8, height = 8)
            ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Ytest6)) + myscale + 
                     coord_fixed(), file = "Test6.png", width = 8, height = 8)
        }
    }
    Epoch_train[epoch, ] <- c(mean(Objective[[epoch]]),
                              median(Objective[[epoch]]),
                              sd(Objective[[epoch]]),
                              mad(Objective[[epoch]]))
}

## If we have trained neural net, then save all variables, otherwise
## load and assign
if(train_net) {
  Trained_Vars <- run(tf$trainable_variables(scope = NULL))
  save(Trained_Vars, file = "~/cache/SSTIDE_weights_TF.rda")
} else {
  load(file = "~/cache/SSTIDE_weights_TF.rda")
  All_Vars_tf <- tf$trainable_variables(scope = NULL)
  for(i in 1:length(All_Vars_tf)) {
      run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
   }
}

# XX <- expand.grid(s1 = 1:5, s2 = 1:5, dim = 1:3, flt = 1:64)
# XX$val <- as.numeric(run(All_Vars_tf[[1]]))
# ggplot(filter(XX, flt  < 21)) + geom_tile(aes(s1, s2, fill = val)) +
#   facet_grid(dim ~ flt) + scale_fill_distiller(palette = "Spectral")

stop()

######################################################
## Part 5: Show filter outputs on moving ball (works?)
######################################################

convfilter_u <- tf$trainable_variables(scope = NULL)[[1]]
convfilter_v <- tf$transpose(convfilter_u, perm = c(1L, 0L, 2L, 3L))
conv1 <- conv(data_in, 3L, N_Filters, convwts = convfilter_v) # 32 x 32 x 16
set.seed(5)
for(i in 1:10) {
  s1c <- runif(1, min = 0.25, max = 0.75)
  s2c <- runif(1, min = 0.25, max = 0.75)
  ds1 <- runif(1, max = 0.1) - 0.05
  ds2 <- runif(1, max = 0.1) - 0.05
  dD <- 1 #runif(1, min = 1.2, max = 2.5)
  
  # ball <- mutate(sgrid, z1 = exp(-((s1-s1c)^2 + (s2 - s2c)^2)/0.001),
  #                       z2 = exp(-((s1-s1c - ds1)^2 + (s2 - s2c - ds2)^2)/(0.001*dD)),
  #                       z3 = exp(-((s1-s1c - 2*ds1)^2 + (s2- s2c- 2*ds2)^2)/0.001*dD))
  
  ball <- mutate(sgrid, 
                 z1 = dnorm(s1, s1c, sqrt(0.001))*dnorm(s2, s2c, sqrt(0.001)),
                 z2 = dnorm(s1, s1c - ds1, sqrt(0.001*(dD^2)))*
                      dnorm(s2, s2c-ds2, sqrt(0.001*(dD^2)))/max(z1),
                 z3 = dnorm(s1, s1c - 2*ds1, sqrt(0.001*(dD^2)))*
                      dnorm(s2, s2c-2*ds2, sqrt(0.001*(dD^2)))/max(z1),
                 z1 = z1 / max(z1))
  
  ball_im <- array(c(ball$z1, ball$z2, ball$z3), dim = c(1, 64, 64, 3))
  #fd <- dict(data_in = d[1,,,, drop = FALSE])
  fd <- dict(data_in = ball_im)
  #XX <- run(conv1, feed_dict = fd)
  #filterout <- expand.grid(s1 = 1:32, s2 = 1:32, channel = 1:64) %>%
  #             mutate(value = as.numeric(XX))
  #ggplot(filterout) + geom_tile(aes(s1, s2, fill = value)) + facet_wrap(~channel) +
  #    scale_fill_gradientn(colours = nasa_palette)
  
  sgrid$U <- as.numeric(run(u_long, feed_dict = fd))
  sgrid$V <- as.numeric(run(v_long, feed_dict = fd))
  sgrid$D <- as.numeric(run(D_long, feed_dict = fd))
  s1sub <- unique(sgrid$s1)[seq(1,64,by = 4)]
  s2sub <- unique(sgrid$s2)[seq(1,64,by = 4)]
  g1 <- ggplot(ball) + geom_contour(aes(s1, s2, z = z1), alpha = 0.2, colour = "black") +
    geom_contour(aes(s1, s2, z = z2), alpha = 0.5, colour = "black") +
    geom_contour(aes(s1, s2, z = z3), colour = "black") + theme_bw() +
    coord_fixed(xlim = c(0,1), ylim = c(0,1))
  g2 <- ggplot(filter(sgrid, s1 %in% s1sub & s2 %in% s2sub)) + 
    geom_contour(data = ball, aes(s1, s2, z = z1), alpha = 0.2, colour = "red") +
    geom_contour(data = ball, aes(s1, s2, z = z2), alpha = 0.5, colour = "red") +
    geom_contour(data = ball, aes(s1, s2, z = z3), colour = "red") +
    geom_quiver(aes(s1, s2, u = -U, v = -V,# colour = atan2(-V, -U), 
                    colour = (sqrt(U^2 + V^2))), vecsize = 1) + 
    scale_colour_distiller(palette = "Greys", trans = "reverse", name = "Amplitude") +
    theme_bw() + coord_fixed() + xlab(expression(s[1])) + 
    ylab(expression(s[2])) + theme(text = element_text(size = 18)) +
    #scale_colour_distiller(name = "Angle (deg)") +
    theme(legend.position = NULL)
  g3 <- ggplot(sgrid) + geom_tile(aes(s1, s2, fill = D)) + 
      scale_fill_gradientn(colours = nasa_palette) + coord_fixed() +
    theme_bw() 
    
  #gall <- grid.arrange(g1,g2,g3, nrow = 1, widths = c(0.3, 0.3, 0.37))
  #gall <- grid.arrange(g1,g2, nrow = 1, widths = c(0.45, 0.55))
  ggsave(g2, file = paste0("Results/BallResults",i,".png"), width = 8, height = 6)
}
  
#############################################
## Part 6: Running the ensemble Kalman filter
#############################################

### 6.1: Initialise

nParticles <- 64L               # 32 particles
nObs <- 1024L                   # 1024 observations
sigma2e <- 0.01                 # measurement-error variance

### 6.2: Set up  EnKF graph for doing EnKF on GPU

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
Taper <- tfconst(Wendland1_R(0.2, run(D)))
Ypred_COV <- tf$multiply(Ypred_COV_untapered, Taper)

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
Z_tf <- tf$placeholder(dtype = "float32", shape = list(nObs, 1L))  # tile the observations
Z_tf_tiled  <- tf$expand_dims(Z_tf, 0L)
Z_tf_tiled  <- tf$tile(Z_tf_tiled, c(thisN_Batch, 1L, 1L))
normvars_obs <- tf$placeholder(dtype = "float32", shape = c(nParticles, nObs, 1L))
Zsim_tf <- tf$linalg$matmul(C_tf_tiled, Ypred_noisy) +  # simulate the observations
    sqrt(sigma2e)*normvars_obs
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

## warning("Just doing 2 zones")
## for(zone in 1:2) {

for(zone in 1:nZones) {

### 6.3 Generate data in each zone
    
    set.seed(zone)                  # set seed
    C <- z_df_list <- Z <- list()   # list of data and obs. matrices
    results <- list()               # list of results
    
    ## Set time axis
    test_zones_dates <- filter(Date_Zone_Map,
                               year(startdate) == "2018" &
                               month(startdate) %in% 9:12)
    this_zone <- zone
    taxis_df <- filter(test_zones_dates, zone == this_zone)
    taxis <- taxis_df$idx

    ## Simulate data with measurement error
    ## Store data in long format (Z) and data frame (z_df_list)
    for(i in seq_along(taxis)) {
       obsidx <- sample(W*H, nObs)     # sample the observations
       C[[i]] <- sparseMatrix(i = 1:nObs,   # measurement-mapping matrix
                              j = obsidx, 
                              x = 1, 
                              dims = c(nObs, W*H))
        Z[[i]] <- as.numeric(C[[i]] %*% c(dfinal[taxis[i],,]) + rnorm(nObs, 0, sqrt(sigma2e)))
        z_df_list[[i]] <- cbind(sgrid[obsidx,], z = Z[[i]], t = i)
    }

    ## Collapse z_df_list into one long data frame
    z_df <- Reduce("rbind", z_df_list)
    #ggplot(z_df) + geom_point(aes(s1, s2, colour = z)) + facet_wrap(~t) +
    #    scale_colour_gradientn(colours = nasa_palette)

     
### 6.4: Initialise the first particles and run the EnkF for each zone in test set

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

    ## For each time point
    for(i in seq_along(taxis)) {

        cat(paste0("Filtering Zone ", zone, " Time point ", i, "\n"))

        ## Ignore the first three time points (init. condition)
        if(i > tau) {

            ## Update using the current Pframe        
            fd <- dict(data_in = Pframe,
                       data_current = Pframe[,,,tau],
                       data_previous = Pframe[,,,tau - 1],
                       C_tf = as.matrix(C[[i]]),
                       Z_tf = as.matrix(Z[[i]]),
                       normvars_process = array(rnorm(W*H*nParticles), dim = c(nParticles, W*H, 1L)),
                       normvars_obs = array(rnorm(nObs*nParticles), dim = c(nParticles, nObs, 1L)))

            ## Extract forecast and updated ensemble
            ParticleUpdates <- run(list(Ypred_noisy, Ypred_updated, Ysmooth1, Ysmooth2),
                                   feed_dict = fd)
            ParticleForecasts <- ParticleUpdates[[1]]
            ParticlePreds <- ParticleUpdates[[2]]
            ParticleSmooth1 <- ParticleUpdates[[3]]
            ParticleSmooth2 <- ParticleUpdates[[4]]
            
            results[[i]] <-
                data.frame(filter_mu = apply(ParticlePreds, 2, mean),
                           filter_sd = apply(ParticlePreds, 2, sd),
                           fcast_mu = apply(ParticleForecasts, 2, mean),
                           fcast_sd = apply(ParticleForecasts, 2, sd),
                           truth = as.numeric(dfinal[taxis[i],,]),
                           zone = zone,
                           date = taxis_df$currentdate[i])
                                       
            ## Shift the particles backwards to produce a new `Pframe'
            for(j in 1:(tau - 1)) {
                Pframe[,,,j] <- Pframe[,,,j + 1]    
            }

            ##New particles (filtered ones) at tau
            Pframe[,,,tau] <-  array(c(ParticlePreds),
                                     c(nParticles, W, H, 1))

            ## Smoothed particles at tau - 1
            Pframe[,,,tau - 1] <-  array(c(ParticleSmooth1),
                                    c(nParticles, W, H, 1))

            ## Smoothed particles at tau - 2
            Pframe[,,,tau - 2] <-  array(c(ParticleSmooth2),
                                    c(nParticles, W, H, 1))

            if(zone == 11 & i %in% 10:12)
                save(Pframe, ParticlePreds, ParticleForecasts,
                     nParticles,
                     file = paste0("data_for_dir_plots_Zone11_i", i, ".rda"))

            if(i == (tau + 1)) {
                u_pars <- run(flow_coef, feed_dict = fd)[, 1:sqrtN_Basis^2, ] %>% apply(2, mean)
                v_pars <- run(flow_coef, feed_dict = fd)[, -(1:sqrtN_Basis^2), ] %>% apply(2, mean)
                Dpars <- run(D_coef, feed_dict = fd) %>% apply(2, mean)
                save(u_pars, v_pars, Dpars, file = paste0("Results/KernelInit/kinit_zone", zone, ".rda"))
            }
                

        }
    }

    all_data <- list(Z = Z, C = C, sgrid = sgrid, zone = zone, taxis_df = taxis_df)
    save(all_data, results, sigma2e, taxis_df, file = paste0("Results/Results_CNNIDE_Zone_", zone, ".rda"))
}

## Plotting
par(mfrow = c(1,3))
XX <- run(Ypred_mean, feed_dict = fd)
image.plot(matrix(XX, 64, 64))
YY <- apply(run(Ypred_updated, feed_dict = fd),2,mean)
image.plot(matrix(YY, 64, 64))
ZZ <- dfinal[taxis[i],,]
image.plot(matrix(ZZ, 64, 64))

hist(run(Z_tf_tiled[1,,] - Zsim_tf[1,,], feed_dict = fd))


myscale <- scale_fill_gradientn(colours = nasa_palette, name = "Z")
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = as.numeric(Pframe[1,,,1]))) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = as.numeric(Pframe[1,,,2]))) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = as.numeric(Pframe[1,,,3]))) + myscale
ggplot(sgrid) + geom_tile(aes(s2, s1, fill = ParticlePreds[10,,])) + myscale

pred_mu <- apply(ParticlePreds, 2, mean)
ggplot(sgrid) + geom_tile(aes(s2, s1, fill = pred_mu)) + myscale
ggplot(sgrid) + geom_tile(aes(s1, s2, fill = c(dfinal[taxis[4],,]))) + myscale


## global_step = tf$Variable(0, trainable = TRUE)      
## lr <- tf$train$exponential_decay(learning_rate = init_learning_rate,
##                                   global_step = global_step,
##                                   decay_steps = nsteps_per_epoch * round(nepochs/2),
##                                   decay_rate = 0.96,
##                                   staircase = TRUE)

####################################################################
## Part 7: Show uncertainty in the dynamics (forecast and predicted)
####################################################################

## Replace updated with forecasts to see effect of update on
## dynamics
for(fname in c("data_for_dir_plots_Zone11_i10.rda",
               "data_for_dir_plots_Zone11_i11.rda",
               "data_for_dir_plots_Zone11_i12.rda")) {

    load(fname)
    print(sum(c(ParticleForecasts) - c(Pframe[,,,tau])))
    print(sum(c(ParticlePreds) - c(Pframe[,,,tau])))
    Pframe2 <- Pframe
    Pframe2[,,,tau] <- array(c(ParticleForecasts),
                             dim = c(nParticles, 64L, 64L))

    fd1 <- dict(data_in = Pframe)
    fd2 <- dict(data_in = Pframe2)

    ## Extract forecast and updated ensemble
    U1 <- -run(u_long, feed_dict = fd1)
    V1 <- -run(v_long, feed_dict = fd1)
    angles_df1 <- as.data.frame(t((atan2(V1,U1)*360/(2*pi))[,,1])) %>%
        cbind(sgrid) %>%
        gather(Sim, Angle, -s1, -s2) %>%
        mutate(Angle = ifelse(Angle < 0, Angle + 360, Angle),
               Group = "Updated")

    U2 <- -run(u_long, feed_dict = fd2)
    V2 <- -run(v_long, feed_dict = fd2)
    angles_df2 <- as.data.frame(t((atan2(V2,U2)*360/(2*pi))[,,1])) %>%
        cbind(sgrid) %>%
        gather(Sim, Angle, -s1, -s2) %>%
        mutate(Angle = ifelse(Angle < 0, Angle + 360, Angle),
               Group = "Forecast")

    angles_df <- rbind(angles_df1, angles_df2)
    angles_df$Group <- as.factor(angles_df$Group)

    s1sub <- unique(sgrid$s1)[seq(8, 64, by = 16)]
    s2sub <- unique(sgrid$s2)[seq(8, 64, by = 16)]
    sgrid_sub <- expand.grid(s1sub, s2sub)
    gempty <- ggplot(data.frame(x = -40, y = 42)) +
        geom_point(aes(x,y), col = "white") +
        coord_fixed(xlim = c(-43.333, -38.0833),
                    ylim = c(40.333 ,45.5833)) + theme_bw() +
        theme(text = element_text(size=20)) +
            xlab("Longitude (deg)") + ylab("Latitude (deg)")

    gall <- gempty
    for(i in 1:nrow(sgrid_sub)) {

        dszone <- (43.333 - 38.083)
        angles_df_sub <- filter(angles_df, s1 == sgrid_sub[i,1] &
                                           s2 == sgrid_sub[i,2]) %>%
            mutate(s1 = s1*dszone - 43.333,
                   s2 = s2*dszone + 40.333)

        ## gi <- ggplot(angles_df_sub, aes(x = Angle, fill = Group,
        ##                                 colour = Group)) +
        ##     geom_histogram(colour = "grey", binwidth = 30, alpha = 0.5,
        ##                    position = "identity") + 
        ##     coord_polar(start = 0) + theme_minimal() + ylab("") + 
        ##     scale_x_continuous("", limits = c(0, 360),
        ##                        breaks = seq(0, 360, by = 45)) +
        ##     ylim(0, 33) +
        ##     theme(text = element_blank(),
        ##           title = element_blank(),
        ##           legend.position = "none")

        gi <- ggplot() +
            geom_histogram(data = filter(angles_df_sub, Group == "Updated"),
                           aes(x = Angle), colour = "black", breaks = seq(0,360, by = 30),
                           alpha = 0.5,
                           position = "identity") +
            coord_polar(start = 0) + theme_minimal() + ylab("") + 
            scale_x_continuous("", limits = c(0, 360),
                               breaks = seq(0, 360, by = 45)) +
            theme(text = element_blank(),
                  title = element_blank(),
                  legend.position = "none") 

        ## gj <- ggplot() +
        ##     geom_histogram(data = filter(angles_df_sub, Group == "Forecast"),
        ##                    aes(x = Angle), fill = "Red", colour = "Red",
        ##                    binwidth = 30, alpha = 0.2,
        ##                    position = "identity") +
        ##     coord_polar(start = 0) + theme_minimal() + ylab("") + 
        ##     scale_x_continuous("", limits = c(0, 360),
        ##                        breaks = seq(0, 360, by = 45)) +
        ##     theme(text = element_blank(),
        ##           title = element_blank(),
        ##           line = element_blank(),
        ##           legend.position = "none")
        
        givp <- ggplotGrob(gi)
        #gjvp <- ggplotGrob(gj)
        gall <- gall + annotation_custom(grob = givp,
                                         xmin = angles_df_sub$s1[1] - dszone/9,
                                         xmax = angles_df_sub$s1[1] + dszone/9,
                                         ymin = angles_df_sub$s2[1] - dszone/9,
                                         ymax = angles_df_sub$s2[1] + dszone/9) 
    }
    ggsave(gall, file = paste0("./", tools::file_path_sans_ext(fname),".png"))
}

################################################################
## Part 8: Compare to true forecasts of SST on Wed Feb 27th 2019
################################################################
load("~/cache/TrainingData3D.rda")
load("~/cache/TrainingDataFinals.rda")
load("~/cache/TrainingDataPreds.rda")

initdate <- as.Date("2006-12-27")    # start date of first record
targetdate <- as.Date("2019-02-24")  # the "d" record containing 27/02 as fcast
dt <- targetdate - initdate
d_indices <- (dt + 1) + nT*(0:(nZones - 1)) # + 1 here is needed
DD <- d[d_indices,,,]
CMEMSPresent <- CNNForecasts <- CMEMSForecasts <- DD[,,,3]*NA
for(i in seq_along(d_indices)) {
    fd <- dict(data_in = DD[i,,,,drop = FALSE],
               data_current = DD[,,,3][i,,,drop=FALSE])
    CNNForecasts[i,,] <- matrix(as.numeric(
                          run(Ypred, feed_dict = fd)*
                          sds[d_indices[i]] + means[d_indices[i]]),
                          64, 64)
    CMEMSForecasts[i,,] <- dpred[d_indices[i],,]*
        sds[d_indices[i]] + means[d_indices[i]]
    CMEMSPresent[i,,] <- dpred[d_indices[i]-1,,]*
        sds[d_indices[i]-1] + means[d_indices[i]-1]
}

load("~/cache/TrainingData3D_forecast.rda")
load("~/cache/TrainingDataFinals_forecast.rda")
load(file = "~/cache/TrainingDataPreds_forecast.rda")
initdate2 <- as.Date("2019-02-04")    # start date of first "forecast" record
targetdate2 <- as.Date("2019-02-27")  # the "dpred" record containing 27/02 as fcast
dt2 <- targetdate2 - initdate2
nT2 <- dim(dpred)[1] / nZones
d_indices2 <- (dt2 + 1) + nT2*(0:(nZones - 1))
CMEMSAnalysis <- dpred[d_indices2,,]
err <- matrix(0, 19L, 3L)
for(i in seq_along(d_indices2)) {
  CMEMSAnalysis[i,,] <- CMEMSAnalysis[i,,]*
      sds[d_indices2[i]] + means[d_indices2[i]]
  Im1long <- as.numeric(CMEMSAnalysis[i,,])[maskidx]
  Im2long <- as.numeric(CMEMSForecasts[i,,])[maskidx]
  Im3long <- as.numeric(CNNForecasts[i,,])[maskidx]
  Im4long <- as.numeric(CMEMSPresent[i,,])[maskidx]
  err[i,1] <- sqrt(mean((Im1long - Im2long)^2))
  err[i,2] <- sqrt(mean((Im1long - Im3long)^2))
  err[i,3] <- sqrt(mean((Im1long - Im4long)^2))  
}

plot(1:19, err[,3], type = 'l',
     ylim = c(0, max(err)), ylab = "RMSPE",
     xlab = "Zone")
lines(1:19, err[,2], col = "red")
lines(1:19, err[,1], col = "green")

